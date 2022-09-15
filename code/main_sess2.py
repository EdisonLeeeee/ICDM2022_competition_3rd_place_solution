import os
import os.path as osp
import copy
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from torch_geometric import seed_everything

from sklearn.metrics import average_precision_score

# custom modules
from logger import setup_logger
from metapath import drop_metapath
from model_sess2 import HeteroGNN


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../data/session1/icdm2022_session1.pt")
parser.add_argument("--labeled-class", type=str, default="item")
parser.add_argument(
    "--batch-size",
    type=int,
    default=1024,
    help="Mini-batch size. If -1, use full graph training.",
)
parser.add_argument(
    "--fanout", type=int, default=100, help="Fan-out of neighbor sampling."
)
parser.add_argument(
    "--n-layers", type=int, default=2, help="number of propagation rounds"
)
parser.add_argument("--h-dim", type=int, default=512, help="number of hidden units")
parser.add_argument("--in-dim", type=int, default=256, help="number of hidden units")
parser.add_argument("--early-stopping", type=int, default=100)
parser.add_argument('--full', action='store_true')
parser.add_argument('--lp', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument("--n-epoch", type=int, default=30)
parser.add_argument("--record-file", type=str, default="session2_record.txt")
parser.add_argument("--model-file", type=str, default="model_sess2.pth")
parser.add_argument("--resume-file", type=str, default="model_resume.pth")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seed", type=int, default=2022)

args = parser.parse_args()

logger = setup_logger(output=args.record_file)

logger.info(args)


if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

device = torch.device(device)

hgraph = torch.load(args.dataset)

labeled_class = args.labeled_class

train_idx = hgraph[labeled_class].pop("train_idx")
val_idx = hgraph[labeled_class].pop("val_idx")

if args.full:
    train_idx = torch.cat([train_idx, val_idx])

logger.info("=" * 70)
logger.info("Node features statistics")
for name, value in hgraph.x_dict.items():
    logger.info(f"Name: {name}, feature shape: {value.size()}")

logger.info("=" * 70)

logger.info("Edges statistics")
for name, value in hgraph.edge_index_dict.items():
    logger.info(f"Relation: {name}, edge shape: {value.size()}")
logger.info("=" * 70)

hgraph_train = copy.copy(hgraph)
hgraph_test = copy.copy(hgraph)

if args.lp:
    logger.info("Add labels for label propagation...")

    y = hgraph_train[labeled_class].y.clone()
    y[y == -1] = 2  # mask unknown nodes

    hgraph_test[labeled_class].y_emb = y.clone()

    y[val_idx] = 2  # mask validation nodes
    hgraph_train[labeled_class].y_emb = y.clone()
    del y

logger.info("Initializing NeighborLoader...")

seed_everything(args.seed)

# Mini-Batch
sampler = ImbalancedSampler(hgraph_train[labeled_class].y, input_nodes=train_idx)

train_loader = NeighborLoader(hgraph_train, input_nodes=(labeled_class, train_idx),
                              num_neighbors=[args.fanout] * args.n_layers, 
                              sampler=sampler, batch_size=args.batch_size)

val_loader = NeighborLoader(hgraph_train, input_nodes=(labeled_class, val_idx),
                            num_neighbors=[args.fanout] * args.n_layers,
                            shuffle=False, batch_size=args.batch_size)

logger.info("NeighborLoader Initialized.")

if args.resume:
    logger.info(f"Resume from {args.resume_file}")
    model = torch.load(f"{args.resume_file}").to(device)
else:
    model = HeteroGNN(metadata=hgraph.metadata(),
        hidden_channels=args.h_dim, out_channels=2, n_layers=args.n_layers
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

logger.info(model)
logger.info(optimizer)


def to_device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d


metapaths_to_drop = [
            [('f', 'a'), ('a', 'e')],
            [('f', 'e'), ('e', 'a')],
            [('f', 'item'), ('item', 'b')],
            [('b', 'item'), ('item', 'f')],
]

def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        
        batch = drop_metapath(batch, metapaths_to_drop, r=[0.5, 0.5])
        
        y_emb = getattr(batch[labeled_class], 'y_emb', None)
        if y_emb is not None:
            y_emb = y_emb.clone().to(device)
            y_emb[:batch_size] = 2 # mask current batch nodes
        
        y_hat, z_dict = model(to_device(batch.x_dict, device), 
                      to_device(batch.edge_index_dict, device), y_emb)
        y_hat = y_hat[:batch_size]
        loss = F.cross_entropy(y_hat, y) 

        for edge_type, pos_edge_index in batch.metapath_dict.items():
            if labeled_class not in edge_type: continue
            src, dst = edge_type
            row = torch.randint(0, batch[src].x.size(0), size=(pos_edge_index.size(1),))
            col = torch.randint(0, batch[dst].x.size(0), size=(pos_edge_index.size(1),))
            neg_edge_index = torch.stack([row, col], dim=0).to(pos_edge_index)
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat([torch.ones(pos_edge_index.size(1)),
                                    torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

            link_pred = model.edge_decoder(
                z_dict, edge_label_index, edge_type
            )
            link_target = edge_label
            loss_link = F.binary_cross_entropy_with_logits(link_pred, link_target)
            loss += 0.25 * loss_link              
            
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                

        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
    pbar.close()
    
    y_true = torch.hstack(y_true).numpy()
    y_pred = torch.hstack(y_pred).numpy()
    ap_score = average_precision_score(y_true, y_pred)

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        y_emb = getattr(batch[labeled_class], 'y_emb', None)
        if y_emb is not None:        
            y_emb = y_emb.to(device)
        
        y_hat = model(to_device(batch.x_dict, device), 
                      to_device(batch.edge_index_dict, device), y_emb)[:batch_size]
        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
    pbar.close()
    
    y_true = torch.hstack(y_true).numpy()
    y_pred = torch.hstack(y_pred).numpy()
    ap_score = average_precision_score(y_true, y_pred)

    return total_loss / total_examples, total_correct / total_examples, ap_score


val_ap_list = []
best_result = 0

logger.info("Start training")

for epoch in range(1, args.n_epoch + 1):
    train_loss, train_acc, train_ap = train(epoch)
    logger.info(
        f"Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}"
    )
    val_loss, val_acc, val_ap = val()
    val_ap_list.append(float(val_ap))

    if best_result < val_ap:
        best_result = val_ap
        torch.save(model, f"{args.model_file}")

    logger.info(
        f"Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}, Best AP: {best_result:.4f}"
    )

    if epoch >= args.early_stopping:
        ave_val_ap = np.average(val_ap_list)
        if val_ap <= ave_val_ap:
            logger.info(f"Early Stopping at {epoch}")
            break

