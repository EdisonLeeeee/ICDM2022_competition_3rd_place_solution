import os
import os.path as osp
import copy
import argparse
import json
import random
import datetime


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric import seed_everything

from sklearn.metrics import average_precision_score

# custom modules
from logger import setup_logger
from model_sess1 import HeteroGNN

# 参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../data/session1/icdm2022_session1.pt")
parser.add_argument("--labeled-class", type=str, default="item")
parser.add_argument(
    "--fanout", type=int, default=150, help="Fan-out of neighbor sampling."
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1024,
    help="Mini-batch size. If -1, use full graph training.",
)
parser.add_argument(
    "--n-layers", type=int, default=2, help="number of propagation rounds"
)
parser.add_argument('--full', action='store_true')
parser.add_argument('--lp', action='store_true')
parser.add_argument(
    "--test-file", type=str, default="../data/session1/icdm2022_session1_test_ids.txt"
)
parser.add_argument("--record-file", type=str, default="session1_record.txt")
parser.add_argument("--model-file", type=str, default="model_sess1.pth")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seed", type=int, default=2022)

args = parser.parse_args()

logger = setup_logger(output=args.record_file, mode='a')

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


seed_everything(args.seed)


logger.info("Initializing VAL NeighborLoader...")

val_loader = NeighborLoader(
    hgraph_train,
    input_nodes=(labeled_class, val_idx),
    num_neighbors=[args.fanout] * args.n_layers,
    shuffle=False,
    batch_size=args.batch_size,
)

logger.info("VAL NeighborLoader Initialized.")

def to_device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d


@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f"Inference...")
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        y_emb = getattr(batch[labeled_class], 'y_emb', None)
        
        if y_emb is not None:        
            y_emb = y_emb.to(device)        
        y_hat = model(
            to_device(batch.x_dict, device),
            to_device(batch.edge_index_dict, device),
            y_emb,
        )[:batch_size]
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


@torch.no_grad()
def test():
    model.eval()
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f"Generate Final Result:")
    y_pred = []
    for batch in test_loader:
        batch_size = batch[labeled_class].batch_size
        y_emb = getattr(batch[labeled_class], 'y_emb', None)
        
        if y_emb is not None:        
            y_emb = y_emb.to(device)   

        y_hat = model(
            to_device(batch.x_dict, device),
            to_device(batch.edge_index_dict, device),
            y_emb,
        )[:batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()
    return torch.hstack(y_pred)

logger.info(f"Loading Best model {args.model_file}.")
model = torch.load(f"{args.model_file}").to(device)
logger.info(model)

val_loss, val_acc, val_ap = val()

logger.info(
    f"Val: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}"
)
    

test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(hgraph[labeled_class].maps[i])
test_idx = torch.LongTensor(converted_test_id)

logger.info(f"Initializing TEST loader...")

test_loader = NeighborLoader(
    hgraph_test,
    input_nodes=(labeled_class, test_idx),
    num_neighbors=[args.fanout] * args.n_layers,
    shuffle=False,
    batch_size=args.batch_size,
)

logger.info(f"TEST loader Initialized.")

y_pred = test()


logger.info("Writing TEST results...")

os.makedirs('../submit', exist_ok=True)
# write JSON file
json_file = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"

with open(json_file, "w") as f:
    for i in range(len(test_id)):
        y_dict = {}
        y_dict["item_id"] = int(test_id[i])
        y_dict["score"] = float(y_pred[i])
        json.dump(y_dict, f)
        f.write("\n")

logger.info(f"TEST results are saved at {json_file}.")

