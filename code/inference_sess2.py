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
from model_sess2 import HeteroGNN

# 参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../data/session2/icdm2022_session2.pt")
parser.add_argument("--labeled-class", type=str, default="item")
parser.add_argument(
    "--fanout", type=int, default=100, help="Fan-out of neighbor sampling."
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
parser.add_argument(
    "--test-file", type=str, default="../data/session2/icdm2022_session2_test_ids.txt"
)
parser.add_argument("--record-file", type=str, default="session2_record.txt")
parser.add_argument("--model-file", type=str, default="model_sess2.pth")
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

logger.info("=" * 70)
logger.info("Node features statistics")
for name, value in hgraph.x_dict.items():
    logger.info(f"Name: {name}, feature shape: {value.size()}")

logger.info("=" * 70)

logger.info("Edges statistics")
for name, value in hgraph.edge_index_dict.items():
    logger.info(f"Relation: {name}, edge shape: {value.size()}")
logger.info("=" * 70)

seed_everything(args.seed)

def to_device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d

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

test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(hgraph[labeled_class].maps[i])
test_idx = torch.LongTensor(converted_test_id)

logger.info(f"Initializing TEST loader...")

test_loader = NeighborLoader(
    hgraph,
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

