# 添加属性n_idx
# 将edge_index 转换成 torch.long
# 数据集划分信息

import argparse
from torch_geometric.data import HeteroData
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle as pkl

edge_size = 157814864
node_size = 13806619


def read_node_atts(node_file, pyg_file, label_file=None):
    node_maps = {}
    node_embeds = {}
    count = 0
    lack_num = {}
    node_counts = node_size
    if osp.exists(pyg_file + ".nodes.pyg") == False:
        process = tqdm(total=node_counts, desc="Generating " + pyg_file + ".nodes.pyg")
        with open(node_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0])
                node_type = info[1].strip()

                node_maps.setdefault(node_type, {})
                node_id_v2 = len(node_maps[node_type])
                node_maps[node_type][node_id] = node_id_v2

                node_embeds.setdefault(node_type, {})
                lack_num.setdefault(node_type, 0)
                # TODO:可以对不同的node_type缺失项采用不同的随机初始化方法
                if node_type == 'item':
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                else:
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)

                count += 1
                if count % 100000 == 0:
                    process.update(100000)

        process.close()

        print("Num of total nodes:", count)
        print('Node_types:', node_maps.keys())
        print('Node_type Totol_Num Lack_Num):')
        for node_type in node_maps:
            print(node_type, len(node_maps[node_type]), lack_num[node_type])

        labels = []
        if label_file is not None:
            labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
            for i in range(len(labels_info)):
                x = labels_info[i]
                item_id = node_maps['item'][int(x[0])]
                label = int(x[1])
                labels.append([item_id, label])

        nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
        nodes_dict['labels'] = {}
        nodes_dict['labels']['item'] = labels
        print('Start saving pkl-style node information\n')
        pkl.dump(nodes_dict, open(pyg_file + ".nodes.pyg", 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Complete saving pkl-style node information\n')

    else:
        nodes = pkl.load(open(pyg_file + ".nodes.pyg", 'rb'))
        node_embeds = nodes['embeds']
        node_maps = nodes['maps']
        labels = nodes['labels']['item']

    # 将结点特征储存到pyg 图数据中
    graph = HeteroData()

    print("Start converting into pyg data")
    # 1. 转换结点特征
    for node_type in tqdm(node_embeds, desc="Node features, numbers and mapping", ascii=True):
        graph[node_type].x = torch.empty(len(node_maps[node_type]), 256)
        for nid, embedding in tqdm(node_embeds[node_type].items()):
            graph[node_type].x[nid] = torch.from_numpy(embedding)
        graph[node_type].num_nodes = len(node_maps[node_type])
        graph[node_type].maps = node_maps[node_type]

    if label_file is not None:
        # 2. 转换标签
        graph['item'].y = torch.zeros(len(node_maps['item']), dtype=torch.long) - 1
        for index, label in tqdm(labels, desc="Node labels", ascii=True):
            graph['item'].y[index] = label

        # 3. 划分数据集
        # 如果不用直接注释就好了，别删
        # 数据集划分
        # 得到有标注的结点索引idx
        indices = (graph['item'].y != -1).nonzero().squeeze()
        print("Num of true labeled nodes:{}".format(indices.shape[0]))
        # 得到训练集和验证集划分
        train_val_random = torch.randperm(indices.shape[0])
        train_idx = indices[train_val_random][:int(indices.shape[0] * 0.8)]
        val_idx = indices[train_val_random][int(indices.shape[0] * 0.8):]
        print("trian_idx:{}".format(train_idx.numpy()))
        print("test_idx:{}".format(val_idx.numpy()))
        # 添加到item类型结点的属性中
        graph['item'].train_idx = train_idx
        graph['item'].val_idx = val_idx

    # 添加每个节点的索引信息n_id
    for ntype in graph.node_types:
        graph[ntype].n_id = torch.arange(graph[ntype].num_nodes)
    print("Complete converting into pyg data")

    print("Start saving into pyg data")
    torch.save(graph, pyg_file + ".pt")
    print("Complete saving into pyg data")
    return graph


def format_pyg_graph(edge_file, node_file, pyg_file, label_file=None):
    if osp.exists(pyg_file + ".pt") and args.reload == False:
        graph = torch.load(pyg_file + ".pt")
    else:
        graph = read_node_atts(node_file, pyg_file, label_file)

    process = tqdm(total=edge_size)

    # graph = HeteroData()
    edges = {}
    count = 0
    with open(edge_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            source_id, dest_id, source_type, dest_type, edge_type = line_info
            source_id = graph[source_type].maps[int(source_id)]
            dest_id = graph[dest_type].maps[int(dest_id)]
            edges.setdefault(edge_type, {})
            edges[edge_type].setdefault('source', []).append(int(source_id))
            edges[edge_type].setdefault('dest', []).append(int(dest_id))
            edges[edge_type].setdefault('source_type', source_type)
            edges[edge_type].setdefault('dest_type', dest_type)
            count += 1
            if count % 100000 == 0:
                process.update(100000)
    process.close()
    print('Complete reading edge information\n')

    print('Start converting edge information\n')
    for edge_type in edges:
        source_type = edges[edge_type]['source_type']
        dest_type = edges[edge_type]['dest_type']
        source = torch.tensor(edges[edge_type]['source'], dtype=torch.long)
        dest = torch.tensor(edges[edge_type]['dest'], dtype=torch.long)
        graph[(source_type, edge_type, dest_type)].edge_index = torch.vstack([source, dest])

    # edge_type 重新排序,pyg处理异质图一般是将其转换为同质图再利用edge_type这个属性确定边的类型,所以最好先把所有图的edge_type按照统一的标准进行排序
    for edge_type in [('b', 'A_1', 'item'),
                      ('f', 'B', 'item'),
                      ('a', 'G_1', 'f'),
                      ('f', 'G', 'a'),
                      ('a', 'H_1', 'e'),
                      ('f', 'C', 'd'),
                      ('f', 'D', 'c'),
                      ('c', 'D_1', 'f'),
                      ('f', 'F', 'e'),
                      ('item', 'B_1', 'f'),
                      ('item', 'A', 'b'),
                      ('e', 'F_1', 'f'),
                      ('e', 'H', 'a'),
                      ('d', 'C_1', 'f')]:
        temp = graph[edge_type].edge_index
        del graph[edge_type]
        graph[edge_type].edge_index = temp

    print('Complete converting edge information\n')
    print('Start saving into pyg data\n')
    torch.save(graph, pyg_file + ".pt")
    print('Complete saving into pyg data\n')


parser = argparse.ArgumentParser()
parser.add_argument('--reload', action='store_true', help="Whether node features should be reloaded")
parser.add_argument('--session', type=int, default=1, help="Session 1 or 2.")
args = parser.parse_args()

assert args.session in [1, 2]

if args.session == 1:
    ####################### Session I #############################################################
    graph_path = '../data/session1/icdm2022_session1_edges.csv'
    node_path = '../data/session1/icdm2022_session1_nodes.csv'
    label_path = '../data/session1/icdm2022_session1_train_labels.csv'
    store_path = '../data/session1/icdm2022_session1'
else:
    ####################### Session II ############################################################
    graph_path = '../data/session2/icdm2022_session2_edges.csv'
    node_path = '../data/session2/icdm2022_session2_nodes.csv'
    label_path = None # Session II does not have node labels
    store_path = '../data/session2/icdm2022_session2'
    
    
format_pyg_graph(graph_path, node_path, store_path, label_path) 