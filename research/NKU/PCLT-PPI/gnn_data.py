import os
import json
import numpy as np
import copy
import mindspore as ms
import random
from tqdm import tqdm

from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from mindspore_gl import Graph, GraphField  # MindSpore Graph Learning框架
from mindspore_gl.nn import GNNCell
from mindspore import Tensor, dtype as mstype
from parameter_setting import *

GNNCell.disable_display()

def data_preparation_protein_all(pseq_path, vec_path, point_path, protein_max_length):
    print('protein data preparation....')
    with open(pseq_path) as f2:
        protein_info = f2.readlines()
    protein_list = []
    protein_idx = 0
    for _ in protein_info:
        protein_list.append(protein_idx)
        protein_idx += 1

    idx = ms.Tensor(np.array(protein_list, dtype=int))
    protein_dict = {}
    vec_dict = {}
    for line in open(vec_path):
        line = line.strip().split('\t')
        vec_dict[line[0]] = np.array([float(x) for x in line[1].split()])

    protein_info = np.loadtxt(pseq_path, delimiter='\t', dtype=str)
    for i in tqdm(idx):
        pro_name = protein_info[int(idx[i]), 0]
        pro_seq = protein_info[int(idx[i]), 1][:protein_max_length]
        point_data = np.loadtxt(point_path + pro_name + '.txt', max_rows=protein_max_length)
        point_seq = np.zeros((len(pro_seq), 13))
        for seq_index in range(len(pro_seq)):
            point_seq[seq_index] = vec_dict[pro_seq[seq_index]]
        point_data = np.hstack((point_data, point_seq))

        if point_data.shape[0] >= protein_max_length:
            protein_point = point_data[:protein_max_length, :16]
        else:
            protein_point = np.zeros((protein_max_length, 16))
            protein_point[:point_data.shape[0], :] = point_data[:, :16]

        protein_dict[pro_name] = protein_point

    return protein_dict

class SetEdgeAttr(GNNCell):
    def construct(self, nh, eh, g: Graph):
        g.set_vertex_attr({"nh": nh})
        g.set_edge_attr({"eh": eh})
        return [v.nh for v in g.dst_vertex]

class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1, label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []
        self.ppi_emb = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.esm_dict = {}
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len

        name = 0
        ppi_name = 0
        self.node_num = 0
        self.edge_num = 0
        if exclude_protein_path is not None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5, 'expression': 6}

        with open(ppi_path, 'r') as file:
            for line in tqdm(file):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                    continue

                # get node and node name
                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1

                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1

                # get edge and its label
                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]

                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        if bigger_ppi_path is not None:
            skip_head = True
            with open(bigger_ppi_path, 'r') as file:
                for line in tqdm(file):
                    if skip_head:
                        skip_head = False
                        continue
                    line = line.strip().split('\t')

                    if line[p1_index] not in self.protein_name.keys():
                        self.protein_name[line[p1_index]] = name
                        name += 1

                    if line[p2_index] not in self.protein_name.keys():
                        self.protein_name[line[p2_index]] = name
                        name += 1

                    temp_data = ""
                    if line[p1_index] < line[p2_index]:
                        temp_data = line[p1_index] + "__" + line[p2_index]
                    else:
                        temp_data = line[p2_index] + "__" + line[p1_index]

                    if temp_data not in self.ppi_dict.keys():
                        self.ppi_dict[temp_data] = ppi_name
                        temp_label = [0, 0, 0, 0, 0, 0, 0]
                        temp_label[class_map[line[label_index]]] = 1
                        self.ppi_label_list.append(temp_label)
                        ppi_name += 1
                    else:
                        index = self.ppi_dict[temp_data]
                        temp_label = self.ppi_label_list[index]
                        temp_label[class_map[line[label_index]]] = 1
                        self.ppi_label_list[index] = temp_label

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)

        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def get_feature_origin(self, pseq_path, vec_path, point_path, protein_max_length):
        self.pvec_dict = {}
        self.pvec_dict = data_preparation_protein_all(pseq_path, vec_path, point_path, protein_max_length)

        self.protein_dict = {}
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = self.pvec_dict[name]

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndarray = np.array(self.ppi_list, dtype=int)
        for edge in ppi_ndarray:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)
        
    def generate_data(self):
        self.get_connected_num()
        print("Connected domain num: {}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list, dtype=int)
        ppi_label_list = np.array(self.ppi_label_list, dtype=int)

        self.edge_index = Tensor(ppi_list.T, mstype.int32)
        self.edge_attr = Tensor(ppi_label_list, mstype.int32)
        self.x = []

        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])

        self.x = np.array(self.x, dtype=np.float32)
        self.x = Tensor(self.x, mstype.float32)
        
        self.gf = GraphField(self.edge_index[0], self.edge_index[1], self.x.shape[0], self.edge_index.shape[1])

        self.data = SetEdgeAttr()(self.x, self.edge_attr, *(self.gf).get_graph())


    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} method to split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("Your mode is '{}', you should use 'bfs', 'dfs' or 'random'".format(mode))
                return
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
