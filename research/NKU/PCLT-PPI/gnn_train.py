import os
import time
import math
import random
import numpy as np
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, save_checkpoint
import mindspore.ops as ops
from mindspore import dtype as mstype
import mindspore.ops.functional as F
# 如果需要，可以使用MindSpore的SummaryRecord来替代SummaryWriter
from mindspore.train.summary import SummaryRecord

from gnn_data import GNN_DATA
from gnn_model import Graph_Net
from utils import Metrictor_PPI, print_file

from parameter_setting import *
from mindspore.experimental.optim.lr_scheduler import ReduceLROnPlateau
from mindspore.experimental import optim

np.random.seed(1)
ms.set_seed(1)

def get_parameter_by_name(param_list, target_name):
    for param in param_list:
        if param.name == target_name:
            return param.value()
    return None

def cal_similarity(x1, x2):
    norm_x1 = ops.norm(x1)
    norm_x2 = ops.norm(x2)
    if norm_x1 < 1e-2 or norm_x2 < 1e-2:
        return ms.Tensor(0)
    return F.cosine_similarity(x1, x2, dim=0)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def set_args():
    
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--description', default=description, type=str,
                        help='train description')
    parser.add_argument('--task', default=task, type=str, help='task name')
    parser.add_argument('--ppi_path', default=ppi_path, type=str,
                        help="ppi path")
    parser.add_argument('--pseq_path', default=pseq_path, type=str,
                        help="protein sequence path")
    parser.add_argument('--vec_path', default=vec_path, type=str,
                        help="protein sequence path")
    parser.add_argument('--point_path', default=point_path, type=str,
                        help="protein point path")
    parser.add_argument('--protein_max_length', default=protein_max_length, type=int,
                        help="protein max length")
    parser.add_argument('--split_new', default=split_new, type=boolean_string,
                        help='split new index file or not')
    parser.add_argument('--split_mode', default=split_mode, type=str,
                        help='split method, random, bfs or dfs')
    parser.add_argument('--train_valid_index_path', default=train_valid_index_path, type=str,
                        help='cnn_rnn and gnn unified train and valid ppi index')
    parser.add_argument('--use_lr_scheduler', default=use_lr_scheduler, type=boolean_string,
                        help="train use learning rate scheduler or not")
    parser.add_argument('--save_path', default=save_path, type=str,
                        help='model save path')
    parser.add_argument('--graph_only_train', default=graph_only_train, type=boolean_string,
                        help='train ppi graph conctruct by train or all(train with test)')
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help="gnn train batch size, edge batch size")
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='train epoch number')
    return parser.parse_args()

def compute_in_out_degrees(edges, num_nodes):
    in_degrees = np.zeros(num_nodes, dtype=int)
    out_degrees = np.zeros(num_nodes, dtype=int)
    
    for i in range(edges.shape[1]):
        start_node = edges[0, i]
        end_node = edges[1, i]
        
        out_degrees[start_node] += 1
        in_degrees[end_node] += 1
    
    return in_degrees, out_degrees


def train(model, graph, loss_fn, optimizer, result_file_path, save_path, ppi_data, ppi_graph,
          batch_size=512, epochs=1000, scheduler=None,
          got=False):
    
    global_best_valid_f1 = 0
    global_best_valid_auc = 0
    global_best_valid_aupr = 0
    global_best_valid_hmloss = 0
    
    ones = ops.Ones()
    edge_weight = ones((ppi_data.edge_index.shape[1], 1), ms.float32)
    truth_edge_num = ppi_data.edge_index.shape[1] // 2
    
    if got:
        in_deg, out_deg = compute_in_out_degrees(ppi_data.edge_index_got, ppi_data.x.shape[0])
    else:
        in_deg, out_deg = compute_in_out_degrees(ppi_data.edge_index, ppi_data.x.shape[0])

    for epoch in range(epochs):
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)
        model.set_train(True)

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)
        
        model.set_train(True)
        for step in range(steps):
            if step == steps - 1:
                train_edge_id = graph.train_mask_got[step * batch_size:] if got else graph.train_mask[step * batch_size:]
            else:
                train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size] if got else graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                label = ppi_data.edge_attr_got[train_edge_id]
                label = Tensor(label, mstype.float32)
                
                def forward_fn(label):
                    output, node_embedding = model(ppi_data.x.asnumpy(), ppi_data.edge_index_got.asnumpy(), train_edge_id, ppi_graph, in_deg, out_deg, edge_weight.asnumpy())
                    if use_similarity:
                        similarity_origin = ms.tensor(protein_protein_similarity_val)
                        similarity_new = []
                        for enum, pp in enumerate(protein_protein_similarity_id):
                            x1 = get_parameter_by_name(node_embedding, str(pp[0]))
                            x2 = net_parameter_by_name(node_embedding, str(pp[1]))
                            if x1 is not None and x2 is not None:
                                similarity_new.append(cal_similarity(x1, x2))
                            else:
                                similarity_new.append(similarity_origin[enum])
                        
                        values = [tensor.asnumpy() for tensor in similarity_new]
                        similarity_new = (ms.Tensor(np.array(values), mindspore.float32) + 1) / 2
                        loss_mse = ((similarity_new - similarity_origin) ** 2).mean()

                        loss = loss_fn(output, label) + alpha * loss_mse
                    else:
                        loss = loss_fn(output, label)
                    return loss, output, node_embedding
                
            else:
                label = ppi_data.edge_attr[train_edge_id]
                label = Tensor(label, mstype.float32)
                
                def forward_fn(label):
                    output, node_embedding = model(ppi_data.x.asnumpy(), ppi_data.edge_index.asnumpy(), train_edge_id, ppi_graph, in_deg, out_deg, edge_weight.asnumpy())
                    if use_similarity:
                        similarity_origin = ms.tensor(protein_protein_similarity_val)
                        similarity_new = []
                        for enum, pp in enumerate(protein_protein_similarity_id):
                            x1 = get_parameter_by_name(node_embedding, str(pp[0]))
                            x2 = get_parameter_by_name(node_embedding, str(pp[1]))
                            if x1 is not None and x2 is not None:
                                similarity_new.append(cal_similarity(x1, x2))
                            else:
                                similarity_new.append(similarity_origin[enum])
                        
                        values = [tensor.asnumpy() for tensor in similarity_new]
                        similarity_new = (ms.Tensor(np.array(values), mindspore.float32) + 1) / 2
                        loss_mse = ((similarity_new - similarity_origin) ** 2).mean()

                        loss = loss_fn(output, label) + alpha * loss_mse
                    else:
                        loss = loss_fn(output, label)
                    return loss, output, node_embedding
                
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, output, node_embedding), grads = grad_fn(label)
            optimizer(grads)

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).astype(mstype.float32)

            metrics = Metrictor_PPI(pre_result.asnumpy(), label.asnumpy())
            metrics.show_result()
            f1_sum += metrics.F1
            loss_sum += loss.item()
            print(step)
            print(steps)
            

        save_checkpoint(model, os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.set_train(False)

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)
        
        for step in range(valid_steps):
            if step == valid_steps - 1:
                valid_edge_id = graph.val_mask[step * batch_size:]
            else:
                valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]
            
            output, node_embedding = model(ppi_data.x.asnumpy(), ppi_data.edge_index.asnumpy(), valid_edge_id, ppi_graph, in_deg, out_deg, edge_weight.asnumpy())
            label = ppi_data.edge_attr[valid_edge_id]
            label = Tensor(label, mstype.float32)
            
            if use_similarity:
                similarity_origin = ms.tensor(protein_protein_similarity_val)
                similarity_new = []
                for enum, pp in enumerate(protein_protein_similarity_id):
                    x1 = get_parameter_by_name(node_embedding, str(pp[0]))
                    x2 = get_parameter_by_name(node_embedding, str(pp[1]))
                    if x1 is not None and x2 is not None:
                        similarity_new.append(cal_similarity(x1, x2))
                    else:
                        similarity_new.append(similarity_origin[enum])
                        
                values = [tensor.asnumpy() for tensor in similarity_new]
                similarity_new = (ms.Tensor(np.array(values), mindspore.float32) + 1) / 2
                loss_mse = ((similarity_new - similarity_origin) ** 2).mean()

                loss = loss_fn(output, label) + alpha * loss_mse
            else:
                loss = loss_fn(output, label)

            valid_loss_sum += loss.item()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).astype(mstype.float32)

            valid_pre_result_list.append(pre_result.asnumpy())
            valid_label_list.append(label.asnumpy())
            
            print(step)
            print(valid_steps)

        valid_pre_result_list = np.concatenate(valid_pre_result_list, axis=0)
        valid_label_list = np.concatenate(valid_label_list, axis=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

        metrics.show_result()

        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps
        
        if scheduler is not None:
            scheduler.step(loss)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_auc = metrics.auc
            global_best_valid_aupr = metrics.aupr
            global_best_valid_hmloss = metrics.hmloss

            save_checkpoint(model, os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
        
        print_file("epoch:{}, Train loss:{:.4f} -- Valid loss:{:.4f}, F1:{:.4f}, Auc:{:.4f}, Aupr:{:.4f}, hmloss:{:.4f} -- "
                   "best F1:{:.4f}, best Auc:{:.4f}, best Aupr:{:.4f}, best hmloss:{:.4f}"
            .format(epoch, loss, valid_loss, metrics.F1, metrics.auc, metrics.aupr, metrics.hmloss,global_best_valid_f1,
                    global_best_valid_auc, global_best_valid_aupr, global_best_valid_hmloss), save_file_path=result_file_path)


def main():

    args = set_args()

    # 设置MindSpore上下文
    context.set_context(device_target="Ascend")

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path,
                                point_path=args.point_path, protein_max_length=args.protein_max_length)

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new:
        print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list
    ppi_graph = ppi_data.gf.get_graph()
    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    graph.edge_index_got = ops.Concat(axis=1)((ppi_data.edge_index[:, graph.train_mask],
                                               ppi_data.edge_index[:, graph.train_mask][[1, 0]]))
    graph.edge_attr_got = ops.Concat(axis=0)((ppi_data.edge_attr[graph.train_mask], ppi_data.edge_attr[graph.train_mask]))
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    model = Graph_Net()

    optimizer = optim.Adam(model.trainable_params(), lr=learning_rate, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    loss_fn = nn.BCEWithLogitsLoss()

    if not os.path.exists(args.save_path): 
        os.mkdir(args.save_path)
        
    if not use_similarity: 
        stamp = str(use_similarity) + "_" + args.split_mode + "_" + str(args.graph_only_train)
    else: 
        stamp = str(use_similarity) + str(alpha) + "_" + args.split_mode + "_" + str(args.graph_only_train)
    
    save_path = os.path.join(args.save_path, "{}_{}".format(args.task, stamp))  # SHS27k_True0.1_random_0.2_False
    result_file_path = os.path.join(save_path, "valid_results.txt")
    if not os.path.exists(save_path): 
        os.mkdir(save_path)
    
    train(model, graph, loss_fn, optimizer,
          result_file_path, save_path, ppi_data, ppi_graph,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler,
          got=args.graph_only_train)


if __name__ == "__main__":
    main()