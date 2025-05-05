import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore import Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore_gl.nn import GINConv, GCNConv, GATConv, GNNCell
from mindspore_gl import Graph, GraphField
import mindspore.numpy as np

from proteinpointnet import get_model
GNNCell.disable_display()

class GCNNet(GNNCell):
    def __init__(self, num_features, out_features, hidden_dim):
        super(GCNNet, self).__init__()
        self.GCN1 = GCNConv(num_features, hidden_dim)
        self.GCN2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Dense(hidden_dim, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lin = nn.Dense(hidden_dim, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0)

    def construct(self, x, in_deg, out_deg, g:Graph):
        x = self.GCN1(x, in_deg, out_deg, g)
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.GCN2(x, in_deg, out_deg, g)
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.lin(x)
        x = self.dropout(self.relu(x))
        return x

class GINNet(GNNCell):
    def __init__(self, num_features, out_features, hidden_dim, num_layers):
        super(GINNet, self).__init__()
        self.convs = nn.CellList()
        self.convs.append(GINConv(nn.SequentialCell([
            nn.Dense(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        ])))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.SequentialCell([
                nn.Dense(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dense(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])))

        self.lin = nn.Dense(hidden_dim, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def construct(self, x, edge_weight, g:Graph):
        for conv in self.convs:
            x = conv(x, edge_weight, g)
        x = self.lin(x)
        x = self.dropout(self.relu(x))
        return x

class GATNet(GNNCell):
    def __init__(self, num_feature, out_feature, him):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, him, num_attn_head=8, attn_drop_out_rate=0.2)
        self.GAT2 = GATConv(8 * him, 8 * him, num_attn_head=1, attn_drop_out_rate=0.2)
        self.fc1 = nn.Dense(8 * him, 8 * him)
        self.fc2 = nn.Dense(8 * him, 8 * him)
        self.bn1 = nn.BatchNorm1d(8 * him)
        self.bn2 = nn.BatchNorm1d(8 * him)
        self.lin = nn.Dense(8 * him, out_feature)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0)

    def construct(self, x, g:Graph):
        x = self.GAT1(x, g)
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.GAT2(x, g)
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.lin(x)
        x = self.dropout(self.relu(x))
        return x

class WeightedFeatureFusion(Cell):
    def __init__(self, feature_dim):
        super(WeightedFeatureFusion, self).__init__()
        self.weights = Parameter(Tensor(mnp.ones(3), mindspore.float32))
        self.bn = nn.BatchNorm1d(feature_dim)

    def construct(self, input1, input2, input3):
        weighted_sum = self.weights[0] * input1 + self.weights[1] * input2 + self.weights[2] * input3
        output = self.bn(weighted_sum)
        return output

class Graph_Net(Cell):
    def __init__(self, hidden=512, feature_fusion=None, class_num=7):
        super(Graph_Net, self).__init__()
        
        self.gcn = GCNNet(256, 512, 128)
        self.gin = GINNet(256, 512, 128, 2)
        self.gat = GATNet(256, 512, 10)
        self.fusion_model = WeightedFeatureFusion(512)
        
        self.feature_fusion = feature_fusion
        self.lin1 = nn.Dense(hidden, hidden)
        self.lin2 = nn.Dense(hidden, hidden)
        self.fc2 = nn.Dense(hidden, class_num)
        
        self.ppc = get_model(256, True)
        
        self.relu = nn.ReLU()
        self.concat = ops.Concat(axis=1)

    def construct(self, x, edge_index, train_edge_id, graph, in_deg, out_deg, edge_weight, p=0.5):
        x = mindspore.Tensor(x, dtype=mindspore.float32)
        edge_index = mindspore.Tensor(edge_index, dtype=mindspore.int32)
        in_deg = mindspore.Tensor(in_deg, dtype=mindspore.int32)
        out_deg = mindspore.Tensor(out_deg, dtype=mindspore.int32)
        edge_weight = mindspore.Tensor(edge_weight, dtype=mindspore.int32)
        
        x = x[:, :, :16]
        x = ops.Transpose()(x, (0, 2, 1))
        x = self.ppc(x)
        
        x_gcn = self.gcn(x, in_deg, out_deg, *graph)
        x_gin = self.gin(x, edge_weight, *graph)
        x_gat = self.gat(x, *graph)

        x = self.fusion_model(x_gcn, x_gin, x_gat)

        x = self.relu(self.lin1(x))
        x = ops.dropout(x, p=p, training=self.training)
        x = self.lin2(x)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = self.concat((x1, x2))
        else:
            x = ops.mul(x1, x2)
        x = self.fc2(x)
        
        node_embedding = {}
        for index in range(node_id.shape[1]):
            p0 = node_id[0][index].item()
            p1 = node_id[1][index].item()
            if p0 not in node_embedding: node_embedding[p0] = x1[index]
            if p1 not in node_embedding: node_embedding[p1] = x2[index]
        
        node_embedding = [Parameter(value, name=str(key)) for key, value in node_embedding.items()]

        return x, node_embedding
