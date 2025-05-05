import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from parameter_setting import *


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    view_shape = tuple(view_shape)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = B
    batch_indices = ops.broadcast_to(ops.arange(B).view(view_shape), repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def uniform_center_sample(sequence, npoint):
    B, N, C = sequence.shape
    step = N // npoint if (N // npoint) > 1 else 1
    indices = ops.arange(0, N, step)
    if indices.shape[0] > npoint:
        indices = indices[:npoint]
    elif indices.shape[0] < npoint:
        indices = ops.concat([indices, ops.tile(indices[-1:], (npoint - indices.shape[0],))])
    indices = indices.expand_dims(0).tile((B, 1))
    return indices

# def uniform_center_sample(sequence, npoint):
#     B, N, C = sequence.shape
#     all_indices = []
#     for b in range(B):
#         non_zero_mask = (sequence[b] != 0).any(dim=-1)
#         valid_length = non_zero_mask.sum().item()
#         step = max(valid_length // npoint, 1)
#         indices = torch.arange(0, valid_length, step, device=sequence.device)
#         if len(indices) < npoint:
#             indices = torch.cat([indices, indices.new_zeros(npoint - len(indices))])
#         indices = indices[:npoint]
#         all_indices.append(indices)
#     return torch.stack(all_indices, dim=0)

def fixed_neighborhood_points(center_indices, sequence, n_neighborhood):
    B, N, C = sequence.shape
    half_window = n_neighborhood // 2

    center_indices = ops.maximum(center_indices, half_window)
    center_indices = ops.minimum(center_indices, N - half_window - 1)
    offsets = ops.arange(-half_window, half_window).expand_dims(0)

    neighbor_indices = center_indices.expand_dims(-1) + offsets
    neighbor_indices = ops.clip_by_value(neighbor_indices, 0, N - 1)
    return neighbor_indices


def sample_and_group_uniform(npoint, n_neighborhood, xyz, points):
    B, N, C = xyz.shape
    center_idx = uniform_center_sample(xyz, npoint)  # 使用均匀采样替代 `farthest_point_sample`    
    # view_shape = list(center_idx.shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    new_xyz = index_points(xyz, center_idx)
    idx = fixed_neighborhood_points(center_idx, xyz, n_neighborhood)  # 替代 `query_ball_point`
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = ops.concat([grouped_xyz_norm, grouped_points], axis=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = ops.zeros((B, 1, C), xyz.dtype)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = ops.concat([grouped_xyz, points.view(B, 1, N, -1)], axis=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Cell):
    def __init__(self, npoint, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def construct(self, xyz, points):
        xyz = ops.transpose(xyz, (0, 2, 1))
        if points is not None:
            points = ops.transpose(points, (0, 2, 1))

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group_uniform(self.npoint, self.nsample, xyz, points)
        new_points = ops.transpose(new_points, (0, 3, 2, 1))  # [B, C+D, nsample, npoint]

        for i in range(len(self.mlp_convs)):
            conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            new_points = ops.relu(bn(conv(new_points)))
        new_points = ops.ReduceMax(keep_dims=False)(new_points, 2)
        new_xyz = ops.transpose(new_xyz, (0, 2, 1))
        return new_xyz, new_points


class get_model(nn.Cell):
    def __init__(self, num_class=256, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 16 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=32, nsample=32, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=64 + 3, mlp=[64, 64, protein_max_length], group_all=True)
        self.fc1 = nn.Dense(protein_max_length, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Dense(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Dense(256, num_class)

    def construct(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l3_xyz, l3_points = self.sa3(l1_xyz, l1_points)
        x = l3_points.view(B, protein_max_length)
        x = self.drop1(ops.relu(self.bn1(self.fc1(x))))
        x = self.drop2(ops.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
