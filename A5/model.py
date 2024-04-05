from scipy.spatial import KDTree
import numpy as np

import torch
import torch.nn as nn

from A5.utils import bilinear_interp


class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))


class DenseGrid(nn.Module):
    def __init__(self, base_lod, num_lods, feat_dim, device='cuda'):
        super(DenseGrid, self).__init__()

        self.LOD = [2 ** l for l in range(base_lod, base_lod + num_lods)]
        self.feat_dim = feat_dim
        self.device = device
        self.feature_grids = nn.ParameterList()
        self.init_feature_grids()

    def init_feature_grids(self):
        for l, lod in enumerate(self.LOD):
            feature_grid = nn.Parameter(torch.zeros(lod ** 3, self.feat_dim, dtype=torch.float32, device=self.device))
            torch.nn.init.normal_(feature_grid, mean=0, std=0.01)
            self.feature_grids.append(feature_grid)

    def forward(self, pts):
        # TODO: Given 3D points, use the bilinear interpolation function in the utils file to interpolate the
        #  features from the feature grids TODO: concat interpolated feature from each LoD and return the
        #   concatenated tensor
        feats = []
        for lod, feature_grid in zip(self.LOD, self.feature_grids):
            interpolated_feat = bilinear_interp(feature_grid, pts, lod, grid_type='dense')
            feats.append(interpolated_feat)
        return torch.cat(feats, dim=-1)


class HashGrid(nn.Module):
    def __init__(self, min_res, max_res, num_lod, hash_bandwidth, feat_dim, device='cuda'):
        super(HashGrid, self).__init__()

        self.min_res = min_res  # N_min
        self.max_res = max_res  # N_max
        self.num_lod = num_lod  # L
        self.feat_dim = feat_dim
        self.device = device
        self.hash_table_size = 2 ** hash_bandwidth

        b = np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_lod - 1))

        self.LOD = [int(1 + np.floor(self.min_res * (b ** l))) for l in range(self.num_lod)]
        self.feature_grids = nn.ParameterList()
        self.init_feature_grids()
        a = 0

    def init_feature_grids(self):
        for l, lod in enumerate(self.LOD):
            feature_grid = nn.Parameter(
                torch.zeros(min(lod ** 3, self.hash_table_size), self.feat_dim, dtype=torch.float32,
                            device=self.device))
            torch.nn.init.normal_(feature_grid, mean=0, std=0.001)
            self.feature_grids.append(feature_grid)

    def forward(self, x):
        # TODO: Given 3D points, use the hash function to interpolate the features from the feature grids
        # TODO: concat interpolated feature from each LoD and return the concatenated tensor
        feats = []
        # print("Input shape:", x.shape)  # 打印输入的形状

        for lod, feature_grid in zip(self.LOD, self.feature_grids):
            if x.dim() != 2 or x.shape[1] != 3:
                raise ValueError(f"Expected pts to be [num_pts, 3], got: {x.shape}")
            interpolated_feat = bilinear_interp(feature_grid, x, lod, grid_type='hash')
            feats.append(interpolated_feat)
        # print(f"Interpolated features shape at LOD {lod}: {interpolated_feat.shape}")

        concatenated_feats = torch.cat(feats, dim=-1)  # 在特征维度上拼接
        # print("Concatenated features shape:", concatenated_feats.shape)
        return concatenated_feats


class MLP(nn.Module):
    def __init__(self, num_mlp_layers, mlp_width, feat_dim, num_lod):
        super(MLP, self).__init__()

        self.num_layers = num_mlp_layers
        self.width = mlp_width

        self.layers = nn.ModuleList()
        input_dim = feat_dim * num_lod

        # TODO: Create a multi-layer perceptron with num_layers layers and width width
        self.layers.append(nn.Linear(input_dim, self.width))
        self.layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.width, self.width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.width, 1))
        self.layers = nn.Sequential(*self.layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # x is the concatenated feature tensor from the feature grids
        # TODO: pass x through the MLP and return the output which is p(point is inside the object)
        out = self.layers(x)
        out = torch.sigmoid(out)
        return out


class OCCNet(nn.Module):
    def __init__(self, opts):
        # OCCNet is the main model that combines a feature grid (Dense or Hash grid) and an MLP
        # TODO: Initialize the feature grid and MLP
        super(OCCNet, self).__init__()
        self.opts = opts
        if self.opts.grid_type == 'dense':
            self.dense_grid = DenseGrid(base_lod=self.opts.base_lod, num_lods=self.opts.num_lods, feat_dim=self.opts.grid_feat_dim)
            self.mlp = MLP(num_mlp_layers=self.opts.num_mlp_layers, mlp_width=self.opts.mlp_width, feat_dim=self.opts.grid_feat_dim, num_lod=self.opts.num_lods)
        elif self.opts.grid_type == 'hash':
            self.dense_grid = HashGrid(min_res=2**self.opts.base_lod, max_res=2**(self.opts.base_lod+self.opts.num_lods-1), num_lod=self.opts.num_lods, hash_bandwidth=13, feat_dim=self.opts.grid_feat_dim)
            self.mlp = MLP(num_mlp_layers=self.opts.num_mlp_layers, mlp_width=self.opts.mlp_width, feat_dim=self.opts.grid_feat_dim, num_lod=self.opts.num_lods)
        else:
            raise NotImplementedError('Grid type not implemented')

    def get_params(self, lr):
        params = [
            {'params': self.dense_grid.parameters(), 'lr': lr * 10},
            {'params': self.mlp.parameters(), 'lr': lr}
        ]
        return params

    def forward(self, x):
        # check if x is numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().cuda()
        x = self.dense_grid(x)
        x = self.mlp(x)
        return x



