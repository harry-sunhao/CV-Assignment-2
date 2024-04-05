import os

import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    def __init__(self, obj_path, device=torch.device('cuda:0'), model="train", debug=False):
        self.device = device
        self.debug = debug
        self.color = []
        if self.debug:
            self.points, self.labels, self.color = self.load_data(obj_path)
        else:
            self.points, self.labels = self.load_data(obj_path)
        total_points = self.points.shape[0]
        permutation = torch.randperm(total_points)
        self.points = self.points[permutation]
        self.labels = self.labels[permutation]

        if model == 'train':
            indices = torch.arange(0, int(0.8 * total_points))
        else:
            indices = torch.arange(int(0.8 * total_points), total_points)

        self.points = self.points[indices]
        self.labels = self.labels[indices]
        if self.debug:
            self.color = self.color[permutation]
            self.color = self.color[indices]

    def load_data(self, obj_path):
        pcd = trimesh.load(obj_path)

        # TODO: extract point coordinates and labels for the pointcloud
        points = torch.tensor(pcd.vertices, dtype=torch.float32, device=self.device)
        colors = torch.tensor(pcd.visual.vertex_colors, dtype=torch.float32, device=self.device)[:, :3] / 255.

        red_threshold = torch.tensor([1, 0, 0], device=self.device)
        green_threshold = torch.tensor([0, 1, 0], device=self.device)

        self.points = points
        self.labels = torch.zeros(points.shape[0], dtype=torch.float32, device=self.device)
        self.labels[(colors == green_threshold).all(dim=1)] = 1
        self.labels[(colors == red_threshold).all(dim=1)] = 0
        if self.debug:
            return self.points, self.labels, colors.detach().cpu().numpy()
        return self.points, self.labels

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx=None):
        # TODO: return points and lables for the given index
        if idx is None:
            idx = torch.randint(0, self.points.shape[0], (1,)).item()
        if self.debug:
            return self.points[idx], self.labels[idx], self.color[idx]
        else:
            return self.points[idx], self.labels[idx]


if __name__ == '__main__':
    for cur_obj in os.listdir("processed"):
        print("Processing{}".format(cur_obj))
        dataset = ShapeDataset(f"processed/{cur_obj}", debug=True)
        points, labels, color = dataset[0]
        print(points, labels, color)
