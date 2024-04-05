import csv
import os

import numpy as np
import trimesh
from scipy.spatial import KDTree
from skimage import measure

from train import *
from model import *
from A5.dataloader import ShapeDataset


def generate_grid(point_cloud, res):
    """Generate grid over the point cloud with given resolution
    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    coords = np.mgrid[:res, :res, :res]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    length += length / res
    coords_matrix[0, 0] = length[0] / res
    coords_matrix[1, 1] = length[1] / res
    coords_matrix[2, 2] = length[2] / res
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.T

    return coords, coords_matrix


def batch_eval(points, eval_func, is_tensor, num_samples):
    """Predict occupancy of values batch-wise
    Args:
        points (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_samples (int): number of points to evaluate at once
    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """

    num_pts = points.shape[0]
    occ = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        occ[i * num_samples: i * num_samples + num_samples] = eval_func(
            points[i * num_samples: i * num_samples + num_samples]
        ).detach().cpu().numpy().squeeze()
    if num_pts % num_samples:
        occ[num_batches * num_samples:] = eval_func(
            points[num_batches * num_samples:]
        ).detach().cpu().numpy().squeeze()

    return occ


def eval_grid(coords, eval_func, is_tensor, num_per_sample=1024):
    """Predict occupancy of values on a grid
    Args:
        coords (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_per_sample (int): number of points to evaluate at once

    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """
    coords = coords.reshape([-1, 3])
    occ = batch_eval(coords, eval_func, is_tensor, num_samples=num_per_sample)
    return occ


def reconstruct(model, grid, res, transform, is_tensor=False):
    """Reconstruct mesh by predicting occupancy values on a grid
    Args:
        model (function): function that takes a batch of points and returns occupancy values
        grid (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
        transform (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]

    Returns:
        verts (np.array, [M, 3]): 3D coordinates of M vertices
        faces (np.array, [K, 3]): indices of K faces
    """

    occ = eval_grid(grid, model, is_tensor)
    occ = occ.reshape([res, res, res])
    if occ.max() < 0.5:
        # surface_level = (occ.max()+occ.min())/2
        surface_level = None
    else:
        surface_level = 0.5
    # surface_level = np.percentile(occ, 50)
    verts, faces, normals, values = measure.marching_cubes(occ, surface_level)

    verts = np.matmul(transform[:3, :3], verts.T) + transform[:3, 3:4]
    verts = verts.T

    return verts, faces


def compute_metrics(reconstr_path, gt_path, num_samples=1000000):
    """Compute chamfer and hausdorff distances between the reconstructed mesh and the ground truth mesh
    Args:
        reconstr_path (str): path to the reconstructed mesh
        gt_path (str): path to the ground truth mesh
        num_samples (int): number of points to sample from each mesh

    Returns:
        chamfer_dist (float): chamfer distance between the two meshes
        hausdorff_dist (float): hausdorff distance between the two meshes
    """
    reconstr = trimesh.load(reconstr_path)
    gt = trimesh.load(gt_path)

    # sample points on the mesh surfaces using trimesh
    reconstr_pts = reconstr.sample(num_samples)
    gt_pts = gt.sample(num_samples)

    # compute chamfer distance between the two point clouds
    reconstr_tree = KDTree(reconstr_pts)
    gt_tree = KDTree(gt_pts)
    dist1, _ = reconstr_tree.query(gt_pts)
    dist2, _ = gt_tree.query(reconstr_pts)
    chamfer_dist = (dist1.mean() + dist2.mean()) / 2
    hausdorff_dist = max(dist1.max(), dist2.max())

    return chamfer_dist, hausdorff_dist


if __name__ == "__main__":
    from model import Baseline, OCCNet

    file = open("evaluation.csv", mode="a+", encoding="utf-8")
    try:
        log = csv.writer(file)
        log.writerow(["Obj", "Type", "Resolution", "Chamfer distance", "Hausdorff distance"])
        config = ["1LoD", "MLoD", "Hash"]
        res = [64, 128, 256]
        for cur_obj in os.listdir("processed"):
            for name in config:
                opt = Opt(name)
                print(f"Processing {cur_obj}")
                model = OCCNet(opt)
                model_name = f"{name}_{cur_obj}.pth"
                if model_name not in os.listdir():
                    raise FileNotFoundError()
                model.load_state_dict(torch.load(model_name))
                model.eval()
                model.to(opt.device)
                pc = trimesh.load(f"processed/{cur_obj}")
                verts = np.array(pc.vertices)
                for resolution in res:
                    if cur_obj != "bunny.obj" and name == "Hash":
                        continue
                    grid, transform = generate_grid(verts, res=resolution)
                    rec_verts, rec_faces = reconstruct(model, grid, resolution, transform)

                    reconstr_path = f"reconstructions/{cur_obj.split('.')[0]}_{name}_{resolution}.obj"
                    # os.makedirs(os.path.dirname(reconstr_path), exist_ok=True)
                    # trimesh.Trimesh(rec_verts, rec_faces).export(reconstr_path)

                    gt_path = f"data/{cur_obj}"

                    chamfer_dist, hausdorff_dist = compute_metrics(
                        reconstr_path, gt_path, num_samples=1000000
                    )
                    log.writerow([cur_obj, name, resolution, chamfer_dist, hausdorff_dist])
                    print(cur_obj, name, resolution)
                    print(f"Chamfer distance: {chamfer_dist:.4f}")
                    print(f"Hausdorff distance: {hausdorff_dist:.4f}")
                    print("##################")
    finally:
        file.close()
