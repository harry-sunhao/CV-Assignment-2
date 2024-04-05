import argparse
import os

import trimesh
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from A5.dataloader import ShapeDataset
import A5.reconstruction
from A5.model import OCCNet


def compute_error_torch(pred, labels):
    absolute_error = torch.abs(pred - labels)
    mape = absolute_error / (1e-2 + torch.abs(labels))

    if len(mape.shape) == 3:
        mape = torch.mean(mape, dim=2)
    mean_loss = torch.mean(mape)
    return mean_loss


def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.train_dataset = ShapeDataset(self.opt.obj_path, device=self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=4096, shuffle=True)
        self.val_dataset = ShapeDataset(self.opt.obj_path, device=self.device, model='val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=4096, shuffle=False)
        self.model = OCCNet(self.opt).to(self.device)

        # lr = 5e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), eps=1e-15,
                                          weight_decay=self.opt.weight_decay)
        self.criterion = torch.nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.lr_decay_step,
                                                         gamma=self.opt.lr_decay_gamma)

        self.nepochs = self.opt.epochs

    def validate(self, validation_dataloader):
        total_loss = 0
        total_psnr = 0
        with torch.no_grad():
            for points, labels in validation_dataloader:
                pred = self.model(points)
                # loss = compute_error_torch(pred.squeeze(), labels)
                loss = self.criterion(pred, labels)
                total_loss += loss.item()
                total_psnr += get_psnr(pred, labels)
        avg_loss = total_loss / len(validation_dataloader)
        avg_psnr = total_psnr / len(validation_dataloader)
        return avg_loss, avg_psnr

    def run(self):
        self.model.train()
        best_loss = float('inf')

        for epoch in range(self.nepochs):
            total_loss = 0.0
            with tqdm(self.train_dataloader, unit="batch") as pbar:
                for points, labels in pbar:
                    pbar.set_description(f"Epoch {epoch + 1}")
                    labels = labels.view(-1, 1)
                    self.optimizer.zero_grad()
                    pred = self.model(points)
                    # loss = compute_error_torch(pred.squeeze(), labels)
                    loss = self.criterion(pred, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss / len(self.train_dataloader))
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.model.state_dict(), f'{opt.name}_{opt.cur_obj}.pth')
            # pbar.set_description(f'Epoch: {epoch}, PSNR: {val_psnr:.2f},Val Loss: {val_loss:.8f}')

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class Opt:
    # 这些是类属性，对所有实例共享
    lr = 0.01
    epochs = 200
    cur_obj = "bunny.obj"
    obj_path = f"processed/{cur_obj}"
    resolution = 128
    batch_size = 4096
    weight_decay = 1e-5
    lr_decay_step = 20
    lr_decay_gamma = 0.1

    def __init__(self, name):
        self.name = name
        if self.name == "1LoD":
            self.grid_type = "dense"
            self.grid_feat_dim = 16
            self.base_lod = 8
            self.num_lods = 1
            self.mlp_width = 64
            self.num_mlp_layers = 2
        elif self.name == "MLoD":
            self.grid_type = "dense"
            self.grid_feat_dim = 16
            self.base_lod = 6
            self.num_lods = 3
            self.mlp_width = 64
            self.num_mlp_layers = 2
        elif self.name == "Hash":
            self.grid_type = "hash"
            self.grid_feat_dim = 4
            self.base_lod = 4
            self.num_lods = 6
            self.mlp_width = 256
            self.num_mlp_layers = 9
        else:
            raise ValueError(f"Config name is not avaliable: {self.name}")
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def print_config(self):
        print("Instance Attributes:")
        for attr, value in self.__dict__.items():
            print(f"  {attr}: {value}")

        print("\nClass Attributes:")
        for attr, value in Opt.__dict__.items():
            if not attr.startswith('__') and not callable(getattr(Opt, attr)):
                print(f"  {attr}: {value}")


if __name__ == '__main__':

    config = ["1LoD", "MLoD", "Hash"]
    opt = Opt(config[2])
    print(f'Load config: {opt.name}')
    opt.print_config()
    for cur_obj in os.listdir("processed"):
        print("Obj path: {}".format(cur_obj))
        opt.obj_path = f"processed/{cur_obj}"
        opt.cur_obj = cur_obj
        trainer = Trainer(opt)
        print('# params: {}'.format(trainer.get_num_params()))
        trainer.run()
