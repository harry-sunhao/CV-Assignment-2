import numpy as np
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm
import os

from dataloader import get_data_loader
from models import Discriminator, CycleGenerator

from options import CycleGanOptions

import time


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        # config dirs
        self.expdir = './cycle_gan'
        parmeters_str = f' bs_{opts.batch_size}_ep_{opts.niters}_lr{opts.lr}'
        time_stamp = time.strftime(" %d_%H.%M.%S", time.gmtime())
        self.plotdir = os.path.join(self.expdir, 'plots' + parmeters_str + time_stamp)
        self.ckptdir = os.path.join(self.expdir, 'checkpoints' + parmeters_str + time_stamp)
        if not self.opts.debug:
            os.makedirs(self.plotdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)

        # config data
        self.apple_trainloader, self.apple_testloader = get_data_loader('Apple', self.opts.batch_size,
                                                                        self.opts.num_workers)
        self.windows_trainloader, self.windows_testloader = get_data_loader('Windows', self.opts.batch_size,
                                                                            self.opts.num_workers)

        # config models

        # apple->windows generator
        self.G_a2w = CycleGenerator(self.opts).to(self.opts.device)
        # windows->apple generator
        self.G_w2a = CycleGenerator(self.opts).to(self.opts.device)

        generator_params = list(self.G_a2w.parameters()) + list(self.G_w2a.parameters())

        # apple discriminator
        self.D_a = Discriminator(self.opts).to(self.opts.device)

        # windows discriminator
        self.D_w = Discriminator(self.opts).to(self.opts.device)

        discriminator_params = list(self.D_a.parameters()) + list(self.D_w.parameters())

        # config optimizers
        self.G_optim = torch.optim.Adam(generator_params, lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(discriminator_params, lr=self.opts.lr, betas=(0.5, 0.999))

        # config training
        self.niters = self.opts.niters
        self.criterionCycle = torch.nn.L1Loss()

    def run(self):

        for i in range(self.niters):
            if i % self.opts.eval_freq == 0 or i == self.niters - 1:
                self.eval_step(i)
            if i % self.opts.save_freq == 0 or i == self.niters - 1:
                self.save_step(i)
            self.train_step(i)

    def gan_loss(self, data, real):
        if real:
            return torch.sum(torch.pow(data - 1, 2)) / (2 * self.opts.batch_size)
        else:
            return torch.sum(torch.pow(data, 2)) / (2 * self.opts.batch_size)

    def train_step(self, epoch):
        self.G_w2a.train()
        self.G_a2w.train()

        self.D_a.train()
        self.D_w.train()

        apple_loader = iter(self.apple_trainloader)
        windows_loader = iter(self.windows_trainloader)

        num_iters = min(len(self.apple_trainloader) // self.opts.batch_size,
                        len(self.windows_trainloader) // self.opts.batch_size)

        pbar = tqdm(range(num_iters))
        for i in pbar:
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            # load data
            apple_data = next(apple_loader).to(self.opts.device)  # real A
            windows_data = next(windows_loader).to(self.opts.device)  # real B

            fake_windows = self.G_a2w(apple_data)
            fake_apple = self.G_w2a(windows_data)

            # TODO:train discriminator on real data
            D_a_real_loss = self.gan_loss(self.D_a(apple_data), 1)
            D_w_real_loss = self.gan_loss(self.D_w(windows_data), 1)

            D_real_loss = D_a_real_loss + D_w_real_loss
            D_real_loss.backward()
            self.D_optim.step()

            # TODO:train discriminator on fake data
            self.D_optim.zero_grad()
            D_a_fake_loss = self.gan_loss(self.D_a(fake_apple), 0)
            D_w_fake_loss = self.gan_loss(self.D_w(fake_windows), 0)
            D_fake_loss = D_a_fake_loss + D_w_fake_loss
            D_fake_loss.backward()
            self.D_optim.step()

            # TODO:train generator#####
            fake_apple = self.G_w2a(windows_data)
            rec_windows = self.G_a2w(fake_apple)
            fake_windows = self.G_a2w(apple_data)

            G_loss = self.gan_loss(self.D_a(fake_apple), 1)
            if self.opts.use_cycle_loss:
                G_loss += (windows_data - rec_windows).abs().sum() / windows_data.size(0)
            G_loss.backward()
            self.G_optim.step()

            self.G_optim.zero_grad()
            fake_apple = self.G_w2a(windows_data)
            fake_windows = self.G_a2w(apple_data)
            rec_apple = self.G_w2a(fake_windows)
            G_loss = self.gan_loss(self.D_w(fake_windows), 1)
            if self.opts.use_cycle_loss:
                G_loss += self.criterionCycle(rec_apple, apple_data)
            G_loss.backward()
            self.G_optim.step()

            pbar.set_description('Epoch: {}, G_loss: {:.4f}, D_loss: {:.4f}'.format(epoch, G_loss.item(),
                                                                                    D_real_loss.item() + D_fake_loss.item()))

    def eval_step(self, epoch):
        # TODO: generate 16 images from apple to windows and windows to apple from test data and save them in self.plotdir
        apple_loader = iter(self.apple_testloader)
        windows_loader = iter(self.windows_testloader)

        num_iters = min(16, min(len(self.apple_testloader),
                                len(self.windows_testloader)))

        pbar = tqdm(range(num_iters))
        for i in pbar:
            apple_data = next(apple_loader).to(self.opts.device)  # real A
            windows_data = next(windows_loader).to(self.opts.device)  # real B

            fake_apple = self.G_w2a(windows_data)
            fake_windows = self.G_a2w(apple_data)

            merged = merge_images(apple_data.cpu().detach().numpy(), fake_windows.cpu().detach().numpy(), self.opts)
            path = os.path.join(self.plotdir, '{}-sample-{:06d}-X-Y.png'.format(i, epoch))
            plt.imsave(path, merged)
            # print('Saved {}'.format(path))

            merged = merge_images(windows_data.cpu().detach().numpy(), fake_apple.cpu().detach().numpy(), self.opts)
            path = os.path.join(self.plotdir, '{}-sample-{:06d}-Y-X.png'.format(i, epoch))
            plt.imsave(path, merged)
            # print('Saved {}'.format(path))

    def save_step(self,epoch):
        #####TODO: save models in self.ckptdir#####
        G_a2w_path = os.path.join(self.ckptdir, f'G_a2w_{epoch}.pt')
        G_w2a_path = os.path.join(self.ckptdir, f'G_w2a_{epoch}.pt')
        D_a_path = os.path.join(self.ckptdir, f'D_a_{epoch}.pt')
        D_w_path = os.path.join(self.ckptdir, f'D_w_{epoch}.pt')
        torch.save(self.G_a2w.state_dict(), G_a2w_path)
        torch.save(self.G_w2a.state_dict(), G_w2a_path)
        torch.save(self.D_a.state_dict(), D_a_path)
        torch.save(self.D_w.state_dict(), D_w_path)

        # print('CycleGAN: Checkpoints \nGenerator Saved {}\nDiscriminator Saved{}'.format(G_a2w_path, D_a_path))


def merge_images(sources, targets, opts):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    sources = (sources + 1) / 2
    sources = sources * 255
    sources = sources.astype(np.uint8)
    targets = (targets + 1) / 2
    targets = targets * 255
    targets = targets.astype(np.uint8)
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for (idx, s, t) in (zip(range(row ** 2), sources, targets, )):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)/255


if __name__ == '__main__':
    opts = CycleGanOptions()
    trainer = Trainer(opts)
    trainer.run()
