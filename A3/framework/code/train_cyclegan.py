import torch

from tqdm import tqdm
import os

from dataloader import get_data_loader
from models import Discriminator, CycleGenerator

from options import CycleGanOptions


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        #config dirs
        self.expdir = './cycle_gan'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok = True)
        os.makedirs(self.ckptdir, exist_ok = True)

        #config data
        self.apple_trainloader, self.apple_testloader = get_data_loader('Apple', self.opts.batch_size, self.opts.num_workers)
        self.windows_trainloader, self.windows_testloader = get_data_loader('Windows', self.opts.batch_size, self.opts.num_workers)

        #config models

        ##apple->windows generator
        self.G_a2w = CycleGenerator(self.opts).to(self.opts.device)
        ##windows->apple generator
        self.G_w2a = CycleGenerator(self.opts).to(self.opts.device)

        generator_params = list(self.G_a2w.parameters()) + list(self.G_w2a.parameters())

        ##apple discriminator
        self.D_a = Discriminator(self.opts).to(self.opts.device)

        ##windows discriminator
        self.D_w = Discriminator(self.opts).to(self.opts.device)

        discriminator_params = list(self.D_a.parameters()) + list(self.D_w.parameters())

        #config optimizers
        self.G_optim = torch.optim.Adam(generator_params, lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(discriminator_params, lr = self.opts.lr, betas = (0.5, 0.999))

        #config training
        self.niters = self.opts.niters


    def run(self):

        for i in range(self.niters):
            if i % self.opts.eval_freq == 0:
                self.eval_step(i)
            if i % self.opts.save_freq == 0:
                self.save_step()
            self.train_step(i)




    def train_step(self, epoch):
        self.G_w2a.train()
        self.G_a2w.train()

        self.D_a.train()
        self.D_w.train()

        apple_loader = iter(self.apple_trainloader)
        windows_loader = iter(self.windows_trainloader)

        num_iters = min(len(self.apple_trainloader) // self.opts.batch_size, len(self.windows_trainloader) // self.opts.batch_size)

        pbar = tqdm(range(num_iters))
        for i in pbar:
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            #load data
            apple_data = next(apple_loader).to(self.opts.device)
            windows_data = next(windows_loader).to(self.opts.device)

            #####TODO:train discriminator on real data#####
            D_real_loss = 0.
            ###############################################

            #####TODO:train discriminator on fake data#####
            D_fake_loss = 0.
            ###############################################

            #####TODO:train generator#####
            G_loss = 0.

            if self.opts.use_cycle_loss:
                G_loss += 0.
            ##############################

            pbar.set_description('Epoch: {}, G_loss: {:.4f}, D_loss: {:.4f}'.format(epoch, G_loss.item(), D_real_loss.item() + D_fake_loss.item()))





    def eval_step(self, epoch):
        #####TODO: generate 16 images from apple to windows and windows to apple from test data and save them in self.plotdir#####
        pass
    def save_step(self):
        #####TODO: save models in self.ckptdir#####
        pass





if __name__ == '__main__':
    opts = CycleGanOptions()
    trainer = Trainer(opts)
    trainer.run()