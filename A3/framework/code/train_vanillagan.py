import torch

from tqdm import tqdm
import os

from dataloader import get_data_loader
from models import Generator, Discriminator

from options import VanillaGANOptions


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        #config dirs
        self.expdir = './vanilla_gan'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok = True)
        os.makedirs(self.ckptdir, exist_ok = True)

        #config data
        self.trainloader, self.testloader = get_data_loader(self.opts.emoji_type, self.opts.batch_size, self.opts.num_workers)

        #config models
        self.G = Generator(self.opts).to( self.opts.device)
        self.D = Discriminator(self.opts).to(self.opts.device)

        #config optimizers
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr = self.opts.lr, betas = (0.5, 0.999))
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr = self.opts.lr, betas = (0.5, 0.999))

        #config training
        self.nepochs = self.opts.nepochs

    def run(self):
        for epoch in range(self.nepochs):
            self.train_step(epoch)

            if epoch % self.opts.eval_freq == 0:
                self.eval_step(epoch)
            if epoch % self.opts.save_freq == 0:
                self.save_checkpoint(epoch)



    def train_step(self, epoch):
        self.G.train()
        self.D.train()

        pbar = tqdm(self.trainloader)

        for i, data in enumerate(pbar):
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            real = data.to(self.opts.device)
            noise = self.generate_noise(real.shape[0])
            fake = self.G(noise)

            #train discriminator
            #####TODO: compute discriminator loss and optimize#####

            d_loss = 0.
            ##########################################


            #train generator
            #####TODO: compute generator loss and optimize#####
            g_loss = 0.
            ###################################################
            pbar.set_description("Epoch: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(epoch, g_loss.item(), d_loss.item()))


    def eval_step(self, epoch):
        self.G.eval()
        self.D.eval()

        with torch.no_grad():
            #####TODO: sample from your test dataloader and save results in self.plotdir#####
            pass
            #################################################################################

    def save_checkpoint(self, epoch):
        #####TODO: save your model in self.ckptdir#####
        pass
        ###############################################

if __name__ == '__main__':
    trainer = Trainer(VanillaGANOptions())
    trainer.run()