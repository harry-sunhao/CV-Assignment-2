import numpy as np

import torch
from torch.autograd import Variable

from tqdm import tqdm
import os

from dataloader import get_data_loader
from models import Generator, Discriminator

from options import VanillaGANOptions
import matplotlib.pyplot as plt
import time


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        # config dirs
        self.expdir = './vanilla_gan'
        parmeters_str = f' bs_{opts.batch_size}_ep_{opts.nepochs}_lr{opts.lr}'
        time_stamp = time.strftime(" %d_%H.%M.%S", time.gmtime())
        self.plotdir = os.path.join(self.expdir, 'plots' + parmeters_str + time_stamp)
        self.ckptdir = os.path.join(self.expdir, 'checkpoints' + parmeters_str + time_stamp)
        if not self.opts.debug:
            os.makedirs(self.plotdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)

        # config data
        self.trainloader, self.testloader = get_data_loader(self.opts.emoji_type, self.opts.batch_size,
                                                            self.opts.num_workers)

        # config models
        self.G = Generator(self.opts).to(self.opts.device)
        self.D = Discriminator(self.opts).to(self.opts.device)

        # config optimizers
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=self.opts.lr, betas=(0.5, 0.999))

        # config training
        self.nepochs = self.opts.nepochs

        #
        self.real_label = 1
        self.fake_label = 0

    def run(self):
        stime = time.time()
        for epoch in range(self.nepochs):
            self.train_step(epoch)

            if epoch % self.opts.eval_freq == 0 or epoch == self.nepochs - 1:
                self.eval_step(epoch)
            if epoch % self.opts.save_freq == 0 or epoch == self.nepochs - 1:
                self.save_checkpoint(epoch)
        print("Time taken to train:", time.time() - stime)

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

            # train discriminator
            # TODO: compute discriminator loss and optimize
            d_real_loss = sum((self.D(real) - self.real_label) ** 2) / (2 * self.opts.batch_size)
            d_fake_loss = sum((self.D(fake) - self.fake_label) ** 2) / (2 * self.opts.batch_size)

            d_loss = d_fake_loss + d_real_loss

            d_loss.backward()
            self.D_optim.step()
            ##########################################

            # train generator
            # TODO: compute generator loss and optimize
            noise = self.generate_noise(real.shape[0])
            fake = self.G(noise)
            g_loss = sum((self.D(fake) - self.real_label) ** 2) / self.opts.batch_size
            g_loss.backward()
            self.G_optim.step()
            ###################################################
            pbar.set_description(
                "Epoch: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(epoch, g_loss.item(),
                                                                                       d_loss.item()))

    def eval_step(self, epoch):
        self.G.eval()
        self.D.eval()

        with torch.no_grad():
            # TODO: sample from your test dataloader and save results in self.plotdir
            fixed_noise = self.generate_noise(100)
            generated_images = self.G(fixed_noise)
            generated_images = generated_images.cpu().numpy()
            grid = create_image_grid(generated_images)

            path = os.path.join(self.plotdir, 'sample-{:06d}.png'.format(epoch))
            plt.imsave(path, grid)
            print('DCGAN: Sample Saved {}'.format(path))

    def save_checkpoint(self, epoch):
        # TODO: save your model in self.ckptdir
        G_path = os.path.join(self.ckptdir, f'G_{epoch}.pt')
        D_path = os.path.join(self.ckptdir, f'D_{epoch}.pt')
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('DCGAN: Checkpoints \nGenerator Saved {}\nDiscriminator Saved{}'.format(G_path, D_path))

    def generate_noise(self, batch_size):
        return (torch.rand(batch_size, self.opts.noise_size) * 2 - 1).to(self.opts.device).unsqueeze(2).unsqueeze(3)


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    array = (array + 1) / 2
    array = array * 255
    array = array.astype(np.uint8)
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2,
                                                                                                                 0)

    if channels == 1:
        result = result.squeeze()
    return result


if __name__ == '__main__':
    trainer = Trainer(VanillaGANOptions())
    trainer.run()
