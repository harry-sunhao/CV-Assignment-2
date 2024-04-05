import torch

from tqdm import tqdm
import os

from dataloader import get_data_loader
from models import Generator, Discriminator

from options import VanillaGANOptions
import matplotlib.pyplot as plt
import numpy as np


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
            # Compute loss for real images
            d_real_decision = self.D(real)
            d_real_loss = ((d_real_decision - 1) ** 2).mean()

            # Compute loss for fake images
            d_fake_decision = self.D(fake.detach())  # Detach to avoid computing gradients for G here
            d_fake_loss = ((d_fake_decision - 0) ** 2).mean()
            # Update discriminator
            d_total_loss = d_real_loss + d_fake_loss
            d_total_loss.backward()
            self.D_optim.step()

            ##########################################


            #train generator
            #####TODO: compute generator loss and optimize#####
            noise = self.generate_noise(real.shape[0])
            fake_images = self.G(noise)
            g_decision = self.D(fake_images)
            g_loss = ((g_decision - 1) ** 2).mean()
            g_loss.backward()
            self.G_optim.step()
            ###################################################
            pbar.set_description("Epoch: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(epoch, g_loss.item(), d_total_loss.item()))


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
        #####TODO: save your model in self.ckptdir#####
        torch.save(self.G.state_dict(), os.path.join(self.ckptdir, f'G_epoch_{epoch}.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.ckptdir, f'D_epoch_{epoch}.pth'))
        ###############################################

    def generate_noise(self, num_images):
        return (torch.rand(num_images, self.opts.noise_size) * 2 - 1).to(self.opts.device).unsqueeze(2).unsqueeze(3)
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