import abc
import torch


class BaseOptions(metaclass=abc.ABCMeta):
    def __init__(self):
        # self.device = 'cuda'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.expdir = ''
        self.debug = False


class VanillaGANOptions(BaseOptions):
    def __init__(self):
        super(VanillaGANOptions, self).__init__()

        # Dataset options
        self.data_dir = '../emojis'
        self.emoji_type = 'Apple'
        self.batch_size = 32
        self.num_workers = 0

        # Discriminator options
        self.discriminator_channels = [32, 64, 128, 1]

        # Generator options
        self.generator_channels = [128, 64, 32, 3]
        self.noise_size = 100

        # Training options
        self.nepochs = 1000
        self.lr = 0.0002

        self.eval_freq = 10
        self.save_freq = 100


class CycleGanOptions(BaseOptions):
    def __init__(self):
        super(CycleGanOptions, self).__init__()

        # Generator options
        self.generator_channels = [32, 64]

        # Dataset options
        self.data_dir = '../emojis'

        self.batch_size = 1
        self.num_workers = 0

        # Discriminator options
        self.discriminator_channels = [32, 64, 128, 1]

        # Training options
        self.niters = 500
        self.lr = 0.0003

        self.eval_freq = 100
        self.save_freq = 100

        self.use_cycle_loss = True
