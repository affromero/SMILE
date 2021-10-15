import torch
import torch.nn as nn
from models.utils import print_debug
from misc.utils import PRINT
from misc.normalization import ResBlk
from models.utils import weights_init
import numpy as np


# ==================================================================#
# ==================================================================#
class Discriminator(nn.Module):
    def __init__(self, config, color_dim=3, debug=False, file_debug=None):
        super().__init__()
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        conv_dim = 2**14 // self.image_size
        self.color_dim = color_dim
        self.conv_dim = conv_dim
        dim_in = 2**14 // self.image_size
        max_conv_dim = 512
        self.dataset = config.dataset
        if config.ATTR == 'single' or self.dataset != 'CelebA_HQ':
            num_domains = config.num_domains
        elif color_dim == 22:
            num_domains = 1
        else:
            num_domains = config.num_domains * 2
        file_debug = config.log if file_debug is None and debug else file_debug
        blocks = [nn.Conv2d(color_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(self.image_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

        if debug:
            self.debug(file_debug)

        init_type = 'kaiming'
        weights = weights_init(init_type=init_type)
        self.apply(weights.assign)

    @torch.no_grad()
    def debug(self, file_debug):
        PRINT(file_debug, '-- Discriminator:')
        x = torch.zeros((self.batch_size, self.color_dim, self.image_size,
                         self.image_size))
        x = print_debug(x, self.main, file_debug)

    def forward(self, x, y, sem=None):
        out = self.main(x)
        y = y[:, :, None, None].expand_as(out)
        out = out[y == 1]
        return out
