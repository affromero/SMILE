import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import print_debug
from misc.utils import PRINT
from misc.normalization import ResBlk
from models.utils import weights_init
import numpy as np


# ==================================================================#
# ==================================================================#
class StyleEncoder(nn.Module):
    def __init__(self, config, color_dim=3, debug=False, file_debug=None):
        super().__init__()
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.color_dim = color_dim
        self.dataset = config.dataset
        max_conv_dim = 512
        if config.ATTR == 'single' or self.dataset != 'CelebA_HQ':
            self.num_domains = config.num_domains
        else:
            self.num_domains = config.num_domains * 2
        self.style_dim = config.style_dim
        self.small_dim = config.small_dim
        self.REMOVING_MASK = config.REMOVING_MASK
        self.SPLIT_STYLE = config.SPLIT_STYLE
        dim_in = 2**14 // self.image_size
        file_debug = config.log if file_debug is None and debug else file_debug
        blocks = []
        blocks += [nn.Conv2d(color_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(self.image_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        if self.REMOVING_MASK:
            self.shared = nn.ModuleList(blocks)
        else:
            self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for i in range(self.num_domains):
            style_dim = self.style_dim if i < 2 or not config.SPLIT_STYLE else self.small_dim
            self.unshared += [nn.Linear(dim_out, style_dim)]

        if debug:
            self.debug(file_debug)
        weights = weights_init()
        self.apply(weights.assign)

    def debug(self, file_debug):
        PRINT(file_debug, '-- Encoder:')
        x = torch.zeros((self.batch_size, self.color_dim, self.image_size,
                         self.image_size))
        h = print_debug(x, self.shared, file_debug)
        h = h.view(h.size(0), -1)
        for layer in self.unshared:
            print_debug(h, [layer], file_debug)

    def forward(self, x, y):
        # import torch.nn.functional as F
        # x = F.interpolate(x, (256, 256), mode='nearest')
        x = self.shared(x)
        h = x.view(x.size(0), -1)
        out = []
        for layer in self.unshared:
            _out = layer(h)
            if _out.size(1) == self.small_dim and self.SPLIT_STYLE:
                # Repeat so it could be easier to stack in a tensor
                _out = _out.repeat(1, self.style_dim // self.small_dim)
            out += [_out]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        y = y.unsqueeze(-1).expand_as(out)
        if self.dataset != 'CelebA_HQ':
            s = out * y
        else:
            s = out[y == 1]
            s = s.view(y.size(0), -1, self.style_dim)
        return s
