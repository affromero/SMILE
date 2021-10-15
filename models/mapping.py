import torch
import torch.nn as nn
from misc.utils import PRINT
from models.utils import print_debug
from models.utils import weights_init


# ==================================================================#
# ==================================================================#
class Noise2Style(nn.Module):
    def __init__(self, config, debug=False, file_debug=None):

        super().__init__()
        self.latent_dim = config.noise_dim
        self.dataset = config.dataset
        if config.ATTR == 'single' or self.dataset != 'CelebA_HQ':
            self.num_domains = config.num_domains
        else:
            self.num_domains = config.num_domains * 2
        # self.num_domains = config.num_domains * \
        #     2 if config.ATTR != 'single' else config.num_domains
        self.style_dim = config.style_dim
        self.name = '-- Mapping:'
        self.small_dim = config.small_dim
        self.batch_size = config.batch_size
        self.SPLIT_STYLE = config.SPLIT_STYLE
        file_debug = config.log if file_debug is None and debug else file_debug
        layers = []
        layers += [nn.Linear(self.latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for i in range(self.num_domains):
            style_dim = self.style_dim if i < 2 or not config.SPLIT_STYLE else self.small_dim
            self.unshared += [
                nn.Sequential(nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, style_dim))
            ]

        if debug:
            self.debug(file_debug)

        weights = weights_init()
        self.apply(weights.assign)

    def debug(self, file_debug):
        PRINT(file_debug, self.name)
        x = torch.zeros((self.batch_size, self.latent_dim))
        h = print_debug(x, self.shared, file_debug)
        for layer in self.unshared:
            print_debug(h,
                        layer,
                        file_debug,
                        append='- x%i' % (len(self.unshared)))
            break

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            _out = layer(h)
            if _out.size(1) == self.small_dim and self.SPLIT_STYLE:
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
