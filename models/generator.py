import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import print_debug
from misc.utils import PRINT
from models.utils import weights_init
from misc.normalization import AdainResBlk, ResBlk, MODResBlk, Conv2DMod
import numpy as np


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]
                                    ]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(
            x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


# ----------------- #
# --- GENERATOR --- #
# ----------------- #
class Generator(nn.Module):
    def __init__(self, config, color_dim=3, debug=False, file_debug=None):
        super().__init__()
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.FAN = config.FAN
        # dim_in = config.conv_gen  # 2**14 // self.image_size
        dim_in = 2**13 // self.image_size
        if config.conv_gen != 32:
            dim_in = config.conv_gen
        self.color_dim = color_dim
        self.num_domains = config.num_domains
        self.noise_dim = config.noise_dim
        self.style_dim = config.style_dim
        self.small_dim = config.small_dim
        self.SPLIT_STYLE = config.SPLIT_STYLE
        self.SPLIT_STYLE2 = config.SPLIT_STYLE2
        if self.SPLIT_STYLE:
            self.w_dim = self.style_dim + (
                (self.num_domains - 1) * self.small_dim)
        elif self.SPLIT_STYLE2:
            # Identity and earrings
            self.w_dim = self.style_dim + (
                (self.num_domains - 1) * self.small_dim)
        else:
            self.w_dim = self.num_domains * self.style_dim
        self.from_rgb = nn.Conv2d(color_dim, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.mod = config.MOD

        if config.image_size == 512:
            self.hpf_keys = [32, 64, 128, 256]
            self.hpf_details = [32]
            # self.hpf_keys = [32, 64, 128, 256]
            # self.hpf_details = [32, 64, 128]
        else:
            self.hpf_keys = [32, 64, 128]
            self.hpf_details = [32]

        self.REENACTMENT = config.REENACTMENT
        file_debug = config.log if file_debug is None and debug else file_debug
        max_conv_dim = 512
        w_hpf = float(config.FAN)
        modulation = config.MOD
        if not modulation:
            config_mod = {}
            normblock = AdainResBlk
            self.to_rgb = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(dim_in, color_dim, 1, 1, 0))
        else:
            normblock = MODResBlk
            config_mod = {
                'weight_style': config.WEIGHT_STYLE,
                'replicate_mod': config.REPLICATE_MOD,
            }
            self.to_rgb = Conv2DMod(dim_in,
                                    color_dim,
                                    self.w_dim,
                                    1,
                                    demod=False,
                                    **config_mod)

        # down/up-sampling blocks
        repeat_num = int(np.log2(self.image_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in,
                       dim_out,
                       normalize=True,
                       downsample=True,
                       no_shortcut=config.NO_SHORTCUT))
            self.decode.insert(0,
                               normblock(dim_out,
                                         dim_in,
                                         self.w_dim,
                                         w_hpf=w_hpf,
                                         upsample=True,
                                         **config_mod))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0,
                normblock(dim_out,
                          dim_out,
                          self.w_dim,
                          w_hpf=w_hpf,
                          **config_mod))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)
        if debug:
            self.debug(file_debug)
        weights = weights_init()
        self.apply(weights.assign)

    def debug(self, file_debug):
        PRINT(file_debug, '-- Generator:')
        self.eval()
        x = torch.zeros((self.batch_size, self.color_dim, self.image_size,
                         self.image_size))
        if self.SPLIT_STYLE:
            s = torch.zeros((self.batch_size, self.w_dim))
        else:
            s = torch.zeros(
                (self.batch_size, self.num_domains, self.style_dim))
        x = print_debug(x, [self.from_rgb], file_debug)
        h = print_debug(x, self.encode, file_debug)
        for block in self.decode:
            h = print_debug((h, s), [block], file_debug)
        if self.mod:
            h = print_debug((h, s), [self.to_rgb], file_debug)
        else:
            h = print_debug(h, self.to_rgb, file_debug)
        self.train()

    def encoder(self, x, fan=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (self.FAN) and (x.size(2) in self.hpf_keys):
                cache[x.size(2)] = x
            x = block(x)
        return x, cache

    def decoder(self, x, s, fan=None):
        if isinstance(x, (tuple, list)):
            x, cache = x
        if self.FAN:
            assert fan is not None

        if self.SPLIT_STYLE:
            _s = torch.zeros((s.size(0), self.w_dim)).to(s.device)
            _s[:, :self.style_dim] = s[:, 0]
            for i in range(self.num_domains - 1):
                _s[:, self.style_dim + (i * self.small_dim):self.style_dim +
                   ((i + 1) * self.small_dim)] = s[:, i + 1, :self.small_dim]
            s = _s

        for block in self.decode:
            x = block(x, s)
            if self.FAN and (x.size(2) in self.hpf_keys):
                _fan = fan[0] if x.size(2) in self.hpf_details else fan[1]
                _fan = F.interpolate(_fan, size=x.size(2), mode='bilinear')
                x = x + self.hpf(_fan * cache[x.size(2)])
        if self.mod:
            x = nn.ReLU()(x)
            out = self.to_rgb(x, s)
        else:
            out = self.to_rgb(x)
        return out

    def forward(self, x, s, fan=None):
        x = self.encoder(x, fan=fan)
        out = self.decoder(x, s, fan=fan)
        return out

    def random_noise(self, x, seed=None):
        number = self.get_number(x, seed)
        z = torch.randn(number, self.noise_dim)
        return z

    def get_number(self, x, seed=None):
        if isinstance(x, int):
            number = x
        else:
            number = x.size(0)

        if seed is not None:
            torch.manual_seed(seed)

        return number
