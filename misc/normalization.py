import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re


def get_str(cls):
    name = cls.actv.__class__.__name__
    _str_act = '{}'.format(name)
    if name == 'LeakyReLU':
        _str_act += '{}'.format(cls.actv.negative_slope)

    if hasattr(cls, 'norm1'):
        try:
            _str_norm = '{}'.format(cls.norm1.__name__)
        except AttributeError:
            _str_norm = '{}'.format(cls.norm1.__class__.__name__)
    else:
        _str_norm = ''

    _str_name = '{}['.format(cls.__class__.__name__)

    _str = _str_norm + '-' if _str_norm else ''
    _str += _str_act + '-'

    if cls.upsample:
        _str += 'UpNearest-'

    _str += '{}'.format(cls.conv1.__class__.__name__)

    if hasattr(cls, 'downsample') and cls.downsample:
        _str += '-AvgPool2d'

    if cls.learned_sc:
        _str = 'shortcut[{}]-residual[{}]'.format(
            cls.conv1x1.__class__.__name__, _str)
    else:
        _str = 'residual[{}]'.format(_str)
    _str += ']'
    _str = _str_name + _str
    return _str


# ==================================================================#
# ==================================================================#
class ResBlk(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 actv=nn.LeakyReLU(0.2),
                 normalize=False,
                 downsample=False,
                 upsample=False,
                 no_shortcut=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.upsample = upsample
        self.downsample = downsample
        self.no_shortcut = no_shortcut
        self.learned_sc = dim_in != dim_out and not no_shortcut
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def __str__(self):
        return get_str(self)

    def forward(self, x):
        if not self.no_shortcut:
            x = self._shortcut(x) + self._residual(x)
            return x / math.sqrt(2)  # unit variance
        else:
            return self._residual(x)


# ==================================================================#
# ==================================================================#
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        x = self.norm(x)
        s = s.view(s.size(0), -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        x = (1 + gamma) * x + beta
        return x


# ==================================================================#
# ==================================================================#
class AdainResBlk(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 style_dim=64,
                 w_hpf=0,
                 actv=nn.LeakyReLU(0.2),
                 upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out  # and w_hpf == 0
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def __str__(self):
        return get_str(self)

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# ==================================================================#
# ==================================================================#
class MODResBlk(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 style_dim=64,
                 w_hpf=0,
                 actv=nn.LeakyReLU(0.2),
                 upsample=False,
                 **mod_config):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out and w_hpf == 0
        self._build_weights(dim_in, dim_out, style_dim, **mod_config)

    def _build_weights(self, dim_in, dim_out, style_dim=64, **mod_config):
        self.noise = NoiseInjection()
        self.conv1 = Conv2DMod(dim_in, dim_out, style_dim, 3, **mod_config)
        self.conv2 = Conv2DMod(dim_out, dim_out, style_dim, 3, **mod_config)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, noise=None):
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x, s)
        x = self.noise(x, noise=noise)
        x = self.actv(x)
        x = self.conv2(x, s)
        x = self.noise(x, noise=noise)
        return x

    def __str__(self):
        return get_str(self)

    def forward(self, x, s, noise=None):
        out = self._residual(x, s, noise=noise)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# ==================================================================#
# ==================================================================#
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            if not self.training:
                torch.manual_seed(0)
            noise = x.new_empty(batch, 1, height, width).normal_()
        return x + self.weight * noise
        # if self.training:
        #     if noise is None:
        #         batch, _, height, width = x.shape
        #         noise = x.new_empty(batch, 1, height, width).normal_()
        #     return x + self.weight * noise
        # else:
        #     return x + self.weight


# ==================================================================#
# ==================================================================#
class Conv2DMod(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 w_dim,
                 kernel,
                 demod=True,
                 stride=1,
                 dilation=1,
                 **kwargs):
        super().__init__()
        self.num_features = in_dim
        self.filters = out_dim
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.w_dim = w_dim
        self.affine = nn.Linear(w_dim, in_dim)

        self.EPS = 1e-8
        self.weight = nn.Parameter(
            torch.randn((out_dim, in_dim, kernel, kernel)))

        nn.init.kaiming_normal_(self.weight,
                                a=0,
                                mode='fan_in',
                                nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, s):
        import torch.nn.functional as F
        b, c, h, w = x.shape
        w2 = self.weight[None, :, :, :, :]
        padding = self._get_same_padding(h, self.kernel, self.dilation,
                                         self.stride)
        latent_w = self.affine(s.view(b, -1))
        w1 = latent_w[:, None, :, None, None]
        weights = w2 * (w1 + 1)
        if self.demod:
            d = torch.rsqrt((weights**2).sum(dim=(2, 3, 4), keepdims=True) +
                            self.EPS)
            weights = weights * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, h, w)
        return x

    def __repr__(self):
        name = self.__class__.__name__
        return (f'{name}[{self.num_features}, {self.filters}, {self.kernel}]')


########################################################################
########################################################################
########################################################################


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer),
                                                 affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer),
                                           affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' %
                             subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class ACE(nn.Module):
    def __init__(self,
                 config_text,
                 norm_nc,
                 label_nc,
                 ACE_Name=None,
                 w_dim=64,
                 status='train',
                 spade_params=None,
                 use_rgb=True):
        super().__init__()
        self.ACE_Name = ACE_Name
        self.status = status
        self.save_npy = True
        self.Spade = SPADE(*spade_params)
        self.use_rgb = use_rgb
        self.style_length = w_dim
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)

        assert config_text.startswith('spade')
        parsed = re.search(r'spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc,
                                                           affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                '%s is not a recognized param-free norm type in SPADE' %
                param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        if self.use_rgb:
            self.create_gamma_beta_fc_layers()

            self.conv_gamma = nn.Conv2d(self.style_length,
                                        norm_nc,
                                        kernel_size=ks,
                                        padding=pw)
            self.conv_beta = nn.Conv2d(self.style_length,
                                       norm_nc,
                                       kernel_size=ks,
                                       padding=pw)

    def forward(self, x, segmap, style_codes=None, obj_dic=None, noise=None):

        # Part 1. generate parameter-free normalized activations
        # if noise is None:
        #     noise =  (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).to(x.device)
        # added_noise = (noise * self.noise_var).transpose(1, 3)
        # normalized = self.param_free_norm(x + added_noise)
        if self.status != 'test':
            noise = torch.randn(x.shape[0], x.shape[3], x.shape[2],
                                1).to(x.device)
            added_noise = (noise * self.noise_var).transpose(1, 3)
            normalized = self.param_free_norm(x + added_noise)
        else:
            normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if self.use_rgb:
            [b_size, f_size, h_size, w_size] = normalized.shape
            middle_avg = torch.zeros(
                (b_size, self.style_length, h_size, w_size),
                device=normalized.device)

            if self.status == 'UI_mode':
                # hard coding

                for i in range(1):
                    for j in range(segmap.shape[1]):

                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:
                            if obj_dic is None:
                                print('wrong even it is the first input')
                            else:
                                style_code_tmp = obj_dic[str(j)]['ACE']

                                middle_mu = F.relu(
                                    self.__getattr__('fc_mu' +
                                                     str(j))(style_code_tmp))
                                component_mu = middle_mu.reshape(
                                    self.style_length,
                                    1).expand(self.style_length,
                                              component_mask_area)

                                middle_avg[i].masked_scatter_(
                                    segmap.bool()[i, j], component_mu)

            else:

                for i in range(b_size):
                    for j in range(segmap.shape[1]):
                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:

                            middle_mu = F.relu(
                                self.__getattr__('fc_mu' + str(j))(
                                    style_codes[i][j]))
                            component_mu = middle_mu.reshape(
                                self.style_length,
                                1).expand(self.style_length,
                                          component_mask_area)

                            middle_avg[i].masked_scatter_(
                                segmap.bool()[i, j], component_mu)

                            # REMEMBER TO UNCOMMENT THIS FOR SAVING STYLES
                            # if self.status == 'test' and self.save_npy and self.ACE_Name=='up_2_ACE_0':
                            #     tmp = style_codes[i][j].cpu().numpy()
                            #     dir_path = 'styles_test'

                            #     ############### some problem with obj_dic[i]

                            #     im_name = os.path.basename(obj_dic[i])
                            #     folder_path = os.path.join(dir_path, 'style_codes', im_name, str(j))
                            #     if not os.path.exists(folder_path):
                            #         os.makedirs(folder_path)

                            #     style_code_path = os.path.join(folder_path, 'ACE.npy')
                            #     np.save(style_code_path, tmp)

            gamma_avg = self.conv_gamma(middle_avg)
            beta_avg = self.conv_beta(middle_avg)

            gamma_spade, beta_spade = self.Spade(segmap)

            gamma_alpha = F.sigmoid(self.blending_gamma)
            beta_alpha = F.sigmoid(self.blending_beta)

            gamma_final = gamma_alpha * gamma_avg + (1 -
                                                     gamma_alpha) * gamma_spade
            beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
            out = normalized * (1 + gamma_final) + beta_final
        else:
            gamma_spade, beta_spade = self.Spade(segmap)
            gamma_final = gamma_spade
            beta_final = beta_spade
            out = normalized * (1 + gamma_final) + beta_final

        return out

    def create_gamma_beta_fc_layers(self):

        # These codes should be replaced with torch.nn.ModuleList

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        self.fc_mu8 = nn.Linear(style_length, style_length)
        self.fc_mu9 = nn.Linear(style_length, style_length)
        self.fc_mu10 = nn.Linear(style_length, style_length)
        self.fc_mu11 = nn.Linear(style_length, style_length)
        self.fc_mu12 = nn.Linear(style_length, style_length)
        self.fc_mu13 = nn.Linear(style_length, style_length)
        self.fc_mu14 = nn.Linear(style_length, style_length)
        self.fc_mu15 = nn.Linear(style_length, style_length)
        self.fc_mu16 = nn.Linear(style_length, style_length)
        self.fc_mu17 = nn.Linear(style_length, style_length)
        self.fc_mu18 = nn.Linear(style_length, style_length)


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search(r'spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc,
                                                           affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                '%s is not a recognized param-free norm type in SPADE' %
                param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())

        self.mlp_gamma = nn.Conv2d(nhidden,
                                   norm_nc,
                                   kernel_size=ks,
                                   padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, segmap):

        inputmap = segmap

        actv = self.mlp_shared(inputmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta
