import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import math
from misc.normalization import AdaIN, SEAN, Conv2DMod, Conv2DMod_SEAN


# ==================================================================#
# ==================================================================#
class ResidualBlock(nn.Module):
    """
    Preactivation Residual Block.
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 activation='relu',
                 norm='adain',
                 bias=False,
                 noise=False,
                 down_channels=False,
                 up=False,
                 down=False,
                 w_dim=None,
                 stylegan_mod=False,
                 no_shortcut=False,
                 anti_alias=False,
                 SN=False,
                 sean=False,
                 deepsee=False,
                 seg_dim=None,
                 weight_style=False,
                 **kwargs):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        if stylegan_mod:
            self.noise = noise
            norm = 'none'
        else:
            self.noise = False
        self.norm = norm
        self.down_channels = down_channels
        self.down = down
        self.up = up
        self.no_shortcut = no_shortcut
        self.SN = SN

        if self.down_channels and not self.no_shortcut and dim_in != dim_out:
            self.shortcut = Conv2dBlock(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                activation='none',
                norm='none',
                down=down,
                anti_alias=anti_alias,
                up=up,
                bias=False,
            )
        if self.up:
            main0_out = dim_out
            main1_in = dim_out
        else:
            main0_out = dim_in
            main1_in = dim_in
        self.main0 = Conv2dBlock(dim_in,
                                 main0_out,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm=norm,
                                 activation=activation,
                                 noise=self.noise,
                                 w_dim=w_dim,
                                 stylegan_mod=stylegan_mod,
                                 pre_out=self.down_channels,
                                 SN=SN,
                                 sean=sean,
                                 deepsee=deepsee,
                                 seg_dim=seg_dim,
                                 weight_style=weight_style,
                                 bias=bias)
        self.main1 = Conv2dBlock(main1_in,
                                 dim_out,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm=norm,
                                 activation=activation,
                                 noise=self.noise,
                                 stylegan_mod=stylegan_mod,
                                 w_dim=w_dim,
                                 up=up,
                                 down=down,
                                 anti_alias=anti_alias,
                                 SN=SN,
                                 sean=sean,
                                 deepsee=deepsee,
                                 seg_dim=seg_dim,
                                 weight_style=weight_style,
                                 bias=bias)

    def __str__(self):
        act = self.main0.activation
        name = act.__class__.__name__
        _str_act = '{}'.format(name)
        if name == 'LeakyReLU':
            _str_act += '{}'.format(act.negative_slope)

        if self.norm != 'none' and self.norm != 'sn':
            try:
                _str_norm = '{}'.format(self.main0.norm.__name__)
            except AttributeError:
                _str_norm = '{}'.format(self.main0.norm.__class__.__name__)
        else:
            _str_norm = ''

        _str = '{}['.format(self.__class__.__name__)

        _str = 'Preact' + _str
        _str += _str_norm + '-' if _str_norm else ''
        _str += _str_act + '-'

        if self.up:
            _str += self.main1.up.__class__.__name__ + '-'

        _str += '{}'.format(self.main0.conv_name)

        if self.SN:
            _str += '-SN'

        if self.down:
            _str += '-' + self.main1.down.__class__.__name__

        if self.noise:
            _str += '-' + self.main0.noise.__class__.__name__

        _str += ']'

        return _str

    def forward(self, x):
        if self.down_channels:
            res, pre_norm_relu = self.main0(x.clone())
            if hasattr(self, 'shortcut'):
                x = self.shortcut(pre_norm_relu)
            res = self.main1(res)
        else:
            res = self.main0(x.clone())
            res = self.main1(res)
            if hasattr(self, 'shortcut'):
                x = self.shortcut(x)

        if hasattr(self, 'shortcut') or not self.down_channels:
            out = (x + res) / math.sqrt(2)
        else:
            out = res

        return out


# ==================================================================#
# ==================================================================#
class LinearBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 norm='none',
                 activation='none',
                 dropout='none',
                 SN=False,
                 bias=True):
        super(LinearBlock, self).__init__()
        self.use_bias = bias
        if activation == 'glu':
            output_dim *= 2

        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=self.use_bias)
        if SN:
            self.fc = nn.utils.spectral_norm(self.fc)
            norm = 'none'
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim, affine=True)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim, affine=True)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        activation = activation.split('_')
        if len(activation) == 1:
            value_act = 0.2
        else:
            value_act = float(activation[1])
        activation = activation[0]
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(value_act, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize activation
        if isinstance(dropout, float):
            value_dropout = dropout

        if dropout == 'none':
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(value_dropout, inplace=True)
            # self.dropout = torch.nn.Dropout(value_dropout)

    def __str__(self):
        _str = '{}['.format(self.__class__.__name__)
        _str += '{}'.format(self.fc.__class__.__name__)
        if self.norm:
            try:
                _str += '-{}'.format(self.norm.__name__)
            except BaseException:
                _str += '-{}'.format(self.norm.__class__.__name__)
        if self.activation:
            name = self.activation.__class__.__name__
            _str += '-{}'.format(name)
            if name == 'LeakyReLU':
                _str += '{}'.format(self.activation.negative_slope)
        if self.dropout:
            name = self.dropout.__class__.__name__
            _str += '-{}{}'.format(name, self.dropout.p)
        _str += ']'
        return _str

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.dropout:
            out = self.dropout(out)
        if self.activation:
            out = self.activation(out)
        return out


# ==================================================================#
# ==================================================================#
class Conv2dBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 up=False,
                 down=False,
                 norm='none',
                 activation='relu',
                 noise=False,
                 pre_out=False,
                 w_dim=None,
                 stylegan_mod=False,
                 demod=True,
                 bias_rgb=False,
                 anti_alias=False,
                 sean=False,
                 SN=False,
                 seg_dim=None,
                 deepsee=False,
                 weight_style=False,
                 bias=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        self.pre_out = pre_out
        self.SN = SN
        if stylegan_mod:
            norm = 'none'
            noise = True

        if bias_rgb:
            noise = False

        if '_' in norm:
            affine = eval(norm.split('_')[1])
            norm = norm.split('_')[0]
        else:
            affine = True

        if ',' in norm:
            sean_norm = norm.split(',')[1]
            norm = norm.split(',')[0]

        norm_dim = input_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim, affine=affine)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim, affine=affine)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(16, norm_dim, affine=affine)
        elif norm == 'ws':
            self.norm = nn.GroupNorm(16, norm_dim, affine=affine)
        elif norm == 'adain':
            self.norm = AdaIN(norm_dim, w_dim=w_dim)
        elif norm == 'sean':
            self.norm = SEAN(norm_dim,
                             norm=sean_norm,
                             w_dim=w_dim,
                             seg_dim=seg_dim,
                             deepsee=deepsee)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        activation = activation.split('_')
        if len(activation) == 1:
            value_act = 0.2
        else:
            value_act = float(activation[1])
        activation = activation[0]
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(value_act, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax()
        elif activation == 'logsoftmax':
            self.activation = nn.LogSoftmax()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # if up and not mod2:
        if up:
            assert up in ['nearest', 'bilinear']
            self.up = nn.Upsample(scale_factor=2, mode=up)
        else:
            self.up = False

        # if down and not mod2 and not equal_conv:
        if down:
            if anti_alias:
                self.down = Downsample_AntiAlias(channels=output_dim,
                                                 filt_size=2,
                                                 stride=2)
            else:
                self.down = nn.AvgPool2d(2)
        else:
            self.down = False

        # initialize convolution

        if stylegan_mod:
            if sean:
                _conv = Conv2DMod_SEAN
            else:
                _conv = Conv2DMod
            self.conv = _conv(input_dim,
                              output_dim,
                              kernel_size,
                              upsample=up,
                              downsample=down,
                              demod=demod,
                              bias_rgb=bias_rgb,
                              deepsee=deepsee,
                              seg_dim=seg_dim,
                              weight_style=weight_style,
                              w_dim=w_dim)
        else:
            if self.norm == 'ws':
                _conv = Conv2d_WS
            else:
                _conv = nn.Conv2d
            self.conv = nn.Conv2d(input_dim,
                                  output_dim,
                                  kernel_size,
                                  stride,
                                  padding,
                                  bias=self.use_bias)
            if self.SN:
                self.conv = nn.utils.spectral_norm(self.conv)

        self.conv_name = self.conv.__class__.__name__

        # Introduce gaussian noise
        if noise and stylegan_mod:
            # self.noise = NoiseLayer(output_dim)
            self.noise = NoiseInjection()
        else:
            self.noise = None

    def __str__(self):
        _str = '{}['.format(self.__class__.__name__)

        if self.activation:
            name = self.activation.__class__.__name__
            _str_act = '{}'.format(name)
            if name == 'LeakyReLU':
                _str_act += '{}'.format(self.activation.negative_slope)
        else:
            _str_act = ''

        if self.norm:
            try:
                _str_norm = '{}'.format(self.norm.__name__)
            except BaseException:
                _str_norm = '{}'.format(self.norm.__class__.__name__)
        else:
            _str_norm = ''

        _str += _str_norm + '-' if _str_norm else ''
        _str += _str_act + '-' if _str_act else ''

        if self.up:
            _str += '{}-'.format(self.up.__class__.__name__)

        _str += '{}'.format(self.conv_name)

        if self.SN:
            _str += '-SN'

        if self.down:
            _str += '{}-'.format(self.down.__class__.__name__)

        if self.noise is not None:
            _str += '-' + str(self.noise)

        _str += ']'
        return _str

    def forward(self, x, noise=None):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)

        if self.pre_out:
            pre_norm_relu = x.clone()

        if self.up:
            x = self.up(x)

        x = self.conv(x)

        if self.down:
            x = self.down(x)

        if self.noise is not None:
            x = self.noise(x, noise=noise)

        if self.pre_out:
            return x, pre_norm_relu

        return x


# ==================================================================#
# ==================================================================#
class Conv2d_WS(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(
                                                                dim=3,
                                                                keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1,
                                                              1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


# ==================================================================#
# ==================================================================#
class Conv2dBlockRes(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 up=False,
                 norm='none',
                 activation='none',
                 noise=False,
                 w_dim=None,
                 stylegan_mod=False,
                 no_shortcut=False,
                 down=False,
                 anti_alias=False,
                 SN=False,
                 sean=False,
                 deepsee=False,
                 seg_dim=None,
                 weight_style=False,
                 bias=True):
        super(Conv2dBlockRes, self).__init__()
        self.no_shortcut = no_shortcut
        if up:
            assert up in ['nearest', 'bilinear']

        # Residual
        self.res = ResidualBlock(
            input_dim,
            output_dim,
            # kernel_size,
            # stride,
            # padding,
            bias=bias,
            activation=activation,
            norm=norm,
            noise=noise,
            down_channels=True,
            w_dim=w_dim,
            up=up,
            down=down,
            anti_alias=anti_alias,
            stylegan_mod=stylegan_mod,
            SN=SN,
            sean=sean,
            deepsee=deepsee,
            seg_dim=seg_dim,
            weight_style=weight_style,
            no_shortcut=no_shortcut)

    def __str__(self):
        _str = '{}['.format(self.__class__.__name__)
        if not self.no_shortcut:
            _str += '{}'.format(self.res.shortcut.__class__.__name__)
            _str += '-{}'.format(self.res.__str__())
        else:
            _str += '{}'.format(self.res.__str__())
        _str += ']'
        return _str

    def forward(self, x):
        return self.res(x)


def mask2StyleAttr(segmap, dataset, attr=False, only_attr=False, debug=False):
    segmap_attr = []
    selected_attrs = list(dataset.selected_attrs)
    mask_attr = dict(dataset.mask_attr)
    if attr:
        selected_attrs += list(dataset.semantic_attr.keys())
        mask_attr.update(dataset.semantic_attr)
    elif only_attr:
        selected_attrs = list(dataset.semantic_attr.keys())
        mask_attr = dict(dataset.semantic_attr)
    for idx, attr in enumerate(selected_attrs):
        _mask_attr = mask_attr[attr]
        indexes = [dataset.mask_label[i] for i in _mask_attr]
        out = [segmap[:, i].clone() for i in indexes]
        out = torch.stack(out, dim=1).sum(1)
        if not debug:
            assert out.max() <= 1.0
        segmap_attr.append(out)
    segmap_attr = torch.stack(segmap_attr, dim=1)
    return segmap_attr


class CAM(nn.Module):
    def __init__(self, input_dim, SN=False, activation='lrelu', pre=False):
        super().__init__()
        self.input_dim = input_dim
        # Class Activation Map
        self.gap_fc = nn.Linear(input_dim, 1, bias=False)
        self.gmp_fc = nn.Linear(input_dim, 1, bias=False)
        if SN:
            self.gap_fc = nn.utils.spectral_norm(self.gap_fc)
            self.gmp_fc = nn.utils.spectral_norm(self.gmp_fc)
        self.conv1x1 = nn.Conv2d(input_dim * 2,
                                 input_dim,
                                 kernel_size=1,
                                 stride=1,
                                 bias=True)
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, not pre)
        elif activation == 'relu':
            self.activation = nn.ReLU(not pre)
        elif isinstance(activation, bool) and activation is False:
            self.activation = None
        elif isinstance(activation, bool) and activation is True:
            self.activation = nn.LeakyReLU(0.2, not pre)
        self.preactivation = pre

    def __repr__(self):
        name = self.__class__.__name__
        _str = f'{name}['
        if self.preactivation and self.activation is not None:
            _str += f'{self.activation.__class__.__name__}, '
        _str += f'FC-{self.input_dim}, Conv-{self.input_dim*2}]'
        if not self.preactivation and self.activation is not None:
            _str = _str[:-1] + f', {self.activation.__class__.__name__}]'
        return _str

    def forward(self, x):
        if self.preactivation:
            x = self.activation(x)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)

        heatmap = torch.sum(x, dim=1, keepdim=True)
        x = self.conv1x1(x)
        if self.activation is not None and not self.preactivation:
            x = self.activation(x)

        # heatmap = torch.sum(x, dim=1, keepdim=True)
        return x, cam_logit, heatmap


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


# Anti-aliased block
# https://github.com/adobe/antialiased-cnns
class Downsample_AntiAlias(nn.Module):
    def __init__(self,
                 pad_type='reflect',
                 filt_size=3,
                 stride=2,
                 channels=None,
                 pad_off=0):
        super(Downsample_AntiAlias, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if (self.filt_size == 1):
            a = np.array([
                1.,
            ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp),
                            self.filt,
                            stride=self.stride,
                            groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


##########################################################################
# FaceNet network definition
##########################################################################
class FaceNet(nn.Module):
    """
    https://github.com/timesler/facenet-pytorch
    """
    def __init__(self):
        super(FaceNet, self).__init__()
        from models.perceptual.inception_resnet_v1 import InceptionResnetV1
        self.resnet = InceptionResnetV1('vggface2').eval()
        # self.resnet = InceptionResnetV1('casia-webface').eval()
        from facenet_pytorch import MTCNN  # InceptionResnetV1 # , MTCNN
        # self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.image_size = 160  # 160 was trained on
        self.mtcnn = MTCNN(image_size=self.image_size)

    def forward(self, x):
        x = F.interpolate(x, size=(self.image_size, self.image_size))
        gram_m, features = self.resnet(x)
        return gram_m, features

    def to_input(self, file, mtcnn=False, flip=False, angle=0):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor, rotate
        from torchvision.transforms.functional import hflip, normalize
        img = Image.open(file)
        if mtcnn:
            return self.mtcnn(img).unsqueeze(0)
        if flip:
            img = hflip(img)
        if angle:
            img = rotate(img, angle)
        img = to_tensor(img)
        img = normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0)
        return img

    def demo(self, file, **kwargs):
        img = self.to_input(file, **kwargs)
        img = self.forward(img)
        return img

    def distance(self, i, j, dist='cosine'):
        if dist == 'cosine':
            distance = 1 - nn.CosineSimilarity(1)(i, j).mean().item()
        elif dist == 'mse':
            distance = F.mse_loss(i, j, reduction='mean')
        elif dist == 'l1':
            distance = F.l1_loss(i, j, reduction='mean')
        return distance

    @torch.no_grad()
    def diff(self, file0, file1, distance='cosine', mtcnn=False, **kwargs):
        # fn.diff('data/CelebA/img_celeba/000001.jpg',
        # 'data/CelebA/img_celeba/000009.jpg', angle=45)
        # fn.diff(
        # 'data/RafD/faces/Rafd090_03_Caucasian_male_angry_frontal.jpg',
        # 'data/RafD/faces/Rafd090_03_Caucasian_male_happy_frontal.jpg')
        feat0 = self.demo(file0, mtcnn=mtcnn)
        if kwargs:
            feat1 = self.demo(file0, **kwargs)
        else:
            feat1 = self.demo(file1, mtcnn=mtcnn)
        for (name, i), j in zip(feat0[0].items(), feat1[0].values()):
            dist = self.distance(i, j, dist=distance)
            print('{}: {:0.3f}'.format(name, dist))
        dist = self.distance(i, j, dist=distance)
        print('F.norm: {:0.3f}'.format(dist))


##########################################################################
# VGG network definition
##########################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        # return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]
        return [relu1_2, relu2_2, relu3_3, relu4_3], relu5_3
