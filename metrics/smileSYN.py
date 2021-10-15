import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

import torchvision

from misc.utils import denorm
from PIL import Image
import torchvision.transforms as transforms
from misc.mask_utils import scatterMask, label2mask
import matplotlib.pyplot as plt
from misc.utils import to_cuda
from misc.blending import blend_imgs


MASK_LABELS = {
    'background': 0,
    'skin': 1,
    'nose': 2,
    'eye_g': 3,
    'l_eye': 4,
    'r_eye': 5,
    'l_brow': 6,
    'r_brow': 7,
    'l_ear': 8,
    'r_ear': 9,
    'mouth': 10,
    'u_lip': 11,
    'l_lip': 12,
    'hair': 13,
    'hat': 14,
    'ear_r': 15,
    'neck_l': 16,
    'neck': 17,
    'cloth': 18
}

MASK_ATTRS = {
    'ALL': MASK_LABELS.keys(),
    'Bangs': ['hair'],
    'NOT_Bangs': ['hair'], 
    'Eyeglasses': ['eye_g'],
    'NOT_Eyeglasses': ['eye_g'],
    'Few_Hair': ['hair'], 
    'Much_Hair': ['hair'], 
    # 'Male': ['skin', 'neck', 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'u_lip', 'l_lip', 'hair', 'eye_g', 'hat'],  # --
    # 'Female': ['skin', 'neck', 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'u_lip', 'l_lip', 'hair', 'eye_g', 'hat'],  # --
    'Male': [], # Nothing
    'Female': [],   
    'Earrings': ['ear_r'],
    'NOT_Earrings': ['ear_r'],
    'Hat': ['hat'],
    'NOT_Hat': ['hat'],
    'General_Style': ['eye_g']
    # 'General_Style': ['eye_g', 'mouth']
}

class SMILE_SYN(nn.Module):
    def __init__(self, img_size=256, verbose=True):
        super().__init__()
        self.img_size = img_size
        # weights_path = 'metrics/sean_stylegan2_{}.pth'.format(img_size)
        weights_path = 'models/pretrained_models/smileSYN_{}.pth'.format(img_size)
        self.projection = 'noencoder' in weights_path
        if verbose:
            print("==> Loading SMILE SYNTHESIS weights from", weights_path)        
        self.model = Generator(size=img_size, input_seg=64, encoder= not self.projection)
        weights = torch.load(weights_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(weights["g_ema"])
        self.model.eval()
        self.noise = self.model.make_noise(seed=0)
        self.MASK_LABELS = MASK_LABELS

    @torch.no_grad()
    def forward_from_file(self, sem_file='metrics/demo_sem.png', rgb_file='metrics/demo_rgb.png', sample=False, **kwargs):
        normalize = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_rgb = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                                interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
        ])
        transform_sem = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                                interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        sem = Image.open(sem_file).convert('RGB')
        sem = transform_sem(sem)*255
        sem = scatterMask(sem[0]).unsqueeze(0)
        if os.path.isfile(rgb_file):
            rgb = Image.open(rgb_file).convert('RGB')
            rgb = normalize(transform_rgb(rgb)).unsqueeze(0)
        else:
            rgb = None
        if sample:
            self.sample_one_image(sem, rgb=rgb, **kwargs)
        else:
            self.forward_from_tensor(sem, rgb=rgb, **kwargs)
    
    @torch.no_grad()
    def sample_one_image(self, sem, rgb, attr='', plot=False, keep_background=False):
        sem = to_cuda(sem)
        sample_kwargs = {'style_rec': True, 'input_is_latent': True}
        if rgb is not None:
            rgb = to_cuda(rgb)
            style_rec = self.model.encoder(rgb, sem)       
        else:
            sample_z = torch.randn(n_sample, self.model.style_dim, device=rgb.device)
            style_rec = self.model.style(sample_z)               

        if attr != '':
            Attr = attr.split(',')
            _assert = [key in MASK_LABELS.keys() for key in Attr]
            assert False not in _assert
            n_sample = 10
            rgb = rgb.repeat(n_sample, 1, 1, 1)
            sem = sem.repeat(n_sample, 1, 1, 1)
            style_rec = style_rec.repeat(n_sample, 1, 1)
            sample_z = torch.randn(n_sample, self.model.style_dim, device=rgb.device)
            sample_z = self.model.style(sample_z)
            sample_z = sample_z.view(style_rec.shape)
            for key in Attr:
                attr2idx = MASK_LABELS[key]
                style_rec[:, attr2idx] = sample_z[:, attr2idx]
            _title = f'Random {Attr}'
        else:
            _title = 'Full reconstruction'
        style_rec = style_rec.view(style_rec.size(0), -1)
        out, _ = self.model(style_rec, sem=sem, **sample_kwargs)
        out = denorm(out)
        rgb = denorm(rgb)
        if keep_background:
            mask_bg = sem[:, MASK_LABELS['background']].unsqueeze(1)
            out = (1 - mask_bg)*out + (mask_bg*rgb)
            _title += ' | keeping background'
        
        out = torch.cat(torch.chunk(out, out.size(0)), dim=-1)
        rgb = rgb[0].unsqueeze(0)
        sem = label2mask(sem[0].unsqueeze(0))
        fig = torch.cat([rgb, sem, out], dim=-1)
        if plot:
            fig = fig[0].cpu().numpy().transpose(1,2,0)
            plt.figure(figsize=(40,20))
            plt.imshow(fig)
            plt.axis('off')
            plt.title(_title)
            plt.show()
        return out, fig        

    @torch.no_grad()
    def forward_from_tensor(self, sem, rgb=None, rgb_guide=None, sem_guide=None, 
                style_random=False, style_ref=None, attr='', force_style_ref=False,
                random_across_batch=False, domain=None,
                random_seed=None, **kwargs):
        sem = to_cuda(sem)
        # sample_kwargs = {'style_rec': True, 'input_is_latent': True, 'noise_seed': random_seed}
        sample_kwargs = {'style_rec': True, 'input_is_latent': True, 'noise': self.noise}
        sample_kwargs.update(kwargs)
        if rgb_guide is not None:
            assert not style_random
            assert sem_guide is not None, "if you provide rgb, you must provide sem"
            rgb_guide = to_cuda(rgb_guide)
            if force_style_ref:
                style = style_ref
                # style_guide = self.model.encoder(rgb_guide, sem_guide)
                # style = self.replace_sem_with_w(sem, style, style_ref=style_guide)
            else:
                style_guide = self.model.encoder(rgb_guide, sem_guide)
                style = self.replace_sem_with_w(sem, style_guide, style_ref=style_ref, domain=domain)
                
                if domain is not None:
                    # to replace attribute not present in the domain
                    style = self.replace_sem_with_w(sem, style, style_ref=style_guide)
                # if style_ref is not None:
                #     style = self.replace_sem_with_w(sem, style, style_ref=style_ref)

        elif style_random:
            if random_seed is not None:
                torch.manual_seed(random_seed)
            if random_across_batch:
                sample_z = torch.randn(1, self.model.style_dim, device=sem.device)
                sample_z = sample_z.repeat(sem.size(0), 1)
            else:
                sample_z = torch.randn(sem.size(0), self.model.style_dim, device=sem.device)
            
            style = self.model.style(sample_z)  
            if style_ref is not None and domain is not None:
                style = self.replace_sem_with_w(sem, style_ref, style_ref=style)
            # sample_kwargs['style_rec'] = False
            # sample_kwargs['input_is_latent'] = False
            

        else:
            raise TypeError("You must choose either random or guided generation.")          
        
        if attr != '':
            # apply a random style to a selected attribute
            Attr = attr.split(',')
            style = style.view(sem.size(0), self.model.label_dim, -1)
            _assert = [key in MASK_LABELS.keys() for key in Attr]
            if style_ref is None:
                sample_z = torch.randn(sem.size(0), self.model.style_dim, device=sem.device)
                sample_z = self.model.style(sample_z)
                sample_z = sample_z.view_as(style_ref)
            else:
                sample_z = style_ref.view(sem.size(0), self.model.label_dim, -1)
            for key in Attr:
                attr2idx = MASK_LABELS[key]
                style[:, attr2idx] = sample_z[:, attr2idx]

        style = style.view(style.size(0), -1)
        out, _ = self.model(style, sem=sem, **sample_kwargs)
        out = denorm(out)

        return out

    @torch.no_grad()
    def replace_sem_with_w(self, sem, style, style_ref=None, domain=None):
        # projection W
        # I want to impose style_ref over sem but there are some segmentations without
        # style in the reference image, hence we have to create it
        # where style is the style currently related to sem
        random = True if style_ref is None else False
        style = style.view(style.size(0), self.model.label_dim, -1)
        # if domain is not None:
        #     style_base = style_ref.clone()
        #     style_ref = style.clone()
        #     style = style_base
        
        for idx, _sem in enumerate(sem):
            _sem = _sem.unsqueeze(0)
            Attr = _sem.max(dim=-1)[0].max(dim=-1)[0][0]
            idx2attr = {value:key for key,value in MASK_LABELS.items()}
            Attr = [idx2attr[idx] for idx, key in enumerate(Attr) if key == 1]
            # sample_z = torch.randn(style_ref.size(0), self.model.style_dim, device=sem.device)
            if random:
                sample_z = torch.randn(sem.size(0), self.model.style_dim, device=sem.device)
                sample_z = self.model.style(sample_z)
                sample_z = sample_z.repeat(sem.size(0), 1)
                sample_z = sample_z.view(sample_z.size(0), self.model.label_dim, -1)
            else:
                sample_z = style_ref.clone()
            for key in Attr:
                attr2idx = MASK_LABELS[key]
                
                if domain is not None:
                    if key in MASK_ATTRS[domain] and not (sample_z[idx, attr2idx] == 0).all():
                        style[idx, attr2idx] = sample_z[idx, attr2idx]
                elif (style[idx, attr2idx] == 0).all():
                    
                    style[idx, attr2idx] = sample_z[idx, attr2idx]   
            # if domain is None:
            #     for key in MASK_LABELS.values():
            #         if (style[idx, key] == 0).all():
            #             style[idx, key] = sample_z[idx, key]  
        style = style.view(style.size(0), -1)
        return style

    @torch.no_grad()
    def replace_w_with_seg(self, sem, style, style_ref=None):
        # projection W
        # Wherever there is an empty space in the semantics, it is filled randomly
        random = True if style_ref is None else False
        style = style.view(style.size(0), self.model.label_dim, -1)
        for _sem in sem:
            _sem = _sem.unsqueeze(0)
            Attr = _sem.max(dim=-1)[0].max(dim=-1)[0][0]
            idx2attr = {value:key for key,value in MASK_LABELS.items()}
            Attr = [idx2attr[idx] for idx, key in enumerate(Attr) if key == 0]
            # sample_z = torch.randn(style.size(0), self.model.style_dim, device=sem.device)
            if random:
                sample_z = torch.randn(sem.size(0), self.model.style_dim, device=sem.device)
                sample_z = self.model.style(sample_z)
                sample_z = sample_z.repeat(style.size(0), 1)
                sample_z = sample_z.view(sample_z.size(0), self.model.label_dim, -1)
            else:
                assert atyle_ref is not None
                sample_z = style
            for key in Attr:
                attr2idx = MASK_LABELS[key]
                assert (sample_z[:, attr2idx] == 0).all()
                style[:, attr2idx] = sample_z[:, attr2idx]                
        style = style.view(style.size(0), -1)
        return style

    def forward(self, sem, rgb):
        sem = F.interpolate((self.img_size, self.img_size), mode='nearest')
        rgb = F.interpolate((self.img_size, self.img_size), mode='bilinear')
        out = self.model(sem, rgb)
        out = denorm(out)
        return out

class Generator(nn.Module):
    def __init__(
        self,
        size=256,
        style_dim=64*19,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        sean=True,
        label_dim=19,
        input_seg=128,
        encoder=False,
        only_spade=False,
        color_dim=3,
    ):
        super().__init__()

        self.size = size
        self.sean = sean
        self.style_dim = style_dim
        self.label_dim = label_dim
        self.w_dim = style_dim // label_dim
        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        self.sean = sean
        if encoder:
            self.encoder = Zencoder(size, self.w_dim)
        if sean:
            sean_divider = 2
        else:
            sean_divider = 1

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512 // sean_divider,
            64: 256 * channel_multiplier // sean_divider,
            128: 128 * channel_multiplier // sean_divider,
            256: 64 * channel_multiplier // sean_divider,
            512: 32 * channel_multiplier // sean_divider,
            1024: 16 * channel_multiplier // sean_divider,
        }
        if self.sean:
            input_sem_size = input_seg
            sem_dim = 256
            self.input_sem_size = input_sem_size
            _input = []
            _input.append(ConvLayer(label_dim, sem_dim, 3))
            for _ in range(2):
                _input.append(ConvLayer(sem_dim, sem_dim, 3, downsample=True))
                input_sem_size /= 2
                # self.input.append(ConvLayer(sem_dim, sem_dim//2, 3, downsample=True))
                # sem_dim //= 2
            # self.input = ConstantInput(self.channels[4])
            _input.append(ConvLayer(sem_dim, self.channels[4], 3, downsample=True))
            input_sem_size /= 2
            self.input_seg = input_sem_size
            self.input = nn.Sequential(*_input)
        else:
            self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, sean=self.sean, only_spade=only_spade, label_dim=label_dim,
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, sean=self.sean, only_spade=only_spade, color_dim=color_dim, label_dim=label_dim)
        if self.sean:
            
            self.log_size = int(math.log(size, 2)) - int(math.log(input_sem_size, 2)) + int(math.log(4, 2))
        else:
            self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            if self.sean:
                
                if self.input_sem_size != 128:
                    input_seg = self.input_seg
                elif self.input_sem_size == 128:
                    input_seg = 8 # bug when training stylegan2
                res = (layer_idx + (1+int(math.log2(input_seg)))*2) // 2
            else:
                res = (layer_idx + 6) // 2
            # res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            _only_spade = only_spade
            # if (i == (self.log_size - 1) and size==512):
            #     _only_spade = True
            if i == self.log_size:
                # drop_last_sean is more stable
                _only_spade = True

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    sean=self.sean,
                    only_spade=_only_spade,
                    label_dim=label_dim,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, sean=self.sean, only_spade=_only_spade, label_dim=label_dim,
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, color_dim=color_dim, sean=self.sean, only_spade=_only_spade, label_dim=label_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_latent(self, batch):
        latent_in = torch.randn(batch, self.style_dim)
        return latent_in

    def make_noise(self, seed=0): 
        
        torch.manual_seed(seed)
        if self.sean:
            _pow = int(math.log2(self.input_seg))
            noises = [to_cuda(torch.randn(1, 1, 2 ** _pow, 2 ** _pow))]
        else:
            noises = [to_cuda(torch.randn(1, 1, 2 ** 2, 2 ** 2))]
        for layer_idx in range(self.num_layers):
            if self.sean:
                res = (layer_idx + ((_pow+1)*2)) // 2
            else:
                res = (layer_idx + 6) // 2
            shape = [1, 1, 2 ** res, 2 ** res]        
            noises.append(to_cuda(torch.randn(*shape)))
            
        # noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2)]

        # for i in range(3, self.log_size + 1):
        #     for _ in range(2):
        #         noises.append(torch.randn(1, 1, 2 ** i, 2 ** i))

        return noises        

    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.style_dim)
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input, create_noise=False):
        if create_noise:
            input = self.make_latent(input.size(0))
        return self.style(input)

    def extract_style(self, rgb, sem):
        noise_random = self.make_latent(rgb.size(0)).to(rgb.device)
        style_random = self.get_latent(noise_random)
        style_ref = self.encoder(rgb, sem)
        style_random = style_random.view_as(style_ref)
        style_ref[style_ref == 0] = style_random[style_ref==0]
        style_ref = style_ref.view(style_ref.size(0), -1)
        return style_ref

    def forward(
        self,
        styles,
        sem=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        style_rec=False
    ):
        if self.sean:
            assert sem is not None, 'You should supply semantics'
            assert sem.max() < self.label_dim
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if isinstance(styles, (tuple, list)) and len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        elif style_rec:
            latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if self.sean:
            latent = latent.view(latent.size(0), latent.size(1), self.label_dim, self.w_dim)
            input_sem = F.interpolate(sem, (self.input_sem_size, self.input_sem_size), mode='nearest')
            out = self.input(input_sem)
            
        else:
            out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0], sem=sem)

        skip = self.to_rgb1(out, latent[:, 1], sem=sem)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            
            out = conv1(out, latent[:, i], noise=noise1, sem=sem)
            out = conv2(out, latent[:, i + 1], noise=noise2, sem=sem)
            skip = to_rgb(out, latent[:, i + 2], skip, sem=sem)

            i += 2

        image = skip
        
        assert image.size(-1) == self.size, f"Image size {image.shape}, expected {self.size}"

        if return_latents:
            return image, latent

        else:
            return image, None

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, upsample=False,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.upsample = upsample

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        if self.upsample:
            out = F.conv_transpose2d(
                input, 
                self.weight.transpose(0, 1) * self.scale, 
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,                
                # padding=0, 
                # stride=2
            )
        else:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d_SEAN(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        label_sem=19,
        w_dim=64,
        only_spade=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        self.only_spade = only_spade
        self.style_length = style_dim
        self.w_dim = w_dim
        self.label_sem = label_sem
        assert self.style_length == self.w_dim*self.label_sem
        self.blending_gamma = nn.Parameter(torch.zeros(1))
        # self.noise_var = nn.Parameter(torch.zeros(in_channel))
        self.spade_gamma = SPADE_Mod(in_channel, label_sem)
        # self.spade_gamma = ConvLayer(label_sem, in_channel, 3, activate=False)

        # if distributed:
        #     self.param_free_norm = SynchronizedBatchNorm2d(in_channel, affine=False)
        # else:
        #     self.param_free_norm = nn.BatchNorm2d(in_channel, affine=False)

        if not self.only_spade:
            self.create_gamma_beta_fc_layers()
            self.style_gamma = ConvLayer(w_dim, in_channel, 3, activate=False)

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
            # torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        # self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )
    
    def create_gamma_beta_fc_layers(self):
        style_length = self.w_dim
        self.fc_mu = nn.ModuleList()
        for _ in range(self.label_sem):
            self.fc_mu.append(EqualLinear(style_length, style_length, bias_init=1))
            # self.fc_mu.append(EqualLinear(style_length, style_length, bias_init=1, activation='fused_lrelu'))

    def forward(self, x, style, sem):
        # if self.training:
        #     noise =  torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).to(x.device)
        #     added_noise = (noise * self.noise_var).transpose(1, 3)
        #     x = self.param_free_norm(x + added_noise)            
        # else:
        #     x = self.param_free_norm(x)        
        # x = self.param_free_norm(x)
        batch, in_channel, height, width = x.shape
        sem = F.interpolate(sem, x.shape[-2:], mode='nearest')
        
        gamma_spade = self.spade_gamma(sem)
        if not self.only_spade:
            style_codes = self.extract_style_matrix(x, style, sem)
            style_codes = self.style_gamma(style_codes) # .view(batch, 1, in_channel, 1, 1)        
            gamma_alpha = torch.sigmoid(self.blending_gamma)
            gamma_final = gamma_alpha * style_codes + (1 - gamma_alpha) * gamma_spade
        else:
            gamma_final = gamma_spade
        x = gamma_final * x  # self.scale as well?
        # for spade/sean we modulate the input and not the weights
        # weight = self.scale * self.weight * style

        weight = self.weight * self.scale

        if self.upsample:
            # weight = weight.transpose(0, 1)
            out = F.conv_transpose2d(x, weight.transpose(0, 1), padding=0, stride=2)
            out = self.blur(out)

        elif self.downsample:
            x = self.blur(x)
            out = F.conv2d(x, weight, padding=0, stride=2)

        else:
            out = F.conv2d(x, weight, padding=self.padding)

        if self.demodulate:
            sigma_weight = weight ** 2 # * self.scale
            sigma_style = gamma_final ** 2
            demod = F.conv2d(sigma_style, sigma_weight, padding=self.padding)
            demod /= height * width
            demod = torch.rsqrt(demod.sum([2, 3]) + 1e-8)
            out = out * demod[:, :, None, None]

        
        return out

    def extract_style_matrix(self, features, style, segmap):
        """
            features: bs, feat, h, w
            style: bs, 19, style_dim
            segmap: bs, 19, 256, 256
        """
        segmap = F.interpolate(segmap, size=features.size()[2:], mode='nearest')

        [b_size, f_size, h_size, w_size] = features.shape
        style_codes = torch.zeros((b_size, style.size(-1), h_size, w_size), device=features.device)

        for i in range(b_size):
            for j in range(segmap.shape[1]):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    # middle_mu = F.relu(self.fc_mu[j](style[i][j]))
                    middle_mu = self.fc_mu[j](style[i][j])
                    component_mu = middle_mu.view(style.size(-1), 1).expand(style.size(-1), component_mask_area)
                    style_codes[i].masked_scatter_(segmap.bool()[i, j], component_mu)
        return style_codes

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            # torch.manual_seed(0)
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise
        # if self.training:
        #     return image + self.weight * noise
        # else:
        #     return image
        


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        sean=False,
        only_spade=False,
        label_dim=19,
    ):
        super().__init__()
        self.sean = sean
        if sean:
            conv_mod = ModulatedConv2d_SEAN
        else:
            conv_mod = ModulatedConv2d
        self.conv = conv_mod(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            only_spade=only_spade,
            w_dim=style_dim//label_dim,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, sem=None):
        if self.sean:
            assert sem is not None
            out = self.conv(input, style, sem)
        else:
            out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, color_dim=3, upsample=True, blur_kernel=[1, 3, 3, 1], sean=False, only_spade=False, label_dim=19):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        if sean:
            # self.conv = ModulatedConv2d_SEAN(in_channel, 3, 1, style_dim, demodulate=False, only_spade=True, w_dim=style_dim//19)
            self.conv = ModulatedConv2d_SEAN(in_channel, color_dim, 1, style_dim, demodulate=False, only_spade=only_spade, w_dim=style_dim//label_dim)
        else:
            self.conv = ModulatedConv2d(in_channel, color_dim, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, color_dim, 1, 1))

    def forward(self, input, style, skip=None, sem=None):
        out = self.conv(input, style, sem=sem)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        elif upsample:
            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
                upsample=upsample,
            )
        )

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            layers.append(blur)

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# class Zencoder(BaseNetwork):
class Zencoder(torch.nn.Module):
    def __init__(self, size, w_dim, input_nc=3, ngf=32, n_downsampling=2):
        super(Zencoder, self).__init__()
        self.output_nc = w_dim 
        if size == 512:
            n_downsampling += 1
        model = [ConvLayer(input_nc, ngf, 3)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [ConvLayer(ngf * mult, ngf * mult * 2, 3, downsample=True)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [ConvLayer(ngf * mult, ngf * mult * 2, 3, upsample=True)]

        model += [ConvLayer(ngf * mult * 2, self.output_nc, 3, downsample=True)]
        self.model = nn.Sequential(*model)        


    def forward(self, input, segmap):
        
        codes = self.model(input)
        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')
        b_size = codes.shape[0]
        f_size = codes.shape[1]
        s_size = segmap.shape[1]
        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        return codes_vector

class SPADE_Mod(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, padding=1):
        super().__init__()
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.mlp_shared = ConvLayer(label_nc, nhidden, 3, activate=True)
        self.mlp_gamma = ConvLayer(nhidden, norm_nc, 3, activate=False)

    def forward(self, segmap):
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        return gamma        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='', help='Select one attribute to change randomly')
    parser.add_argument('--keep_background', action='store_true', default=False)
    args = parser.parse_args()        
    model = SEAN()
    if args.attr != 'all' or args.attr == '':
        out = model.forward_from_file(attr=args.attr.lower(), plot=True, keep_background=args.keep_background)
    else:
        fig_all = []
        for key in MASK_LABELS.keys():
            _, fig = model.forward_from_file(attr=key)
            fig_all.append(fig)
        fig_all = torch.cat(fig_all, dim=-2)
        fig_all = fig_all[0].cpu().numpy().transpose(1,2,0)
        plt.figure(figsize=(40,20))
        plt.imshow(fig_all)
        plt.axis('off')
        plt.show()        