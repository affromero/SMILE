"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import json

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from misc.mask_utils import label2mask
from misc.utils import human_format, scale_image, to_cuda, denorm, create_text
from misc.mask_utils import label2mask, label2mask_plain, scatterMask
import cv2
import warnings
from metrics.attribute_model import AttNet
from metrics.smileSYN import SMILE_SYN as SMILE
from metrics.segmentation_model import MaskNet, bisenet2sean
warnings.filterwarnings('ignore')
ATTRS = [
    'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear',
    'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth',
    'hair', 'hat'
]
ATTRS = ['background'] + ATTRS
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


def vis_parsing_maps(im,
                     parsing_anno,
                     stride,
                     show=False,
                     save_im=False,
                     save_path='vis_results/parsing_map_on_im.jpg',
                     SEAN_COLORS=False):
    # Colors for all 20 parts
    if not SEAN_COLORS:
        # These colors are the output of the net which differs in the attribute
        # order
        part_colors = [[255, 255, 255],
                       [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255],
                       [204, 0, 204], [0, 255, 255], [255, 204, 204],
                       [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                       [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                       [0, 51, 0], [255, 153, 51],
                       [0, 204, 0]]  # These colors for CelebA_MASK
        part_colors = [
            part_colors[MASK_LABELS[ATTRS[i]]] for i in range(len(part_colors))
        ]
        # cmap_colors = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in part_colors]
        # cm = LinearSegmentedColormap.from_list(
        #     'CelebA_Mask', cmap_colors, N=19)
    else:
        # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
        #                [255, 0, 85], [255, 0, 170],
        #                [0, 255, 0], [85, 255, 0], [170, 255, 0],
        #                [0, 255, 85], [0, 255, 170],
        #                [0, 0, 255], [85, 0, 255], [170, 0, 255],
        #                [0, 85, 255], [0, 170, 255],
        #                [255, 255, 0], [255, 255, 85], [255, 255, 170],
        #                [255, 0, 255], [255, 85, 255], [255, 170, 255],
        #                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        # part_colors = [(e[2], e[1], e[0]) for e in part_colors]
        part_colors = [(0, 0, 0), (204, 0, 0), (76, 153, 0), (204, 204, 0),
                       (51, 51, 255), (204, 0, 204), (0, 255, 255),
                       (51, 255, 255), (102, 51, 0),
                       (255, 0, 0), (102, 204, 0), (255, 255, 0), (0, 0, 153),
                       (0, 0, 204), (255, 51, 153), (0, 204, 204), (0, 51, 0),
                       (255, 153, 51), (0, 204, 0)]
        # part_colors = [(e[2], e[1], e[0]) for e in part_colors]
        part_colors.pop(0)
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno,
                                  None,
                                  fx=stride,
                                  fy=stride,
                                  interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    for i in range(3):
        vis_parsing_anno_color[:, :, i] += part_colors[0][i]

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # return vis_parsing_anno_color
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4,
                             vis_parsing_anno_color, 0.6, 0)
    return vis_parsing_anno_color, vis_im


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i (%s)" %
          (name, num_params, human_format(num_params)))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight,
                                mode='fan_in',
                                nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight,
                                mode='fan_in',
                                nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def normalize(x):
    out = (x * 2) - 1
    return out.clamp_(-1, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def save_img(x, ncol, filename, denorm=True):
    from torchvision.utils import save_image
    if denorm:
        x = denormalize(x)
    save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets,
                              args,
                              x_src,
                              y_src,
                              x_ref,
                              y_ref,
                              filename,
                              multidomain=(),
                              mask=None,
                              translate_all=False,
                              fill_rgb=False):
    if mask is not None:
        m_src, m_ref = mask
        # style_semantics inverts m_src and x_src

    N, C, H, W = x_src.size()

    if args.STYLE_SEMANTICS:
        fan = nets.FAN.get_heatmap(m_src) if args.FAN else None
    else:
        fan = nets.FAN.get_heatmap(x_src) if args.FAN else None

    if fill_rgb:
        # attnet = to_cuda(AttNet(verbose=False))
        # masknet = to_cuda(MaskNet(verbose=False))
        smile_syn = to_cuda(SMILE(args.image_size, verbose=False))

    s_ref = nets.S(x_ref, y_ref)
    s_src = nets.S(x_src, y_src)

    _, domain = multidomain
    domain_str = args.domains[domain] if not translate_all else 'ALL'
    if not translate_all:
        _s_ref = s_src.clone()
        _s_ref[:, domain // 2] = s_ref[:, domain // 2]
        s_ref = _s_ref
    x_fake = nets.G(x_src, s_ref, fan=fan)
    if args.STYLE_SEMANTICS:
        if fill_rgb:
            x_fake_rec = scatterMask(
                label2mask_plain(x_fake)[:, 0], x_fake.size(1))
            sty_ref_missing_part = smile_syn.model.encoder(m_ref, x_ref)
            rgb = smile_syn.forward_from_tensor(x_fake_rec,
                                                rgb_guide=m_src,
                                                sem_guide=x_src,
                                                style_ref=sty_ref_missing_part,
                                                domain=domain_str)
            x_fake = label2mask(x_fake, n=x_fake.size(1))
            x_fake = scale_image(rgb, x_fake, None, size=64)
        else:
            x_fake_rec = scatterMask(
                label2mask_plain(x_fake)[:, 0], x_fake.size(1))
    x_rec = nets.G(x_fake_rec, s_src, fan=fan)
    if fill_rgb:
        x_fake_rec = scatterMask(label2mask_plain(x_rec)[:, 0], x_rec.size(1))
        sty_ref_missing_part = smile_syn.model.encoder(m_src, x_src)
        rgb = smile_syn.forward_from_tensor(x_fake_rec,
                                            rgb_guide=m_src,
                                            sem_guide=x_src,
                                            style_ref=sty_ref_missing_part,
                                            domain=domain_str)
        x_rec = label2mask(x_rec, n=x_rec.size(1))
        x_rec = scale_image(rgb, x_rec, None, size=64)
        x_src = label2mask(x_src, n=x_src.size(1))
        x_ref = label2mask(x_ref, n=x_ref.size(1))
        x_src = scale_image(denorm(m_src), x_src, None, size=64)
        x_ref = scale_image(denorm(m_ref), x_ref, None, size=64)
    elif args.STYLE_SEMANTICS:
        x_src = ((label2mask(x_src, n=x_src.size(1)) - 0.5) * 2.).clamp_(-1, 1)
        x_ref = ((label2mask(x_ref, n=x_ref.size(1)) - 0.5) * 2.).clamp_(-1, 1)
        x_fake = ((label2mask(x_fake, n=x_fake.size(1)) - 0.5) * 2.).clamp_(
            -1, 1)
        x_rec = ((label2mask(x_rec, n=x_rec.size(1)) - 0.5) * 2.).clamp_(-1, 1)
    x_concat = [x_src, x_ref, x_fake, x_rec]

    x_concat = torch.cat(x_concat, dim=0)
    save_img(x_concat.cpu(), N, filename, denorm=not fill_rgb)


@torch.no_grad()
def translate_using_latent(nets,
                           args,
                           x_src,
                           y_trg_list,
                           z_trg_list,
                           psi,
                           filename,
                           multidomain=(),
                           mask=None,
                           translate_all=False,
                           fill_rgb=False):
    if mask is not None:
        m_src, m_ref = mask

    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)

    if args.STYLE_SEMANTICS:
        x_src_show = ((label2mask(x_src, n=x_src.size(1)) - 0.5) * 2.).clamp_(
            -1, 1)
    else:
        x_src_show = x_src

    if fill_rgb:
        # attnet = to_cuda(AttNet(verbose=False))
        # masknet = to_cuda(MaskNet(verbose=False))
        smile_syn = to_cuda(SMILE(args.image_size, verbose=False))
        if args.STYLE_SEMANTICS:
            x_src_show = scale_image(denorm(m_src),
                                     denorm(x_src_show),
                                     None,
                                     size=64)

    x_concat = [x_src_show]

    if args.STYLE_SEMANTICS:
        fan = nets.FAN.get_heatmap(m_src) if args.FAN else None
    else:
        fan = nets.FAN.get_heatmap(x_src) if args.FAN else None

    y_src, domain = multidomain
    domain_str = args.domains[domain] if not translate_all else 'ALL'
    s_src = nets.S(x_src, y_src)
    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(y_trg.size(0) * 1000, latent_dim)
        z_many = to_cuda(z_many)
        y_many = y_trg.repeat(1000, 1)
        s_many = nets.F(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)

        s_avg = s_avg[:, domain // 2].repeat(N, 1)

        for count, z_trg in enumerate(z_trg_list):
            s_trg = nets.F(z_trg, y_trg)
            if not translate_all:
                s_trg = torch.lerp(s_avg, s_trg[:, domain // 2], psi)
                _s_trg = s_src.clone()
                _s_trg[:, domain // 2] = s_trg
                s_trg = _s_trg
            # import ipdb; ipdb.set_trace()
            x_fake = nets.G(x_src, s_trg, fan=fan)
            if fill_rgb:
                x_fake_smile = scatterMask(
                    label2mask_plain(x_fake)[:, 0], x_fake.size(1))
                # x_fake_smile = ((x_fake_smile - 0.5) * 2.).clamp_(-1, 1)
                # import ipdb; ipdb.set_trace()
                # rgb = smile_syn.forward_from_tensor(x_fake_smile, style_random=True, random_seed=count, random_across_batch=True, domain=domain_str)
                sty_rec = smile_syn.model.encoder(m_src, x_src)
                rgb = smile_syn.forward_from_tensor(x_fake_smile,
                                                    style_random=True,
                                                    random_seed=count,
                                                    random_across_batch=True,
                                                    style_ref=sty_rec,
                                                    domain=domain_str)
                x_fake = label2mask(x_fake, n=x_fake.size(1))
                x_fake = scale_image(rgb, x_fake, None, size=64)
            elif args.STYLE_SEMANTICS:
                x_fake = ((label2mask(x_fake, n=x_fake.size(1)) - 0.5) *
                          2.).clamp_(-1, 1)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_img(x_concat.cpu(), N, filename, denorm=not fill_rgb)
    # import ipdb; ipdb.set_trace()


@torch.no_grad()
def translate_using_reference(nets,
                              args,
                              x_src,
                              x_ref,
                              y_ref,
                              filename,
                              multidomain=(),
                              mask=None,
                              translate_all=False,
                              fill_rgb=False):
    if mask is not None:
        m_src, m_ref = mask

    if args.STYLE_SEMANTICS:
        x_src_show = ((label2mask(x_src, n=x_src.size(1)) - 0.5) * 2.).clamp_(
            -1, 1)
        x_ref_show = ((label2mask(x_ref, n=x_ref.size(1)) - 0.5) * 2.).clamp_(
            -1, 1)
    else:
        x_src_show = x_src
        x_ref_show = x_ref

    if fill_rgb:
        # attnet = to_cuda(AttNet(verbose=translate_all))
        # masknet = to_cuda(MaskNet(verbose=translate_all))
        smile_syn = to_cuda(SMILE(args.image_size, verbose=translate_all))
        if args.STYLE_SEMANTICS:
            # _img = scale_image(denorm(img), label2mask(sem_input[0].unsqueeze(0)), None)
            x_src_show = scale_image(denorm(m_src),
                                     denorm(x_src_show),
                                     None,
                                     size=64)
            x_ref_show = scale_image(denorm(m_ref),
                                     denorm(x_ref_show),
                                     None,
                                     size=64)

    N, C, H, W = x_src_show.size()
    wb = to_cuda(torch.ones(1, C, H, W))
    x_src_with_wb = torch.cat([wb, x_src_show], dim=0)

    if args.STYLE_SEMANTICS:
        fan = nets.FAN.get_heatmap(m_src) if args.FAN else None
    else:
        fan = nets.FAN.get_heatmap(x_src) if args.FAN else None

    s_ref = nets.S(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1, 1)

    if x_src.size(0) == 4:
        x_concat = []
        N -= 1
    else:
        x_concat = [x_src_with_wb]
    y_src, domain = multidomain
    domain_str = args.domains[domain] if not translate_all else 'ALL'
    # if args.REENACTMENT and not translate_all:
    #     domain_str = 'Reenactment'
    # import ipdb; ipdb.set_trace()
    s_src = nets.S(x_src, y_src)
    for i, s_ref in enumerate(s_ref_list):
        if not translate_all:
            _s_ref = s_src.clone()
            _s_ref[:, domain // 2] = s_ref[:, domain // 2]
            s_ref = _s_ref
        # remove style from opposite domain
        x_fake = nets.G(x_src, s_ref, fan=fan)
        if args.STYLE_SEMANTICS:
            if fill_rgb:
                x_fake_smile = scatterMask(
                    label2mask_plain(x_fake)[:, 0], x_fake.size(1))
                # x_fake_smile = ((x_fake_smile - 0.5) * 2.).clamp_(-1, 1)
                # import ipdb; ipdb.set_trace()
                # if translate_all:
                _m_ref = m_ref[i].unsqueeze(0).repeat(N, 1, 1, 1)
                _x_ref = x_ref[i].unsqueeze(0).repeat(N, 1, 1, 1)
                # else:
                #     _m_ref = m_ref
                #     _x_ref = x_ref
                sty_ref_missing_part = smile_syn.model.encoder(_m_ref, _x_ref)
                rgb = smile_syn.forward_from_tensor(
                    x_fake_smile,
                    rgb_guide=m_src,
                    sem_guide=x_src,
                    style_ref=sty_ref_missing_part,
                    domain=domain_str)
                x_fake = label2mask(x_fake, n=x_fake.size(1))
                x_fake = scale_image(rgb, x_fake, None, size=64)
            else:
                x_fake = ((label2mask(x_fake, n=x_fake.size(1)) - 0.5) *
                          2.).clamp_(-1, 1)

        if x_src.size(0) == 4:
            x_fake_with_ref = x_fake
        else:
            x_fake_with_ref = torch.cat([x_ref_show[i:i + 1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_img(x_concat.cpu(), N + 1, filename, denorm=not fill_rgb)
    create_text(filename,
                'Input',
                size_text=x_fake.size(-1) // 6,
                rotate=90,
                row=0,
                column=0,
                force_replace=True)
    create_text(filename,
                'Reference',
                size_text=x_fake.size(-1) // 6,
                rotate=0,
                row=0,
                column=0,
                force_replace=True)


@torch.no_grad()
def debug_image_multidomain(nets,
                            args,
                            val_loader,
                            name,
                            training=True,
                            fill_rgb=False,
                            translate_all=False):

    for key in nets.keys():
        nets[key].eval()

    if not training:
        name = name.replace('.jpg', '')
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, '{}')

    domains = val_loader.dataset.selected_attrs
    number_of_images = args.batch_sample
    num_outs_per_domain = 3  # 10
    non_label = 0 if args.ATTR != 'single' else 1
    num_outputs = 1 if training else 10
    for count in range(num_outputs):
        out_file = 'out_{}_'.format(count) if not training else ''
        if translate_all:
            # Translate input to all reference attributes
            for _iter, data in enumerate(val_loader):
                # (real_x, real_c, mask, _)
                x_src = to_cuda(data['image'])
                y_src = to_cuda(data['label'])
                m_src = to_cuda(data['mask'])
                # inverse = torch.arange(x_src.size(0)-1, -1, -1)
                inverse = torch.arange(0, x_src.size(0), 1)
                x_ref = x_src[inverse]
                y_ref = y_src[inverse]
                m_ref = m_src[inverse]

                if not training:
                    filename = out_file + 'sample_reference_domain_all.jpg'
                    filename = name.format(filename)
                else:
                    filename = out_file + '_reference_domain_all.jpg'
                    filename = name.replace('.jpg', filename)
                translate_using_reference(nets,
                                          args,
                                          x_src,
                                          x_ref,
                                          y_ref,
                                          filename,
                                          multidomain=(y_src, 0),
                                          mask=(m_src, m_ref),
                                          translate_all=True,
                                          fill_rgb=fill_rgb)
                if number_of_images == 4:
                    # ablation experiments
                    return
                # Video
                if not training:
                    filename = out_file + 'sample_reference_domain_all.mp4'
                    filename = name.format(filename)
                    video_ref(nets,
                              args,
                              x_src,
                              x_ref,
                              y_ref,
                              filename,
                              mask=(m_src, m_ref),
                              multidomain=(y_src, 0),
                              translate_all=True,
                              fill_rgb=fill_rgb)
                    filename = out_file + 'sample_cycle_consistency_domain_all.jpg'
                    filename = name.format(filename)
                else:
                    filename = out_file + '_cycle_consistency_domain_all.jpg'
                    filename = name.replace('.jpg', filename)

                inverse = torch.arange(x_src.size(0) - 1, -1, -1)
                # inverse = torch.arange(0, x_src.size(0), 1)
                x_ref = x_src[inverse]
                y_ref = y_src[inverse]
                m_ref = m_src[inverse]

                translate_and_reconstruct(nets,
                                          args,
                                          x_src,
                                          y_src,
                                          x_ref,
                                          y_ref,
                                          filename,
                                          multidomain=(y_src, 0),
                                          mask=(m_src, m_ref),
                                          fill_rgb=fill_rgb,
                                          translate_all=True)
                break
                # import ipdb; ipdb.set_trace()

        for domain in range(len(domains)):
            for _iter, data in enumerate(val_loader):
                # (real_x, real_c, mask, _)
                real_x = data['image']
                real_c = data['label']
                mask = data['mask']
                _x_src = real_x[real_c[:, domain] == non_label]
                _y_src = real_c[real_c[:, domain] == non_label]
                _m_src = mask[real_c[:, domain] == non_label]
                _x_ref = real_x[real_c[:, domain] == 1]
                _y_ref = real_c[real_c[:, domain] == 1]
                _m_ref = mask[real_c[:, domain] == 1]
                if _iter == 0:
                    x_src, x_ref, m_src, m_ref = _x_src, _x_ref, _m_src, _m_ref
                    y_src, y_ref = _y_src, _y_ref
                else:
                    if x_src.size(0) < number_of_images:
                        x_src = torch.cat([x_src, _x_src], dim=0)
                        y_src = torch.cat([y_src, _y_src], dim=0)
                        m_src = torch.cat([m_src, _m_src], dim=0)
                    if x_ref.size(0) < number_of_images:
                        x_ref = torch.cat([x_ref, _x_ref], dim=0)
                        y_ref = torch.cat([y_ref, _y_ref], dim=0)
                        m_ref = torch.cat([m_ref, _m_ref], dim=0)

                n_min = min(x_src.size(0), x_ref.size(0))
                x_src = x_src[:min(n_min, number_of_images)]
                y_src = y_src[:min(n_min, number_of_images)]
                m_src = m_src[:min(n_min, number_of_images)]
                x_ref = x_ref[:min(n_min, number_of_images)]
                y_ref = y_ref[:min(n_min, number_of_images)]
                m_ref = m_ref[:min(n_min, number_of_images)]
                if x_src.size(0) == number_of_images and x_ref.size(
                        0) == number_of_images:
                    break
            x_src = to_cuda(x_src)
            y_src = to_cuda(y_src)
            m_src = to_cuda(m_src)
            x_ref = to_cuda(x_ref)
            y_ref = to_cuda(y_ref)
            m_ref = to_cuda(m_ref)

            # reference-guided image synthesis
            if not training:
                filename = out_file + 'sample_reference_domain_%s.jpg' % (
                    domains[domain])
                filename = name.format(filename)
            else:
                filename = out_file + '_reference_domain_%s.jpg' % (
                    domains[domain])
                filename = name.replace('.jpg', filename)
            translate_using_reference(nets,
                                      args,
                                      x_src,
                                      x_ref,
                                      y_ref,
                                      filename,
                                      multidomain=(y_src, domain),
                                      mask=(m_src, m_ref),
                                      fill_rgb=fill_rgb,
                                      translate_all=translate_all)
            # import ipdb; ipdb.set_trace()
            # translate and reconstruct (reference-guided)
            if not training:
                filename = out_file + 'sample_cycle_consistency_domain_%s.jpg' % (
                    domains[domain])
                filename = name.format(filename)
            else:
                filename = out_file + '_cycle_consistency_domain_%s.jpg' % (
                    domains[domain])
                filename = name.replace('.jpg', filename)

            if args.ATTR == 'single':
                inverse = torch.arange(x_src.size(0) - 1, -1, -1)
                # inverse = torch.arange(0, x_src.size(0), 1)
                x_ref = x_ref[inverse]
                y_ref = y_src[inverse]
                m_ref = m_src[inverse]

            translate_and_reconstruct(nets,
                                      args,
                                      x_src,
                                      y_src,
                                      x_ref,
                                      y_ref,
                                      filename,
                                      multidomain=(y_src, domain),
                                      mask=(m_src, m_ref),
                                      fill_rgb=fill_rgb,
                                      translate_all=translate_all)

            # latent-guided image synthesis
            y_trg_list = y_src.clone()
            y_trg_list[:, domain] = 1
            if args.dataset == 'CelebA_HQ':
                not_domain = 1 - ((domain % 2) * 2)  # -1, 1
                if args.ATTR != 'single':
                    y_trg_list[:, domain + not_domain] = 0
            y_trg_list = [y_trg_list]
            z_trg_list = torch.randn(num_outs_per_domain, 1,
                                     args.noise_dim).repeat(
                                         1, number_of_images, 1)
            z_trg_list = to_cuda(z_trg_list)
            for psi in [0.5, 0.7, 1.0]:
                if not training:
                    filename = out_file + 'sample_latent_psi_%.1f_domain_%s.jpg' % (
                        psi, domains[domain])
                    filename = name.format(filename)
                else:
                    filename = out_file + '_latent_psi_%.1f_domain_%s.jpg' % (
                        psi, domains[domain])
                    filename = name.replace('.jpg', filename)
                translate_using_latent(nets,
                                       args,
                                       x_src,
                                       y_trg_list,
                                       z_trg_list,
                                       psi,
                                       filename,
                                       multidomain=(y_src, domain),
                                       mask=(m_src, m_ref),
                                       fill_rgb=fill_rgb,
                                       translate_all=translate_all)

            # Video
            if not training:
                filename = out_file + 'sample_reference_domain_%s.mp4' % (
                    domains[domain])
                filename = name.format(filename)
                video_ref(nets,
                          args,
                          x_src,
                          x_ref,
                          y_ref,
                          filename,
                          mask=(m_src, m_ref),
                          multidomain=(y_src, domain),
                          fill_rgb=fill_rgb,
                          translate_all=translate_all)
        if not training:
            val_loader.dataset.shuffle(count)

    torch.cuda.empty_cache()


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.25, len_tail=20):
    return [0] + [sigmoid(alpha)
                  for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets,
                args,
                x_src,
                s_prev,
                s_next,
                mask=None,
                multidomain=(),
                translate_all=False,
                fill_rgb=False,
                x_ref=None,
                m_ref=None):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    if args.STYLE_SEMANTICS:
        fan = nets.FAN.get_heatmap(mask) if args.FAN else None
    else:
        fan = nets.FAN.get_heatmap(x_src) if args.FAN else None
    alphas = get_alphas()
    # import ipdb; ipdb.set_trace()
    if multidomain:
        s_src, domain = multidomain
    if not translate_all:
        domain_str = args.domains[domain]
    else:
        domain_str = 'ALL'
    if fill_rgb:
        # attnet = to_cuda(AttNet(verbose=False))
        # masknet = to_cuda(MaskNet(verbose=False))
        smile_syn = fill_rgb
        m_prev, m_next = m_ref
        x_prev, x_next = x_ref

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        s_ref = s_ref.repeat(B, 1, 1)
        if fill_rgb:
            # import ipdb; ipdb.set_trace()
            m_ref = torch.lerp(m_prev, m_next, alpha)
            m_ref = m_ref.repeat(B, 1, 1, 1)
            x_ref = torch.lerp(x_prev, x_next, alpha)
            x_ref = x_ref.repeat(B, 1, 1, 1)
        if multidomain:
            if not translate_all:
                _s_ref = s_src.clone()
                _s_ref[:, domain // 2] = s_ref[:, domain // 2]
                s_ref = _s_ref
        x_fake = nets.G(x_src, s_ref, fan=fan)
        if args.STYLE_SEMANTICS:
            if fill_rgb:
                # import ipdb; ipdb.set_trace()
                x_fake_smile = scatterMask(
                    label2mask_plain(x_fake)[:, 0], x_fake.size(1))
                # _m_ref = m_ref[i].unsqueeze(0).repeat(N,1,1,1)
                # _x_ref = x_ref[i].unsqueeze(0).repeat(N,1,1,1)
                # _m_ref = m_ref
                # _x_ref = x_ref
                # if domain_str == 'General_Style':
                #     sty_ref_missing_part = smile_syn.model.encoder(m_ref, x_ref)
                #     rgb = smile_syn.forward_from_tensor(
                #         x_fake_smile,
                #         rgb_guide=mask,
                #         sem_guide=x_src,
                #         style_ref=sty_ref_missing_part)
                # else:
                # sty_ref_missing_part = smile_syn.model.encoder(mask, x_src)
                sty_ref_missing_part = smile_syn.model.encoder(m_ref, x_ref)
                rgb = smile_syn.forward_from_tensor(
                    x_fake_smile,
                    rgb_guide=mask,
                    sem_guide=x_src,
                    style_ref=sty_ref_missing_part,
                    domain=domain_str)
                # rgb = smile_syn.forward_from_tensor(
                #     x_fake_smile,
                #     rgb_guide=m_ref,
                #     sem_guide=x_ref,
                #     style_ref=sty_ref_missing_part,
                #     domain=domain_str)
                x_fake = label2mask(x_fake, n=x_fake.size(1))
                _x_fake = scale_image(rgb, x_fake, None, size=64)
                _x_src = label2mask(x_src, n=x_src.size(1))
                _x_src = scale_image(denorm(mask), _x_src, None, size=64)
            else:
                _x_fake = ((label2mask(x_fake, n=x_fake.size(1)) - 0.5) *
                           2.).clamp_(-1, 1)
                _x_src = ((label2mask(x_src, n=x_src.size(1)) - 0.5) *
                          2.).clamp_(-1, 1)
        else:
            _x_fake, _x_src = x_fake, x_src
        entries = torch.cat([_x_src, _x_fake], dim=2)
        frame = torchvision.utils.make_grid(entries.cpu(),
                                            nrow=B,
                                            padding=0,
                                            pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next,
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas)  # number of frames

    canvas = -torch.ones((T, C, H * 2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets,
              args,
              x_src,
              x_ref,
              y_ref,
              fname,
              mask,
              multidomain=(),
              translate_all=False,
              fill_rgb=False):
    video = []
    s_ref = nets.S(x_ref, y_ref)
    if multidomain:
        y_src, domain = multidomain
        s_src = nets.S(x_src, y_src)
        multidomain = s_src, domain
    m_src, m_ref = mask
    s_prev = None
    if fill_rgb:
        smile_syn = to_cuda(SMILE(args.image_size, verbose=False))
    else:
        smile_syn = False
    for data_next in tqdm(zip(x_ref, m_ref, y_ref, s_ref), 'video_ref',
                          len(x_ref)):
        x_next, m_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, m_prev, y_prev, s_prev = x_next, m_next, y_next, s_next
            continue
        # if y_prev != y_next:
        #     x_prev, y_prev, s_prev = x_next, y_next, s_next
        #     continue
        interpolated = interpolate(nets,
                                   args,
                                   x_src,
                                   s_prev,
                                   s_next,
                                   mask=m_src,
                                   x_ref=(x_prev, x_next),
                                   m_ref=(m_prev, m_next),
                                   multidomain=multidomain,
                                   translate_all=translate_all,
                                   fill_rgb=smile_syn)
        if args.STYLE_SEMANTICS:
            if fill_rgb:
                _x_prev = label2mask(x_prev, n=x_prev.size(1))
                _x_prev = scale_image(denorm(m_prev), _x_prev, None, size=64)
                _x_next = label2mask(x_next, n=x_next.size(1))
                _x_next = scale_image(denorm(m_next), _x_next, None, size=64)
            else:
                _x_prev = ((label2mask(x_prev, n=x_prev.size(1)) - 0.5) *
                           2.).clamp_(-1, 1)
                _x_next = ((label2mask(x_next, n=x_next.size(1)) - 0.5) *
                           2.).clamp_(-1, 1)
        else:
            _x_prev, _x_next = x_prev, x_next
        entries = [_x_prev, _x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated],
                           dim=3)  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        # import ipdb; ipdb.set_trace()
        # save_img(frames, 1, 'dummy.jpg',denorm=False)
        x_prev, m_prev, y_prev, s_prev = x_next, m_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video).cpu(), denorm=not fill_rgb)
    # import ipdb; ipdb.set_trace()
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets,
                 args,
                 x_src,
                 psi,
                 y_trg_list,
                 z_trg_list,
                 fname,
                 mask,
                 multidomain=(),
                 translate_all=False,
                 fill_rgb=False):
    latent_dim = z_list[0].size(1)
    s_list = []

    if multidomain:
        y_src, domain = multidomain
        s_src = nets.S(x_src, y_src)
        multidomain = s_src, domain
    m_src, m_ref = mask

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim)
        z_many = to_cuda(z_many)
        y_many = torch.LongTensor(10000).fill_(y_trg[0])
        y_many = to_cuda(y_many)
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []

    s_prev = None
    if fill_rgb:
        smile_syn = to_cuda(SMILE(args.image_size, verbose=False))
    else:
        smile_syn = False
    for data_next in tqdm(zip(x_ref, m_ref, y_trg_list, s_list),
                          'video_latent', len(x_ref)):
        x_next, m_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, m_prev, y_prev, s_prev = x_next, m_next, y_next, s_next
            continue

        interpolated = interpolate(nets,
                                   args,
                                   x_src,
                                   s_prev,
                                   s_next,
                                   mask=m_src,
                                   x_ref=(x_prev, x_next),
                                   m_ref=(m_prev, m_next),
                                   multidomain=multidomain,
                                   translate_all=translate_all,
                                   fill_rgb=smile_syn)

        video.append(interpolated)
        x_prev, m_prev, y_prev, s_prev = x_next, m_next, y_next, s_next

    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video), denorm=fill_rgb)
    save_video(fname, video)


def save_video(fname, images, output_fps=15, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:',
                          format='rawvideo',
                          pix_fmt='rgb24',
                          s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts',
                           '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream,
                           fname,
                           pix_fmt='yuv420p',
                           vcodec=vcodec,
                           r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images, denorm=True):
    if denorm:
        images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255
