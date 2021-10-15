import torch
import numpy as np
import torch.nn.functional as F
from misc.utils import to_cuda
# from cityscapesscripts.helpers import labels as LABELS


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 19:  # CelebAMask-HQ
        cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0), (204, 204, 0),
                         (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0),
                         (102, 204, 0), (255, 255, 0), (0, 0, 153),
                         (0, 0, 204), (255, 51, 153), (0, 204, 204),
                         (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
    elif N == 18:  # Flickr
        cmap = np.array([(32, 32, 32), (204, 0, 0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204),
                         (0, 255, 255), (51, 255, 255), (102, 51, 0),
                         (255, 0, 0), (102, 204, 0), (255, 255, 0),
                         (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51)],
                        dtype=np.uint8)
    elif N == 29:  # CityScapes
        colors = [LABELS.labels[4].color]
        colors += [label.color for label in LABELS.labels[6:-1]]
        cmap = np.array(colors, dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def tensor2label(label_tensor, n_label=19, imtype=np.uint8):
    assert len(
        label_tensor.shape) == 3, f"One image at a time. {label_tensor.shape}."
    if n_label == 0:
        raise NotImplementedError
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size(0) > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor) / 255.
    # label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    # label_numpy = label_tensor.numpy()
    # label_numpy = label_numpy / 255.0

    return label_tensor


def scatterMask(mask, num_channels=19, single=False):
    if len(mask.shape) == 2:
        bs = 1
        mask = mask.unsqueeze(0)
        one = True
    else:
        assert len(mask.shape) == 3, "bs, w, h"
        bs = mask.size(0)
        one = False
    oneHot_size = (bs, num_channels, mask.size(-2), mask.size(1))
    labels_real = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    labels_real = labels_real.to(mask.device)
    for i in range(bs):
        labels_real[i] = labels_real[i].scatter_(
            0, mask[i, None, :, :].data.long(), 1.0)
    if one:
        labels_real = labels_real[0]
    return labels_real


def mix_mask(tensor, mask, alpha=0.7):
    return (tensor * alpha) + ((1 - alpha) * mask)


def compare_mask(real_x, mask, fake_mask, only_mask=False):
    from misc.mask_utils import label2mask
    from misc.utils import denorm, imshow_fromtensor
    if only_mask:
        real_mix = label2mask(mask)
        fake_mix = label2mask(fake_mask)
    else:
        real_mix = mix_mask(denorm(real_x), label2mask(mask))
        fake_mix = mix_mask(denorm(real_x), label2mask(fake_mask))
    xx = torch.cat((real_mix, fake_mix), dim=-1)
    imshow_fromtensor(xx, nrows=1)


def label2mask(inputs, n=19):
    if inputs.size(1) == n:
        pred_batch = label2mask_plain(inputs)
    else:
        pred_batch = inputs

    label_batch = []
    for p in pred_batch:
        label_batch.append(tensor2label(p, n))

    label_batch = torch.stack(label_batch, dim=0).to(inputs.device)

    return label_batch


def mask2label(inputs):
    device = inputs.device
    pred_batch = []
    for input in inputs:
        pred = input.data.max(1)[0].max(1)[0]
        pred_batch.append(pred)

    pred_batch = torch.stack(pred_batch, dim=0)
    pred_batch = pred_batch.to(device)
    return pred_batch


def mask2unique(inputs, n=19):
    device = inputs.device
    pred_batch = []
    for input in inputs:
        pred = input.data.max(0, keepdim=True)[1].cpu()
        one_hot = []
        _unique = pred.unique()
        one_hot = torch.zeros(n)
        for i in range(n):
            one_hot[i] = (i in _unique) * 1

        pred_batch.append(one_hot)

    pred_batch = torch.stack(pred_batch, dim=0)
    pred_batch = pred_batch.to(device)
    return pred_batch


def label2mask_plain(inputs):
    device = inputs.device
    pred_batch = []
    for input in inputs:
        pred = input.data.max(0, keepdim=True)[1].cpu()
        pred_batch.append(pred)

    pred_batch = torch.stack(pred_batch, dim=0)
    pred_batch = pred_batch.to(device)
    return pred_batch


def cross_entropy2d(input,
                    target,
                    weight=None,
                    size_average=True,
                    pascal=False):
    # import ipdb; ipdb.set_trace()
    if not pascal:
        target = label2mask_plain(target).long()
    bs, c, h, w = input.size()
    nt, _, ht, wt = target.size()

    # Handle inconsistent size between input and target

    if h != ht or w != wt:
        input = F.interpolate(input,
                              size=(ht, wt),
                              mode="bilinear",
                              align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    loss = F.cross_entropy(input,
                           target,
                           weight=weight,
                           size_average=size_average,
                           ignore_index=255)
    #    ignore_index=250)
    return loss


# ==================================================================#
# ==================================================================#
def generate_mask(masknet,
                  x,
                  domain,
                  style='',
                  fan=None,
                  seg2=None,
                  seed=False,
                  mix_attr=0):
    if not hasattr(masknet, 'G_mask'):
        return x

    if seed:
        seed1, seed2 = 1, 2
    else:
        seed1 = seed2 = None

    def get_style(_style, seed=None):
        assert isinstance(_style, (str, torch.Tensor))
        if isinstance(_style, str):
            z_fake = to_cuda(masknet.G_mask.random_noise(domain, seed=seed))
            style_m = masknet.G_mask.Noise2Style(z_fake)
        else:
            style_m = masknet.S_mask(_style)
        return style_m

    style_matrix = get_style(style, seed=seed1)
    if mix_attr:
        _style = get_style(seg2, seed=seed2)
        style_matrix[:, mix_attr - 1] = _style[:, mix_attr - 1]
    return masknet.G_mask(x, domain=domain, style=style_matrix, fan=fan)


def pre_removing_mask(segmap):
    # replacing mouth and lips for skin
    segmap = segmap.clone()
    indexes = torch.nonzero(segmap[:, 10:13] == 1)  # bs, ch, y, x
    segmap[indexes[:, 0], indexes[:, 1] + 10, indexes[:, 2], indexes[:, 3]] = 0
    # segmap[indexes[:, 0], 1, indexes[:, 2], indexes[:, 3]] = 1

    # replacing hair for background
    indexes = torch.nonzero(segmap[:, 13] == 1)  # bs, ch, y, x
    segmap[indexes[:, 0], 13, indexes[:, 1], indexes[:, 2]] = 0
    # segmap[indexes[:, 0], 0, indexes[:, 1], indexes[:, 2]] = 1

    # replacing eyeglasses for skin
    indexes = torch.nonzero(segmap[:, 3] == 1)  # bs, ch, y, x
    if len(indexes) > 1:
        segmap[indexes[:, 0], 3, indexes[:, 1], indexes[:, 2]] = 0
        # segmap[indexes[:, 0], 1, indexes[:, 1], indexes[:, 2]] = 1

    # replacing hat for background
    # indexes = torch.nonzero(segmap[:, 14] == 1) # bs, ch, y, x
    # if len(indexes) > 1:
    #     segmap[indexes[:,0], 14, indexes[:,1], indexes[:,2]] = 0
    #     segmap[indexes[:,0], 0, indexes[:,1], indexes[:,2]] = 1

    return segmap
