import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms.functional as F


def rgb2tensor(img, normalize=True):
    """ Converts a RGB image to tensor.

    Args:
        img (np.array or list of np.array): RGB image of shape (H, W, 3) or a list of images
        normalize (bool): If True, the tensor will be normalized to the range [-1, 1]

    Returns:
        torch.Tensor or list of torch.Tensor: The converted image tensor or a list of converted tensors.
    """
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(o) for o in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)


def bgr2tensor(img, normalize=True):
    """ Converts a BGR image to tensor.

    Args:
        img (np.array or list of np.array): BGR image of shape (H, W, 3) or a list of images
        normalize (bool): If True, the tensor will be normalized to the range [-1, 1]

    Returns:
        torch.Tensor or list of torch.Tensor: The converted image tensor or a list of converted tensors.
    """
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(o, normalize) for o in img]
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)


def tensor2rgb(img_tensor):
    """ Convert an image tensor to a numpy RGB image.

    Args:
        img_tensor (torch.Tensor): Tensor image of shape (3, H, W)

    Returns:
        np.array: RGB image of shape (H, W, 3)
    """
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')

    return output_img


def tensor2bgr(img_tensor):
    """ Convert an image tensor to a numpy BGR image.

    Args:
        img_tensor (torch.Tensor): Tensor image of shape (3, H, W)

    Returns:
        np.array: BGR image of shape (H, W, 3)
    """
    output_img = tensor2rgb(img_tensor)
    output_img = output_img[:, :, ::-1]

    return output_img


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def blend_imgs_bgr(source_img, target_img, mask):
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) -
                                               np.min(a[1])) <= 10:
        return target_img

    center = (np.min(a[1]) + np.max(a[1])) // 2, (np.min(a[0]) +
                                                  np.max(a[0])) // 2
    output = cv2.seamlessClone(source_img, target_img, mask * 255, center,
                               cv2.NORMAL_CLONE)

    return output


def blend_imgs(source_tensor, target_tensor, mask_tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = tensor2bgr(source_tensor[b])
        target_img = tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].squeeze().cpu().numpy()
        out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)
