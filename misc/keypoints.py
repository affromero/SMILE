"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Adapted from
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
"""

from collections import namedtuple
from copy import deepcopy
from functools import partial

from munch import Munch
import numpy as np
import cv2
from skimage.filters import gaussian
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.pose_hrnet import get_pose_net

import argparse
import csv
import os
import shutil

from PIL import Image
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
import sys
import time

# import _init_paths
import matplotlib.pyplot as plt

from yacs.config import CfgNode as CN


class Pose(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = CN()
        self.cfg = cfg.load_cfg(open('misc/pose_hrnet.yaml'))
        self.pose_model = get_pose_net(self.cfg)
        self.pose_model.eval()
        # resize = self.cfg.MODEL.IMAGE_SIZE
        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.box_model.eval()

        self.pose_transform = transforms.Compose([
            # transforms.Resize(resize,
            #                 interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        if self.cfg.TEST.MODEL_FILE:
            print('=> loading Keypoints from {}'.format(
                self.cfg.TEST.MODEL_FILE))
            self.pose_model.load_state_dict(torch.load(
                self.cfg.TEST.MODEL_FILE,
                map_location=lambda storage, loc: storage),
                                            strict=False)
        else:
            raise TypeError(
                'expected model defined in config at TEST.MODEL_FILE')

    @torch.no_grad()
    def forward(self, x):
        from misc.utils import denorm
        device = x.device
        crop_images = []
        heatmaps_batch = []
        for image in x:
            image_org = image.clone()
            image = denorm(image.unsqueeze(0))
            import ipdb
            ipdb.set_trace()
            image_pose = (image.clone().cpu().numpy().transpose(1, 2, 0) *
                          255.).astype(np.uint8)
            pred_boxes = get_person_detection_boxes(self.box_model,
                                                    image,
                                                    threshold=0.9)
            center, scale = box_to_center_scale(box,
                                                self.cfg.MODEL.IMAGE_SIZE[0],
                                                self.cfg.MODEL.IMAGE_SIZE[1])
            centers = []
            scales = []
            box = pred_boxes[0]
            center, scale = box_to_center_scale(box,
                                                self.cfg.MODEL.IMAGE_SIZE[0],
                                                self.cfg.MODEL.IMAGE_SIZE[1])
            centers = (center)
            scales = (scale)

            pose_preds = get_pose_estimation_prediction(
                pose_model,
                image_pose,
                centers,
                scales,
                transform=self.pose_transform)
            heatmaps = get_heatmap(pose_preds, image_pose.shape[:2])
            heatmaps = torch.from_numpy(heatmaps).unsqueeze(0)
            crop_images.append(crop_bbox(image_org, box))
            heatmaps = crop_bbox(heatmaps, box)
            heatmaps_batch.append(heatmaps)

        heatmaps_batch = torch.cat(heatmaps_batch, dim=0)
        heatmaps_batch = heatmaps_batch.to(device)
        crop_images = torch.stack(crop_images, dim=0).to(device)
        return crop_images, heatmaps_batch

    def get_heatmap(self, x):
        return self(x)

    @torch.no_grad()
    def get_visualization(self, x, current=False, dirname='.'):
        from misc.utils import denorm, save_img
        import os
        org_x = denorm(x.clone())
        x0 = self.get_heatmap(x)
        x0 = F.interpolate(x0, size=org_x.size(-1), mode='bilinear')
        x0 = (x0 * org_x).expand_as(org_x)
        list_h = torch.cat([org_x, x0], dim=3)
        file_h = os.path.join(dirname, 'keypoints.jpg')
        print(f'Saving Keypoints visualization on: {file_h}.')
        save_img(list_h, 3, file_h, denormalize=False)
        self.general_attr = mode


# -------------------------------------

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def crop_bbox(tensor, bbox, margin=(0, 0)):
    y_axis = [int(box[0][1]) - margin[0], int(box[1][1]) + margin[0]]
    x_axis = [int(box[0][0]) - margin[1], int(box[1][0]) + margin[1]]
    return image[:, y_axis[0]:y_axis[1], x_axis[0]:x_axis[1]]


def get_person_detection_boxes(model, img, threshold=0.5, transform=True):
    if transform:
        pil_image = Image.fromarray(img)  # Load the image
        transform = transforms.Compose([transforms.ToTensor()
                                        ])  # Defing PyTorch Transform
        transformed_img = transform(
            pil_image)  # Apply the transform to the image
        img = [transformed_img.to(CTX)]
    pred = model(img)  # Pass the image to the model
    # Use the first detected person
    pred_classes = [
        COCO_INSTANCE_CATEGORY_NAMES[i]
        for i in list(pred[0]['labels'].cpu().numpy())
    ]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())
                  ]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes,
                                                pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales,
                                   transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation,
                                     self.cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(image,
                                     trans,
                                     (int(self.cfg.MODEL.IMAGE_SIZE[0]),
                                      int(self.cfg.MODEL.IMAGE_SIZE[1])),
                                     flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)  # .unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(cfg,
                                output.cpu().detach().numpy(),
                                np.asarray(centers), np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Showing keypoints network')
    # general
    parser.add_argument('--imageFile', type=str, required=True)
    parser.add_argument('--writeBoxFrames', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    args = parse_args()

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = get_pose_net(cfg)

    if self.cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(self.cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            self.cfg.TEST.MODEL_FILE,
            map_location=lambda storage, loc: storage),
                                   strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    # import ipdb; ipdb.set_trace()
    pose_model.eval()

    # Loading an image
    image_bgr = cv2.imread(args.imageFile)
    # image_bgr = image_bgr[:300]
    frame_width = int(image_bgr.shape[1])
    frame_height = int(image_bgr.shape[0])

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Clone 2 image for person detection and pose estimation
    image_per = image_rgb.copy()
    image_pose = image_rgb.copy()

    # Clone 1 image for debugging purpose
    image_debug = image_bgr.copy()

    # object detection box
    total_now = time.time()
    now = time.time()
    pred_boxes = get_person_detection_boxes(box_model,
                                            image_per,
                                            threshold=0.9)
    then = time.time()
    print("Find person bbox in: {} sec".format(then - now))

    # Can not find people. Move to next frame
    if not pred_boxes:
        count += 1
        print('No people')
        return

    if args.writeBoxFrames:
        for box in pred_boxes:
            cv2.rectangle(image_debug,
                          box[0],
                          box[1],
                          color=(0, 255, 0),
                          thickness=3)  # Draw Rectangle with the coordinates

    # pose estimation : for multiple people
    centers = []
    scales = []
    for box in pred_boxes:
        center, scale = box_to_center_scale(box, self.cfg.MODEL.IMAGE_SIZE[0],
                                            self.cfg.MODEL.IMAGE_SIZE[1])
        centers.append(center)
        scales.append(scale)

    now = time.time()
    pose_preds = get_pose_estimation_prediction(pose_model,
                                                image_pose,
                                                centers,
                                                scales,
                                                transform=pose_transform)

    then = time.time()
    print("Find person pose in: {} sec".format(then - now))

    image_heatmap = np.zeros(
        (image_debug.shape[0], image_debug.shape[1])).astype(np.float32)
    for coords in pose_preds:
        # Draw each point on image
        for idx, coord in enumerate(coords):
            if idx < 5:
                cov = 80
            else:
                cov = 20
            if (coord[0] in coords[:idx, 0]
                    or coord[0] in coords[idx + 1:, 0]) and not idx < 5:
                # non-regular keypoint
                continue

            if (coord[1] in coords[:idx, 1]
                    or coord[1] in coords[idx + 1:, 1]) and not idx < 5:
                # non-regular keypoint
                continue
            x_coord, y_coord = int(coord[0]), int(coord[1])
            _image_heatmap = get_gaussian_coord(y_coord,
                                                x_coord,
                                                size=image_heatmap.shape,
                                                cov=cov)
            # import ipdb; ipdb.set_trace()
            image_heatmap += _image_heatmap
            cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)

    total_then = time.time()

    text = "{:03.2f} sec".format(total_then - total_now)
    cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA)

    image_debug = cv2.cvtColor(image_debug, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(image_debug)
    plt.subplot(1, 2, 2)
    image_heatmap = image_heatmap.clip(min=0, max=1)
    # import ipdb; ipdb.set_trace()
    plt.imshow(image_heatmap)
    plt.show()


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    # if config.TEST.POST_PROCESS:
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([
                    hm[py][px + 1] - hm[py][px - 1],
                    hm[py + 1][px] - hm[py - 1][px]
                ])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_heatmap(pose_preds, size):
    ''' outputs 0-1 normalized heatmap '''
    image_heatmap = np.zeros(size).astype(np.float32)
    for coords in pose_preds:
        # Draw each point on image
        for idx, coord in enumerate(coords):
            if idx < 5:
                cov = 80
            else:
                cov = 20
            if (coord[0] in coords[:idx, 0]
                    or coord[0] in coords[idx + 1:, 0]) and not idx < 5:
                # non-regular keypoint
                continue

            if (coord[1] in coords[:idx, 1]
                    or coord[1] in coords[idx + 1:, 1]) and not idx < 5:
                # non-regular keypoint
                continue
            x_coord, y_coord = int(coord[0]), int(coord[1])
            _image_heatmap = get_gaussian_coord(y_coord,
                                                x_coord,
                                                size=image_heatmap.shape,
                                                cov=cov)
            # import ipdb; ipdb.set_trace()
            image_heatmap += _image_heatmap
    return heatmaps


def get_gaussian_coord(center_y=None,
                       center_x=None,
                       size=None,
                       cov=4,
                       show=False):
    from scipy.stats import multivariate_normal
    import numpy as np
    import cv2
    # Probability as a function of distance from the center derived
    # from a gaussian distribution with mean = 0 and stdv = 1
    if size is None:
        size = (512, 512)
    image = np.zeros(size, np.uint8)

    if center_x is None:
        center_x = image.shape[1] // 2

    if center_y is None:
        center_y = image.shape[0] // 2

    # Numpy
    pos = np.dstack(np.mgrid[0:image.shape[0]:1, 0:image.shape[1]:1])

    # Pytorch
    # pos = torch.stack(torch.meshgrid([torch.arange(0,image.shape[1]), torch.arange(0,image.shape[0])]), -1)

    # Scipy
    normal = multivariate_normal(mean=[center_y, center_x], cov=cov)
    HeatmapImage = normal.pdf(pos)
    # import ipdb; ipdb.set_trace()
    HeatmapImage = HeatmapImage / HeatmapImage.max()

    # Pytorch - it does not work
    # from torch.distributions.multivariate_normal import MultivariateNormal
    # normal = MultivariateNormal(torch.FloatTensor([center_y,center_x]), torch.eye(2)*cov)

    if show:
        plt.imshow(HeatmapImage)
        plt.show()
    return HeatmapImage


if __name__ == '__main__':
    main()
