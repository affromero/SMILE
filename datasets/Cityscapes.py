from cityscapesscripts.helpers import labels as LABELS
import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
from torchvision import transforms
from glob import glob
from misc.mask_utils import scatterMask
from munch import Munch
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)
# ==================================================================#
# == VOC2012
# ==================================================================#

ATTR = ['background']
ATTR += [label.name for label in LABELS.labels[6:-1]]

MASK_LABELS = {i: idx for idx, i in enumerate(ATTR)}
SEMANTIC_ATTRS = {i.capitalize(): [i] for i in ATTR}
MASK_ATTR = {i: [i] for i in ATTR}


class Cityscapes(Dataset):
    def __init__(self,
                 image_size,
                 transform,
                 mode,
                 shuffling=False,
                 all_attr=0,
                 verbose=False,
                 sampled=100,
                 show_attr='',
                 train_mask=False,
                 create_labels=False,
                 **kwargs):
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        self.name = self.__class__.__name__
        self.all_attr = all_attr
        self.verbose = verbose
        self.show_attr = show_attr

        self.root = 'data'
        self.all_masks = sorted(
            glob(
                'data/Cityscapes/gtFine_trainvaltest/gtFine/{}/*/*_gtFine_labelIds.png'
                .format(mode)))
        # self.all_filenames = sorted(
        #     glob('data/Cityscapes/leftImg8bit/{}/*/*leftImg8bit.png'.format(mode)))
        # self.mode_allowed = [0, 1, 2]
        # import ipdb; ipdb.set_trace()

        self.mask_label = MASK_LABELS
        self.semantic_attr = SEMANTIC_ATTRS
        self.mask_attr = MASK_ATTR

        self.transform_resize_img = transform.resize_rgb
        self.transform_resize_mask = transform.resize_mask
        self.transform_common = transform.common
        self.transform = transform
        self.create_labels = create_labels
        self.all_attr2idx = {}
        self.all_idx2attr = {}
        self.attr2idx = {}
        self.idx2attr = {}
        self.attr2filenames = {}
        self.NOTattr2filenames = {}
        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        if 'config' in kwargs.keys():
            self.config = kwargs['config']
            self.config.mask_dim = max([label.id
                                        for label in LABELS.labels]) - 4
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            self.config.MASK = False
            self.config.TRAIN_MASK = False

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        self.preprocess()
        if self.verbose:
            _str = str(self.num_data)
            print('Finished preprocessing %s: %s (%s)!' %
                  (self.name, mode, _str))

    def transform_cityscapes(self, mask):
        # mask = (mask - 4).clamp(min=0)
        labels = mask.unique()
        # import ipdb; ipdb.set_trace()
        new_labels = torch.zeros_like(mask)
        for ll in labels:
            ll = int(ll.item())
            cityscapes_attr = LABELS.id2label[ll][0]
            if cityscapes_attr not in self.selected_attrs:
                continue
            index = self.mask_label[cityscapes_attr]
            new_labels[mask == ll] = index
        return new_labels

    def histogram(self, labels):
        from misc.utils import PRINT
        values = np.array(labels[0]) * 0
        n_images = 0
        for line in labels:
            values += np.array([int(i) for i in line]).clip(min=0)
            n_images += 1
        dict_ = {}
        for key, value in zip(self.selected_attrs, values):
            dict_[key] = value
        # total = 0
        with open('datasets/{}_histogram_attributes.txt'.format(self.name),
                  'w') as f:
            for key, value in sorted(dict_.items(),
                                     key=lambda kv: (kv[1], kv[0]),
                                     reverse=True):
                # total += value
                PRINT(f, '{} {}'.format(key, value))
            PRINT(f, 'TOTAL {} images'.format(n_images))

    def preprocess(self):
        if self.show_attr != '':
            self.selected_attrs = self.show_attr
        else:
            self.selected_attrs = [
                # 'ground',
                # 'road',
                # 'sidewalk',
                # 'building',
                # 'tunnel',
                'traffic light',
                'traffic sign',
                'vegetation',
                'terrain',
                'sky',
                'person',
                'rider',
                'car',
                'truck',
                'bus',
                'train',
                'motorcycle',
                'bicycle',
            ]
            # list(self.mask_label.keys())
            self.parent_attrs = {i: [i] for i in self.selected_attrs}
            self.children_attrs = self.selected_attrs
        # make sure children and parents has the same order
        for i, attr in enumerate(self.selected_attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
            self.attr2filenames[attr] = []
            self.NOTattr2filenames[attr] = []
        self.labels = []
        self.filenames = []
        self.mask = []
        for i in range(len(self.all_masks)):
            seg = self.all_masks[i]
            filename = seg.replace('_gtFine_labelIds', '_leftImg8bit')
            filename = filename.replace('gtFine_trainvaltest', 'leftImg8bit')
            filename = filename.replace('gtFine/', '')
            # import ipdb; ipdb.set_trace()
            if not self.create_labels:
                values_sem = [
                    i.strip()
                    for i in open(seg.replace('.png', '.txt')).readlines()
                ]
                # values_sem = self.get_mask_from_file(seg, label=True)
                # values_sem = values_sem.unique()

                label = []
                # import ipdb; ipdb.set_trace()
                no_show_attr = True
                for attr in self.selected_attrs:
                    # index = list(self.mask_label.keys()).index(attr.lower())
                    # selected_value = int(index in values_sem)
                    selected_value = int(attr in values_sem)
                    if selected_value >= 1:
                        label.append(selected_value)
                        self.attr2filenames[attr].append(filename)
                        no_show_attr = False
                    else:
                        label.append(0)
                        self.NOTattr2filenames[attr].append(filename)

                if self.show_attr and no_show_attr:
                    continue

                if 1 not in label and self.mode == 'train':
                    continue
            else:
                label = [0]
            self.labels.append(label)
            self.filenames.append(filename)
            self.mask.append(seg)
            if self.show_attr and i > 300 and not self.create_labels:
                break

        self.filenames_ref = list(self.filenames)
        random.seed(1)
        random.shuffle(self.filenames_ref)
        self.labels_ref = list(self.labels)
        random.seed(1)
        random.shuffle(self.labels_ref)
        self.mask_ref = list(self.mask)
        random.seed(1)
        random.shuffle(self.mask_ref)

        self.num_data = len(self.filenames)
        if self.verbose and not self.create_labels:
            self.histogram(self.labels)

    def get_data(self):
        return self.filenames, self.labels

    def get_mask_from_file(self, filename, label=False):
        mask = Image.open(filename)
        mask = self.transform_resize_mask(mask)
        mask = self.transform_common(mask)[0] * 255.  # 0, 255
        if self.create_labels:
            mask = self.transform_cityscapes(mask)
        else:
            mask = (mask - 5).clamp(min=0)
        # mask = self.transform_mask(mask)[0] * 255.  # 0, 255
        if self.show_attr:
            # import ipdb; ipdb.set_trace()
            # labels_real = mask.unsqueeze(0) #
            labels_real = self.get_partial_mask(mask).unsqueeze(0)
        elif label:
            labels_real = mask
        else:
            labels_real = scatterMask(mask, num_channels=self.config.mask_dim)
        # labels_real: C x size x size
        return labels_real  # 20 attrs

    def get_partial_mask(self, mask):
        new_mask = torch.zeros_like(mask)
        for attr in self.selected_attrs:
            # import ipdb; ipdb.set_trace()
            label = self.mask_label[attr]
            new_mask[mask == label] = label

        return new_mask

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = self.file2img(filename)
        label = torch.FloatTensor(self.labels[index])
        mask = self.mask[index]
        mask = self.get_mask_from_file(mask)
        if self.config.TRAIN_MASK:
            _mask = image
            image = mask
            mask = _mask
        # return image, label, mask, filename

        filename_ref = self.filenames_ref[index]
        image_ref = self.file2img(filename_ref)
        label_ref = torch.FloatTensor(self.labels_ref[index])
        mask_ref = self.get_mask_from_file(self.mask_ref[index])
        if self.config.TRAIN_MASK:  # or self.config.ONLY_GEN:
            _mask_ref = image_ref
            image_ref = mask_ref
            mask_ref = _mask_ref

        if self.create_labels:
            return image, label, mask, self.mask[index]
        else:
            return image, label, mask, (image_ref, label_ref, mask_ref)

    def file2img(self, filename):
        image = Image.open(filename).convert('RGB')
        image = self.transform_resize_img(image)
        image = self.transform_common(image)
        image = self.transform.norm(image)
        return image

    def __len__(self):
        return self.num_data

    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.filenames)
        random.seed(seed)
        random.shuffle(self.labels)
        random.seed(seed)
        random.shuffle(self.mask)


def show_me(args):
    from data_loader import get_transformations
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from misc.utils import denorm
    import numpy as np
    import matplotlib.pyplot as plt
    from misc.mask_utils import label2mask
    from tqdm import tqdm
    attrs = args.attr.split(',') if not args.create_labels else ATTR
    assert len(attrs) >= 1, "Please provide at least one attribute"
    mode = 'train'
    transform = get_transformations('test', 256)
    data = Cityscapes(256,
                      transform,
                      mode,
                      show_attr=attrs,
                      create_labels=args.create_labels,
                      verbose=True)
    data_loader = DataLoader(dataset=data,
                             batch_size=64,
                             shuffle=False,
                             num_workers=0 if not args.create_labels else 4)
    if args.create_labels:
        loop = tqdm(enumerate(data_loader), total=len(data_loader), ncols=30)
    else:
        loop = enumerate(data_loader)
    for i, (data, label, mask, filename) in loop:
        if args.create_labels:
            for _mask, _filename in zip(mask, filename):
                _mask = _mask.unique().long().tolist()
                # import ipdb; ipdb.set_trace()
                attr = '\n'.join([ATTR[i] for i in _mask])
                assert '.png' in _filename
                with open(_filename.replace('.png', '.txt'), 'w') as f:
                    f.writelines(attr)
            continue

        data = denorm(data)
        # up_size = 1024
        # data = resize(data, up_size)
        if args.MASK:
            alpha = 0.5
            # import ipdb; ipdb.set_trace()
            mask = label2mask(mask, n=len(ATTR))
            # mask = resize(mask, up_size, 'nearest')
            data = ((1 - alpha) * data + alpha * mask)

        data = make_grid(data).numpy()
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(data, (1, 2, 0)), interpolation='nearest')
        plt.axis('off')
        plt.show(block=True)


def resize(tensor, up_size, mode='bilinear'):
    import torch.nn.functional as F
    return F.interpolate(tensor, size=(up_size, up_size), mode=mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='')
    parser.add_argument('--MASK', action='store_true', default=False)
    parser.add_argument('--create_labels', action='store_true', default=False)
    args = parser.parse_args()
    show_me(args)
