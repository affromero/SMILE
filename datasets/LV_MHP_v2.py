import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
import json
from glob import glob
from PIL import ImageDraw
from misc.mask_utils import scatterMask
from misc.utils import denorm
import glob
from scipy.io import loadmat
from tqdm import tqdm
module_path = os.path.abspath(os.getcwd())

if module_path not in sys.path:
    sys.path.append(module_path)

# ==================================================================#
# == CelebA
# ==================================================================#

MASK_LABELS = {
    0: 'Background',
    1: 'Cap/hat',
    2: 'Helmet',
    3: 'Face',
    4: 'Hair',
    5: 'Left-arm',
    6: 'Right-arm',
    7: 'Left-hand',
    8: 'Right-hand',
    9: 'Protector',
    10: 'Bikini/bra',
    11: 'Jacket/windbreaker/hoodie',
    12: 'Tee-shirt',
    13: 'Polo-shirt',
    14: 'Sweater',
    15: 'Singlet',
    16: 'Torso-skin',
    17: 'Pants',
    18: 'Shorts/swim-shorts',
    19: 'Skirt',
    20: 'Stockings',
    21: 'Socks',
    22: 'Left-boot',
    23: 'Right-boot',
    24: 'Left-shoe',
    25: 'Right-shoe',
    26: 'Left-highheel',
    27: 'Right-highheel',
    28: 'Left-sandal',
    29: 'Right-sandal',
    30: 'Left-leg',
    31: 'Right-leg',
    32: 'Left-foot',
    33: 'Right-foot',
    34: 'Coat',
    35: 'Dress',
    36: 'Robe',
    37: 'Jumpsuit',
    38: 'Other-full-body-clothes',
    39: 'Headwear',
    40: 'Backpack',
    41: 'Ball',
    42: 'Bats',
    43: 'Belt',
    44: 'Bottle',
    45: 'Carrybag',
    46: 'Cases',
    47: 'Sunglasses',
    48: 'Eyewear',
    49: 'Glove',
    50: 'Scarf',
    51: 'Umbrella',
    52: 'Wallet/purse',
    53: 'Watch',
    54: 'Wristband',
    55: 'Tie',
    56: 'Other-accessary',
    57: 'Other-upper-body-clothes',
    58: 'Other-lower-body-clothes',
}

MASK_ATTRS = {value: key for key, value in MASK_LABELS.items()}

# Pose
# 0: Right-ankle
# 1: Right-knee
# 2: Right-hip
# 3: Left-hip
# 4: Left-knee
# 5: Left-ankle
# 6: Pelvis
# 7: Thorax
# 8: Upper-neck
# 9: Head-top
# 10: Right-wrist
# 11: Right-elbow
# 12: Right-shoulder
# 13: Left-shoulder
# 14: Left-elbow
# 15: Left-wrist
# 16: Face-bbox-top-left-corner-point
# 17: Face-bbox-bottom-right-corner-point
# 18: Instance-bbox-top-left-corner-point
# 19: Instance-bbox-bottom-right-corner-point


class LV_MHP_v2(Dataset):
    def __init__(self,
                 image_size,
                 transform,
                 mode,
                 shuffling=False,
                 all_attr=0,
                 verbose=False,
                 sampled=100,
                 show_attr='',
                 CREATE_DATASET=False,
                 **kwargs):
        self.image_size = image_size
        self.shuffling = shuffling
        mode = 'train' if mode == 'train' else 'val'
        self.mode = mode
        self.name = self.__class__.__name__
        self.all_attr = all_attr
        self.verbose = verbose
        self.show_attr = show_attr.split(',')
        self.sampled = sampled  # How much data to train (percentage)
        self.data_dir = 'data/{}'.format(self.name)
        ids = os.path.join(self.data_dir, 'list', self.mode + '.txt')
        self.ids = [f.strip() for f in open(ids).readlines()]
        self.colormap = loadmat(
            os.path.join(self.data_dir,
                         'LV-MHP-v2_colormap.mat'))['MHP_colormap']
        self.colorize = Colorize(self.colormap)
        self.data_dir = os.path.join(self.data_dir, self.mode)
        self.attr2idx = {}
        self.idx2attr = {}
        self.mask_label = MASK_LABELS
        self.mask_attr = MASK_ATTRS
        self.attr2filenames = {}
        self.NOTattr2filenames = {}
        self.transform_resize_img = transform.resize_rgb
        self.transform_resize_mask = transform.resize_mask
        self.transform_common = transform.common
        self.transform = transform

        if 'config' in kwargs.keys():
            self.config = kwargs['config']
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace()

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(1)
        if CREATE_DATASET:
            self.create_dataset()
        else:
            self.preprocess()
        self.filenames, self.labels = self.subsample(self.filenames,
                                                     self.labels)
        if self.verbose:
            _str = str(self.num_data)
            print('Finished preprocessing %s: %s (%s)!' %
                  (self.name, mode, _str))
        # self.write_lines()

    def write_lines(self):
        with open('{}/LV_MHP_v2_list_{}.txt'.format(self.data_dir, self.mode),
                  'w') as f:
            for line in self.filenames:
                f.writelines(line + '\n')

    def histogram(self):
        from misc.utils import PRINT
        values = np.sum(self.labels, axis=0)
        dict_ = {}
        # import ipdb; ipdb.set_trace()
        for key, value in zip(self.selected_attrs, values):
            dict_[key] = value
        total = 0
        with open('datasets/{}_histogram_attributes.txt'.format(self.name),
                  'w') as f:
            for key, value in sorted(dict_.items(),
                                     key=lambda kv: (kv[1], kv[0]),
                                     reverse=True):
                total += value
                PRINT(f, '{} {}'.format(key, value))
            PRINT(f, 'TOTAL {}'.format(total))

    def preprocess(self):
        if self.show_attr != '':
            self.selected_attrs = self.show_attr
            self.config.ATTR = self.show_attr
        else:
            self.selected_attrs = [
                'NOT_Cap/hat',
                'Cap/hat',
                'NOT_Jacket/windbreaker/hoodie',
                'Jacket/windbreaker/hoodie',
            ]

        for i, attr in enumerate(self.selected_attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
            self.attr2filenames[attr] = []
            self.NOTattr2filenames[attr] = []

        # lines = self.subsample(lines)
        # if self.shuffling:
        #     random.shuffle(self.lines)
        # random.shuffle(self.lines)
        self.filenames = []
        self.labels = []
        self.segs = []
        self.pose = []
        no_pose = 0
        for i, line in enumerate(tqdm(self.ids, leave=False)):
            filename = os.path.join(self.data_dir, 'images', line + '.jpg')
            pose = os.path.join(self.data_dir, 'pose_annos', line + '.mat')
            segs = sorted(
                glob.glob(
                    os.path.join(self.data_dir, 'parsing_annos', line + '_*')))

            # import ipdb; ipdb.set_trace()
            no_show_attr = True
            # import ipdb; ipdb.set_trace()
            for seg in segs:
                person_id = int(
                    os.path.splitext(os.path.basename(seg))[0].split('_')
                    [-1]) - 1  # starts from 0
                # segmap = self.get_mask_from_file(seg, no_show=True)
                values_sem = self.get_mask_from_file(seg, label=True)
                values_sem = values_sem.unique()
                label = []
                for attr in self.selected_attrs:
                    selected_value = self.get_value(values_sem, attr)
                    if selected_value >= 1:
                        label.append(selected_value)
                        self.attr2filenames[attr].append(line[0])
                        no_show_attr = False
                    else:
                        label.append(0)
                        self.NOTattr2filenames[attr].append(line[0])

                if self.show_attr and no_show_attr:
                    continue
                try:
                    pose_id = loadmat(pose)['person_%d' % person_id]
                except BaseException:
                    # import ipdb; ipdb.set_trace()
                    no_pose += 1
                    continue
                # import ipdb; ipdb.set_trace()
                self.filenames.append(filename)
                self.labels.append(label)
                self.segs.append(seg)
                self.pose.append(pose_id)
        print("No pose found:", no_pose)
        if not self.show_attr:
            self.histogram()
        self.num_data = len(self.filenames)

    def create_dataset(self):
        no_pose = 0
        new_images = os.path.join(self.data_dir, 'new_images')
        os.makedirs(new_images, exist_ok=True)
        new_segs = os.path.join(self.data_dir, 'new_segs')
        os.makedirs(new_segs, exist_ok=True)
        new_pose = os.path.join(self.data_dir, 'new_pose')
        os.makedirs(new_pose, exist_ok=True)
        new_labels = os.path.join(self.data_dir, 'new_labels')
        os.makedirs(new_labels, exist_ok=True)
        self.selected_attrs = self.mask_attr.keys()
        for i, line in enumerate(tqdm(self.ids, leave=False)):
            filename = os.path.join(self.data_dir, 'images', line + '.jpg')
            pose = os.path.join(self.data_dir, 'pose_annos', line + '.mat')
            segs = sorted(
                glob.glob(
                    os.path.join(self.data_dir, 'parsing_annos', line + '_*')))

            # import ipdb; ipdb.set_trace()
            no_show_attr = True
            # import ipdb; ipdb.set_trace()
            for seg in segs:
                person_id = int(
                    os.path.splitext(os.path.basename(seg))[0].split('_')
                    [-1]) - 1  # starts from 0
                # segmap = self.get_mask_from_file(seg, no_show=True)
                segmap = self.get_mask_from_file(seg, label=True)[0]
                try:
                    pose_id = loadmat(pose)['person_%d' % person_id]
                except BaseException:
                    # import ipdb; ipdb.set_trace()
                    no_pose += 1
                    continue
                import ipdb
                ipdb.set_trace()
                labels = [self.mask_label[i] for i in segmap.unique()]
                nonzero = torch.nonzero(segmap)
                bbox = nonzero.min(0)[0].tolist()
                bbox.extend(nonzero.max(0)[0].tolist())  # x1, y1, x2, y2
                self.filenames.append(filename)
                self.labels.append(label)
                self.segs.append(seg)
                self.pose.append(pose_id)
        print("No pose found:", no_pose)
        if not self.show_attr:
            self.histogram()
        self.num_data = len(self.filenames)

    def get_value(self, values, attr):
        NOT = False
        if 'NOT_' in attr:
            NOT = True
            attr = attr.replace('NOT_', '')

        index = list(self.mask_label.values()).index(attr)
        value = int(index in values)

        if NOT:
            value = 1 - value
        if value == -1:
            import ipdb
            ipdb.set_trace()
        assert value != -1
        return value

    def get_data(self):
        return self.filenames, self.labels

    def get_mask_from_file(self, maskname, no_show=False, label=False):
        mask = Image.open(maskname).convert('RGB')
        mask = self.transform_resize_mask(mask)
        mask = self.transform_common(mask)[0] * 255.  # 0, 255
        if self.show_attr and not no_show:
            labels_real = self.get_partial_mask(mask).unsqueeze(0)
        elif label:
            labels_real = mask
        else:
            labels_real = scatterMask(mask, num_channels=len(self.mask_label))
        # labels_real: C x size x size
        return labels_real  # 59 attrs

    def get_partial_mask(self, mask):
        new_mask = torch.zeros_like(mask)
        for attr in self.selected_attrs:
            label = self.mask_attr[attr]
            new_mask[mask == label] = label
        return new_mask

    def __getitem__(self, index):
        filename = self.filenames[index]
        seg = self.segs[index]
        label = self.labels[index]
        pose = self.pose[index]

        image = Image.open(filename)
        # import ipdb; ipdb.set_trace()
        if self.show_attr:
            image = image.convert('RGBA')
            img2 = image.copy()
            zero_seg = np.zeros((image.size[::-1])).astype(np.uint8)
            org_seg = self.get_mask_from_file(seg, no_show=True)
            for idx, attr in enumerate(self.attributes):
                _label = label[idx]
                if _label == 1:
                    zero_seg += org_seg[self.mask_attr[attr]]
            import ipdb
            ipdb.set_trace()
            zero_seg = self.colorize(zero_seg) / 255.

        else:
            image = image.convert('RGB')
            # import ipdb; ipdb.set_trace()
            seg = np.zeros((1 + len(self.attr2idx.keys()),
                            *image.size[::-1])).astype(np.uint8)
            for label, segs in self.segs[index].items():
                img_temp = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(img_temp)
                draw.polygon(segs, outline=1, fill=1)
                img_temp = np.array(img_temp)
                seg[label + 1][img_temp == 1] = 1
            seg[0][seg.sum(0) == 0] = 1  # background
            # seg = seg[None,:,:,:].repeat(3,0).transpose(1,0,2,3)
            # to match the transform variable
            # import ipdb; ipdb.set_trace()

        bbox = self.bbox[index]  # x1,y1,x2,y2
        margin = (0.075, 0.075)
        width_ = image.size[0] * margin[0]
        height_ = image.size[1] * margin[1]
        bbox[0] = max(0, bbox[0] - width_)
        bbox[1] = max(0, bbox[1] - height_)
        bbox[2] = min(image.size[0], bbox[2] + width_)
        bbox[3] = min(image.size[1], bbox[3] + height_)
        image = image.crop(bbox)
        # import ipdb; ipdb.set_trace()
        keyp = keyp.crop(bbox)
        keyp = self.transform_resize_img(keyp)
        keyp = self.transform_common(keyp)[0].unsqueeze(0)
        seg = [Image.fromarray(i).crop(bbox).convert('RGB') for i in seg]
        seg = [self.transform_resize_mask(i) for i in seg]
        seg = [self.transform_common(i)[0] for i in seg]
        seg = torch.stack(seg, dim=0) * 255
        # seg = scatterMask(seg, num_channels=1+len(self.attr2idx.keys()))
        image = self.transform_resize_img(image)
        image = self.transform_common(image)
        image = self.transform.norm(image)
        # import ipdb; ipdb.set_trace()
        if self.show_attr:
            alpha = 0.4
            image = (alpha * image) + (1 - alpha) * keyp
        label = torch.FloatTensor(self.labels[index])

        if self.config.TRAIN_MASK:  # or self.config.ONLY_GEN:
            _seg = image
            image = seg
            seg = _seg

        return image, label, seg, keyp

    def __len__(self):
        return self.num_data

    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.filenames)
        random.seed(seed)
        random.shuffle(self.labels)


def show_me(args):
    from data_loader import get_transformations
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from misc.utils import denorm
    import numpy as np
    import matplotlib.pyplot as plt
    attrs = args.attr  # .split(',')
    mode = 'train'
    transform = get_transformations(mode='test', image_size=256)
    data = LV_MHP_v2(256,
                     transform,
                     mode,
                     show_attr=attrs,
                     CREATE_DATASET=args.CREATE_DATASET,
                     verbose=True)
    data_loader = DataLoader(dataset=data,
                             batch_size=64,
                             shuffle=False,
                             num_workers=4)
    for i, (data, label, *_) in enumerate(data_loader):
        data = denorm(data)
        data = make_grid(data).numpy()
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(data, (1, 2, 0)), interpolation='nearest')
        plt.show(block=True)


class Colorize(object):
    def __init__(self, cmap):
        # self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(cmap)

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


if __name__ == '__main__':
    # ipython datasets/DeepFashion2.py -- --attr=vest
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='all')
    parser.add_argument('--CREATE_DATASET', action='store_true', default=False)
    args = parser.parse_args()
    # train_inception()
    show_me(args)
