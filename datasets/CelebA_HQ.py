from munch import Munch
import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
from datasets import MASK_LABELS, MASK_ATTRS, SEMANTIC_ATTRS, SEMANTIC_ATTRS_NO_EYEGLASSES, SEMANTIC_ATTRS_MISSING, SEMANTIC_ATTRS_MISSING_WITH_HAIR, SEMANTIC_ATTRS_ONLY
from misc.mask_utils import scatterMask
from misc.utils import denorm
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)
# ==================================================================#
# == CelebA
# ==================================================================#

# I am deeply sorry for that poor soul that have to go through this mess.
# Essentially it will read Celeb_HQ, and create both images and label dataloader (only the labels that come from the argument
# eg., --GENDER and --BANGS will create only two binary labels, and so forth.)


class CelebA_HQ(Dataset):
    def __init__(self,
                 image_size,
                 transform,
                 mode,
                 shuffling=False,
                 verbose=False,
                 sampled=100,
                 show_attr='',
                 train_mask=False,
                 **kwargs):
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        self.name = self.__class__.__name__
        self.verbose = verbose
        self.show_attr = show_attr
        self.sampled = sampled  # How much data to train (percentage)
        self.root = 'data/{}'.format(self.name)
        self.data_dir = '{}/celeba-{}'.format(self.root, self.image_size)

        if 'config' in kwargs.keys():
            self.config = kwargs['config']
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            self.config.MASK = False
            self.config.TRAIN_MASK = False
            self.config.STYLE_SEMANTICS = False
            self.config.STYLE_SEMANTICS_ATTR = False
            self.config.STYLE_SEMANTICS_ALL = False
            self.config.USE_MASK_TRAIN_SEMANTICS_MISSING = False
            self.config.USE_MASK_TRAIN_SEMANTICS_MISSING_WITH_HAIR = False
            self.config.USE_MASK_TRAIN_SEMANTICS_ONLY = False
            self.config.USE_MASK_TRAIN_SEMANTICS_NO_EYEGLASSES = False
            self.config.HAIR = False

        self.mask_label = MASK_LABELS
        self.mask_attr = MASK_ATTRS
        if self.config.USE_MASK_TRAIN_SEMANTICS_ONLY:
            self.semantic_attr = SEMANTIC_ATTRS_ONLY
        elif self.config.USE_MASK_TRAIN_SEMANTICS_MISSING_WITH_HAIR:
            self.semantic_attr = SEMANTIC_ATTRS_MISSING_WITH_HAIR
        elif self.config.USE_MASK_TRAIN_SEMANTICS_NO_EYEGLASSES:
            self.semantic_attr = SEMANTIC_ATTRS_NO_EYEGLASSES
            self.config.STYLE_SEMANTICS_ATTR = True
        elif self.config.USE_MASK_TRAIN_SEMANTICS_MISSING:
            self.semantic_attr = SEMANTIC_ATTRS_MISSING
            self.config.STYLE_SEMANTICS_ATTR = True
        else:
            self.semantic_attr = SEMANTIC_ATTRS

        self.MASK = self.config.MASK
        self.STYLE_SEMANTICS = self.config.STYLE_SEMANTICS
        self.STYLE_SEMANTICS_ATTR = self.config.STYLE_SEMANTICS_ATTR
        self.STYLE_SEMANTICS_ALL = self.config.STYLE_SEMANTICS_ALL
        self.lines = [
            line.strip().split() for line in open(
                os.path.abspath('data/{}/list_attr_celeba.txt'.format(
                    self.name))).readlines()
        ]
        self.splits = self.get_splits()
        self.MODES = {'train': 0, 'val': 1, 'test': 1, 'demo': 1}
        self.mode_allowed = [self.MODES[self.mode]]
        # self.mode_allowed = [0, 1, 2]
        self.all_attr2idx = {}
        self.all_idx2attr = {}
        self.attr2idx = {}
        self.idx2attr = {}
        self.attr2filenames = {}
        self.NOTattr2filenames = {}

        self.mask_dir = 'data/{}/celeba-mask-{}'.format(
            self.name, self.image_size)
        self.transform_resize_img = transform.resize_rgb
        self.transform_resize_mask = transform.resize_mask
        self.transform_common = transform.common
        self.transform = transform

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(333)
        self.preprocess()
        self.filenames, self.labels = self.subsample(self.filenames,
                                                     self.labels)

        self.filenames_ref = list(self.filenames)
        random.seed(1)
        random.shuffle(self.filenames_ref)
        self.labels_ref = list(self.labels)
        random.seed(1)
        random.shuffle(self.labels_ref)
        if self.verbose:
            _str = str(self.num_data)
            print('Finished preprocessing %s: %s (%s)!' %
                  (self.name, mode, _str))

    # def get_splits(self):
    #     splits = {
    #         line.split(',')[0].split('.')[0]: int(line.strip().split(',')[1])
    #         for line in open(
    #             os.path.abspath('data/{}/train_val_test.txt'.format(
    #                 self.name))).readlines()[1:]
    #     }
    #     return splits

    def get_splits(self):
        # import ipdb; ipdb.set_trace()
        dirname = 'data/celeba_hq'
        files = os.listdir(dirname + '/train/male')
        files += os.listdir(dirname + '/train/female')
        splits = {f: 0 for f in files}

        files = os.listdir(dirname + '/val/male')
        files += os.listdir(dirname + '/val/female')
        splits.update({f: 1 for f in files})
        # splits.update({f.split('.')[0]:2 for f in files})
        return splits

    def histogram(self):
        from misc.utils import PRINT
        values = np.array([int(i) for i in self.lines[2][1:]]) * 0
        n_images = 0
        for line in self.lines[2:]:
            if not self.image_exist(line[0]):
                continue
            values += np.array([int(i) for i in line[1:]]).clip(min=0)
            n_images += 1
        dict_ = {}
        for key, value in zip(self.lines[1], values):
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

    def subsample(self, filenames, labels):
        if self.sampled == 100 or self.mode not in ['train', 'val']:
            return filenames, labels
        else:
            self.num_data = int(self.num_data * self.sampled / 100)
            new_filenames = filenames[:self.num_data]
            new_labels = labels[:self.num_data]
            return new_filenames, new_labels

    def get_value(self, values, attr, semantic_attr=False):
        def str2int(v):
            _v = int(values[self.all_attr2idx[v]])
            return int(max(_v, 0))  # [0, 1]

        if semantic_attr:
            index = list(self.mask_label.keys()).index(attr.lower())
            value = int(index in values)
            return value

        value = -1
        NOT = False
        if 'NOT_' in attr:
            NOT = True
            attr = attr.replace('NOT_', '')

        if attr in self.all_attr2idx.keys():
            value = str2int(attr)
        else:
            if attr in ['General_Style']:
                value = 1

            elif attr in ['Female']:
                value = 1 - str2int('Male')

            elif attr in ['Few_Hair']:
                value = str2int('Bald')

            elif attr in ['Much_Hair']:
                value = 1 - str2int('Bald')

            elif attr in ['Short_Hair']:
                value = str2int('Male')

            elif attr in ['Long_Hair']:
                value = 1 - str2int('Male')

            elif attr in ['Aged']:
                value = 1 - str2int('Young')

            elif attr in ['Happiness']:
                value = str2int('Smiling')

            elif attr in ['Makeup']:
                value = str2int('Heavy_Makeup')

            elif attr in ['Earrings']:
                value = str2int('Wearing_Earrings')

            elif attr in ['Hat']:
                value = str2int('Wearing_Hat')

            elif attr == 'Color_Eyeglasses':
                value = str2int('Eyeglasses')

            elif attr in ['Color_Hair', 'Color_Bangs']:
                _bangs = str2int('Bangs') == 1
                if (attr == 'Color_Bangs' and _bangs) or attr == 'Color_Hair':
                    _attrs = [
                        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
                    ]
                    for _attr in _attrs:
                        if str2int(_attr) == 1:
                            value = 1
                            break
            elif attr == 'Facial_Hair':
                value = 1 - str2int('No_Beard')
                if value == 0:
                    _attrs = ['5_o_Clock_Shadow', 'Mustache', 'Goatee']
                    for _attr in _attrs:
                        if str2int(_attr) == 1:
                            value = 1
                            break
            elif attr == 'Hair':
                value = 1 - str2int('Bald')

        if NOT:
            value = 1 - value
        if value == -1:
            import ipdb
            ipdb.set_trace()
        assert value != -1
        return value

    def preprocess(self):
        attrs = self.lines[1]
        if self.verbose:
            self.histogram()

        for i, attr in enumerate(attrs):
            self.all_attr2idx[attr] = i
            self.all_idx2attr[i] = attr

        if self.config.STYLE_SEMANTICS_ATTR:
            self.mask_attr.update(self.semantic_attr)
            self.selected_attrs = []

        if self.show_attr != '':
            self.selected_attrs = self.show_attr
            self.config.ATTR = self.show_attr

        elif self.STYLE_SEMANTICS_ALL:
            self.selected_attrs = list(self.mask_attr.keys())

        # else:
        #     self.selected_attrs = [
        #         '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
        #         'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        #         'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
        #         'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        #         'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        #         'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
        #         'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
        #         'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
        #         'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        #         'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        #         'Wearing_Necktie', 'Young'
        #     ]  # Total: 40

        # elif self.config.STYLE_SEMANTICS:
        # elif not self.config.STYLE_SEMANTICS_ATTR:
        else:
            if self.config.ATTR == 'gender':
                self.selected_attrs = ['Male', 'Female']
            elif self.config.ATTR == 'eyeglasses':
                # self.selected_attrs = ['Eyeglasses', 'NOT_Eyeglasses']
                self.selected_attrs = ['NOT_Eyeglasses', 'Eyeglasses']
            elif self.config.ATTR == 'bangs':
                # self.selected_attrs = ['Bangs', 'NOT_Bangs']
                self.selected_attrs = ['NOT_Bangs', 'Bangs']
            elif self.config.ATTR == 'smile':
                # self.selected_attrs = ['Smiling', 'NOT_Smiling']
                self.selected_attrs = ['NOT_Smiling', 'Smiling']
            elif self.config.ATTR == 'hair':
                # self.selected_attrs = ['Hair', 'NOT_Hair']
                self.selected_attrs = ['NOT_Hair', 'Hair']
            elif self.config.ATTR == 'single':
                self.selected_attrs = ['General_Style']
            else:
                self.selected_attrs = [
                    # 'Eyeglasses',
                    # 'NOT_Eyeglasses',
                    # 'Smiling',
                    # 'NOT_Smiling',
                    # 'Bangs',
                    # 'NOT_Bangs',
                    # 'Male',
                    # 'Female',
                ]
                if self.config.NOT_GENDER:
                    self.selected_attrs.pop(-1)
                    self.selected_attrs.pop(-1)
                if self.config.GENDER:
                    self.selected_attrs.append('Male')
                    self.selected_attrs.append('Female')
                if self.config.EYEGLASSES:
                    self.selected_attrs.append('NOT_Eyeglasses')
                    self.selected_attrs.append('Eyeglasses')
                if self.config.SMILE:
                    self.selected_attrs.append('NOT_Smiling')
                    self.selected_attrs.append('Smiling')
                if self.config.HAIR:
                    self.selected_attrs.append('Few_Hair')
                    self.selected_attrs.append('Much_Hair')
                    # self.selected_attrs.append('Short_Hair')
                    # self.selected_attrs.append('Long_Hair')
                if self.config.SHORT_HAIR:
                    self.selected_attrs.append('Short_Hair')
                    self.selected_attrs.append('Long_Hair')
                if self.config.BANGS:
                    self.selected_attrs.append('NOT_Bangs')
                    self.selected_attrs.append('Bangs')
                if self.config.EARRINGS:
                    self.selected_attrs.append('NOT_Earrings')
                    self.selected_attrs.append('Earrings')
                if self.config.HAT:
                    self.selected_attrs.append('NOT_Hat')
                    self.selected_attrs.append('Hat')
                if len(self.selected_attrs) == 0:
                    raise AttributeError("Please select attributes")
        if self.config.STYLE_SEMANTICS_ATTR and not self.STYLE_SEMANTICS_ALL:
            self.selected_attrs += self.semantic_attr.keys()

        if self.config.ATTR and not self.show_attr:
            self.parent_attrs = {self.config.ATTR: self.selected_attrs}
        elif self.MASK and self.STYLE_SEMANTICS_ALL:
            self.parent_attrs = {i: [i] for i in self.selected_attrs}
        elif self.config.USE_MASK_TRAIN_SEMANTICS_ONLY:
            self.parent_attrs = self.semantic_attr
        # elif self.STYLE_SEMANTICS:
        else:
            self.parent_attrs = {
                # 'Eyeglasses': ['Eyeglasses', 'NOT_Eyeglasses'],
                # 'Eyeglasses': ['Eyeglasses'],
                # 'Bangs': ['Bangs', 'NOT_Bangs'],
                # 'Smile': ['Smiling', 'NOT_Smiling'],
                # 'Gender': ['Male', 'Female'],
            }
            if self.config.NOT_GENDER:
                # hypothesis: maybe mixing gender with all_attr is too
                # difficult
                self.parent_attrs.pop('Gender')
            if self.config.GENDER:
                self.parent_attrs.update({'Gender': ['Male', 'Female']})
            if self.config.EYEGLASSES:
                self.parent_attrs.update(
                    {'Eyeglasses': ['Eyeglasses', 'NOT_Eyeglasses']})
            if self.config.SMILE:
                self.parent_attrs.update({'Smile': ['Smiling', 'NOT_Smiling']})
            if self.config.HAIR:
                # self.parent_attrs.update({'Hair': ['Hair', 'NOT_Hair']})
                self.parent_attrs.update({'Hair': ['Much_Hair', 'Few_Hair']})
                # self.parent_attrs.update({'Hair': ['Few_Hair', 'Short_Hair', 'Long_Hair']})
            if self.config.SHORT_HAIR:
                self.parent_attrs.update(
                    {'Short_Hair': ['Short_Hair', 'Long_Hair']})
            if self.config.BANGS:
                self.parent_attrs.update({'Bangs': ['Bangs', 'NOT_Bangs']})
                # self.parent_attrs.update({'Bangs': ['Bangs']})
            if self.config.EARRINGS:
                self.parent_attrs.update(
                    {'Earrings': ['Earrings', 'NOT_Earrings']})
                # self.parent_attrs.update({'Earrings': ['Earrings']})
            if self.config.HAT:
                self.parent_attrs.update({'Hat': ['Hat', 'NOT_Hat']})
                # self.parent_attrs.update({'Hat': ['Hat']})
            if self.config.STYLE_SEMANTICS_ATTR:
                self.parent_attrs.update(self.semantic_attr)
        # import ipdb; ipdb.set_trace()
        self.children_attrs = self.selected_attrs
        # make sure children and parents has the same order
        for i, attr in enumerate(self.selected_attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
            self.attr2filenames[attr] = []
            self.NOTattr2filenames[attr] = []
        self.filenames = []
        self.labels = []
        # self.labels_attr = []
        self.mask = []
        lines = self.lines[2:]
        # lines = self.subsample(lines)
        if self.shuffling:
            random.shuffle(lines)
        for i, line in enumerate(lines):
            if not self.image_exist(line[0]):
                continue

            # Values from masks
            if self.STYLE_SEMANTICS_ATTR or self.config.USE_MASK_TRAIN_SEMANTICS_NO_EYEGLASSES:
                img_file = os.path.join(self.data_dir, line[0])
                values_sem = self.get_mask_from_file(img_file, label=True)
                values_sem = values_sem.unique()
                # label = []
                # for attr in self.mask_label.keys():
                #     selected_value = self.get_value(values_sem,
                #                                     attr,
                #                                     semantic_attr=True)
                #     label.append(selected_value)
                # self.labels_attr.append(label)

            values = line[1:]

            label = []

            no_show_attr = True
            # import ipdb; ipdb.set_trace()
            for attr in self.selected_attrs:
                if attr in self.semantic_attr and self.STYLE_SEMANTICS_ATTR:
                    selected_value = self.get_value(values_sem,
                                                    attr,
                                                    semantic_attr=True)
                else:
                    selected_value = self.get_value(values, attr)
                if selected_value >= 1:
                    label.append(selected_value)
                    self.attr2filenames[attr].append(line[0])
                    no_show_attr = False
                else:
                    label.append(0)
                    self.NOTattr2filenames[attr].append(line[0])

            if self.show_attr and no_show_attr:
                continue

            if 1 not in label and self.mode == 'train':
                continue

            self.filenames.append(line[0])
            self.mask.append(line[0])
            self.labels.append(label)

        self.num_data = len(self.filenames)

    def image_exist(self, name):
        # import ipdb; ipdb.set_trace()
        if name not in self.splits.keys():
            return False
        if self.splits[name] not in self.mode_allowed:
            return False
        filename = os.path.abspath(os.path.join(self.data_dir, name))
        if not os.path.isfile(filename):
            return False
        return True

    def get_data(self):
        return self.filenames, self.labels

    def get_mask_from_file(self, filename, label=False):
        maskname = filename.replace(self.data_dir, self.mask_dir)
        maskname = maskname.replace('jpg', 'png')
        mask = Image.open(maskname).convert('RGB')
        # import ipdb; ipdb.set_trace()
        mask = self.transform_resize_mask(mask)
        mask = self.transform_common(mask)[0] * 255.  # 0, 255
        if self.show_attr:
            labels_real = self.get_partial_mask(mask).unsqueeze(0)
        elif label:
            labels_real = mask
        else:
            labels_real = scatterMask(mask, num_channels=len(self.mask_label))
        # labels_real: C x size x size
        return labels_real  # 19 attrs

    def get_partial_mask(self, mask):
        new_mask = torch.zeros_like(mask)
        for attr in self.selected_attrs:
            for number_mask in self.mask_attr[attr]:
                label = self.mask_label[number_mask]
                new_mask[mask == label] = label
        return new_mask

    def __getitem__(self, index):
        filename = os.path.join(self.data_dir, self.filenames[index])
        image = self.file2img(filename)
        label = torch.FloatTensor(self.labels[index])
        mask = self.get_mask_from_file(filename)
        # if self.config.USE_MASK_TRAIN_ATTR:
        #     _label_attr = torch.FloatTensor(self.labels_attr[index])
        #     filename = _label_attr
        if self.config.TRAIN_MASK:  # or self.config.ONLY_GEN:
            _mask = image
            image = mask
            mask = _mask
        # import ipdb; ipdb.set_trace()

        filename_ref = os.path.join(self.data_dir, self.filenames_ref[index])
        image_ref = self.file2img(filename_ref)
        label_ref = torch.FloatTensor(self.labels_ref[index])
        mask_ref = self.get_mask_from_file(filename_ref)
        if self.config.TRAIN_MASK:  # or self.config.ONLY_GEN:
            _mask_ref = image_ref
            image_ref = mask_ref
            mask_ref = _mask_ref

        data = Munch(image=image,
                     label=label,
                     mask=mask,
                     image_ref=image_ref,
                     label_ref=label_ref,
                     mask_ref=mask_ref,
                     filename=filename,
                     filename_ref=filename_ref)
        # return image, label, mask, (image_ref, label_ref, mask_ref)
        return data

    def file2img(self, filename):
        image = Image.open(filename).convert('RGB')
        # import ipdb; ipdb.set_trace()
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
        random.seed(seed)
        random.shuffle(self.filenames_ref)
        random.seed(seed)
        random.shuffle(self.labels_ref)
        # random.seed(seed)
        # random.shuffle(self.labels_attr)


def create_dataset(dataset, label, filename, mask=False):
    root = dataset.root
    if mask:
        filename = filename.replace(dataset.data_dir, dataset.mask_dir)
        filename = filename.replace('jpg', 'png')
        attr_folder = os.path.join(root, 'ATTR_MASK')
    else:
        attr_folder = os.path.join(root, 'ATTR')
    for attr in dataset.selected_attrs:
        os.makedirs(os.path.join(attr_folder, dataset.mode, attr),
                    exist_ok=True)
    # import ipdb; ipdb.set_trace()
    for attr, idx in zip(dataset.selected_attrs, label):
        if idx == 1:
            target_dir = os.path.join(attr_folder, dataset.mode, attr)
            ln_file = os.path.join(target_dir, os.path.basename(filename))
            if not os.path.isfile(ln_file):
                os.symlink(os.path.abspath(filename), ln_file)
            # import ipdb; ipdb.set_trace()


def show_me(args):
    from data_loader import get_transformations
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import numpy as np
    import matplotlib.pyplot as plt
    from misc.mask_utils import label2mask
    from tqdm import tqdm

    attrs = args.attr.split(',')
    assert len(attrs) >= 1, "Please provide at least one attribute"
    mode = args.mode
    transform = get_transformations(mode='test', image_size=256)
    dataset = CelebA_HQ(256,
                        transform,
                        mode,
                        show_attr=attrs,
                        COLORS=True,
                        verbose=True)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=24)

    if args.CREATE_DATASET:
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
    else:
        loop = enumerate(data_loader)
    for i, (data, label, mask, filename) in loop:
        if args.CREATE_DATASET:
            for _label, _filename in zip(label, filename):
                create_dataset(dataset, _label, _filename)
                create_dataset(dataset, _label, _filename, mask=True)
            continue
        data = denorm(data)
        # up_size = 1024
        # data = resize(data, up_size)
        if args.MASK:
            alpha = 0.7
            mask = label2mask(mask)
            # mask = resize(mask, up_size, 'nearest')
            data = ((1 - alpha) * data + alpha * mask)

        data = make_grid(data).numpy()
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(data, (1, 2, 0)), interpolation='nearest')
        plt.show(block=True)


def resize(tensor, up_size, mode='bilinear'):
    import torch.nn.functional as F
    return F.interpolate(tensor, size=(up_size, up_size), mode=mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val'])
    parser.add_argument('--MASK', action='store_true', default=False)
    parser.add_argument('--CREATE_DATASET', action='store_true', default=False)
    args = parser.parse_args()
    if args.CREATE_DATASET:
        args.attr = ','.join(MASK_ATTRS.keys())
    show_me(args)
