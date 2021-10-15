import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
from torchvision import transforms
from datasets import MASK_LABELS, MASK_ATTRS, SEMANTIC_ATTRS
from misc.mask_utils import scatterMask

# from misc.utils import timeit
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

# ==================================================================#
# == FFHQ
# ==================================================================#


class FFHQ(Dataset):
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
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        self.name = self.__class__.__name__
        self.verbose = verbose
        self.show_attr = show_attr
        self.sampled = sampled  # How much data to train (percentage)
        self.data_dir = 'data/{}/FFHQ-{}'.format(self.name, self.image_size)
        self.mask_label = MASK_LABELS
        self.mask_attr = MASK_ATTRS
        self.semantic_attr = SEMANTIC_ATTRS

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

        self.MASK = self.config.MASK
        self.STYLE_SEMANTICS = self.config.STYLE_SEMANTICS
        self.STYLE_SEMANTICS_ATTR = self.config.STYLE_SEMANTICS_ATTR
        self.STYLE_SEMANTICS_ALL = self.config.STYLE_SEMANTICS_ALL
        self.mask_dir = 'data/{}/FFHQ-mask-{}'.format(self.name,
                                                      self.image_size)
        # self.transform_mask = transforms.Compose([transforms.ToTensor()])
        self.transform_resize_img = transform.resize_rgb
        self.transform_resize_mask = transform.resize_mask
        self.transform_common = transform.common
        self.transform = transform
        self.lines = [
            line.strip().split(',') for line in open(
                os.path.abspath('data/{}/list_attr_ffhq.txt'.format(
                    self.name))).readlines()
        ]

        self.modes = [60000, 65000, 70000]
        files = [lines[0] for lines in self.lines[1:]]
        self.splits = {}
        self.MODES = {'train': 0, 'val': 1, 'test': 2}
        for idx, line in enumerate(files):
            if idx < self.modes[0]:
                self.splits[line] = self.MODES['train']
            elif idx > self.modes[0] and idx < self.modes[1]:
                self.splits[line] = self.MODES['val']
            else:
                self.splits[line] = self.MODES['test']

        self.mode_allowed = [self.MODES[self.mode]]
        self.all_attr2idx = {}
        self.all_idx2attr = {}
        self.attr2idx = {}
        self.idx2attr = {}
        self.attr2filenames = {}
        self.NOTattr2filenames = {}

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(1)
        self.preprocess()
        self.filenames, self.labels = self.subsample(self.filenames,
                                                     self.labels)
        if self.verbose:
            _str = str(self.num_data)
            print('Finished preprocessing %s: %s (%s)!' %
                  (self.name, mode, _str))

    def histogram(self):
        from misc.utils import PRINT
        values = np.array([int(i) for i in self.lines[1][1:]]) * 0
        n_images = 0
        for line in self.lines[1:]:
            if not self.image_exist(line[0]):
                continue
            values += np.array([int(i) for i in line[1:]]).clip(min=0)
            n_images += 1
        dict_ = {}
        for key, value in zip(self.lines[0][1:], values):
            dict_[key] = value
        total = 0
        with open('datasets/{}_histogram_attributes.txt'.format(self.name),
                  'w') as f:
            for key, value in sorted(dict_.items(),
                                     key=lambda kv: (kv[1], kv[0]),
                                     reverse=True):
                total += value
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
        attrs = self.lines[0][1:]
        assert len(attrs) == 40
        if self.verbose and self.mode == 'train':
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
            if self.config.ATTR == 'beings':
                self.selected_attrs = ['Male', 'Female']
            elif self.config.ATTR == 'eyeglasses':
                self.selected_attrs = ['Eyeglasses', 'NOT_Eyeglasses']
            elif self.config.ATTR == 'bangs':
                self.selected_attrs = ['Bangs', 'NOT_Bangs']
            elif self.config.ATTR == 'smile':
                self.selected_attrs = ['Smiling', 'NOT_Smiling']
            elif self.config.ATTR == 'hair':
                self.selected_attrs = ['Hair', 'NOT_Hair']
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
                if self.config.NOT_BEINGS:
                    self.selected_attrs.pop(-1)
                    self.selected_attrs.pop(-1)
                if self.config.BEINGS:
                    self.selected_attrs.append('Male')
                    self.selected_attrs.append('Female')
                if self.config.EYEGLASSES:
                    self.selected_attrs.append('Eyeglasses')
                    self.selected_attrs.append('NOT_Eyeglasses')
                if self.config.SMILE:
                    self.selected_attrs.append('Smiling')
                    self.selected_attrs.append('NOT_Smiling')
                if self.config.HAIR:
                    self.selected_attrs.append('Much_Hair')
                    self.selected_attrs.append('Few_Hair')
                    # self.selected_attrs.append('Short_Hair')
                    # self.selected_attrs.append('Long_Hair')
                if self.config.SHORT_HAIR:
                    self.selected_attrs.append('Short_Hair')
                    self.selected_attrs.append('Long_Hair')
                if self.config.BANGS:
                    self.selected_attrs.append('Bangs')
                    self.selected_attrs.append('NOT_Bangs')
                if self.config.EARRINGS:
                    self.selected_attrs.append('Earrings')
                    self.selected_attrs.append('NOT_Earrings')
                if self.config.HAT:
                    self.selected_attrs.append('Hat')
                    self.selected_attrs.append('NOT_Hat')
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
                # 'Beings': ['Male', 'Female'],
            }
            if self.config.NOT_BEINGS:
                # hypothesis: maybe mixing beings with all_attr is too
                # difficult
                self.parent_attrs.pop('Beings')
            if self.config.BEINGS:
                self.parent_attrs.update({'Beings': ['Male', 'Female']})
            if self.config.EYEGLASSES:
                self.parent_attrs.update(
                    {'Eyeglasses': ['Eyeglasses', 'NOT_Eyeglasses']})
            if self.config.SMILE:
                self.parent_attrs.update({'Smile': ['Smiling', 'NOT_Smiling']})
            if self.config.HAIR:
                # self.parent_attrs.update({'Hair': ['Hair', 'NOT_Hair']})
                self.parent_attrs.update({'Hair': ['Few_Hair']})
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
        self.children_attrs = self.selected_attrs
        # make sure children and parents has the same order

        for i, attr in enumerate(self.selected_attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
            self.attr2filenames[attr] = []
            self.NOTattr2filenames[attr] = []
        self.filenames = []
        self.labels = []
        self.mask = []
        lines = self.lines[1:]
        if self.shuffling:
            random.shuffle(lines)
        for i, line in enumerate(lines):
            if not self.image_exist(line[0]):
                continue
            if self.STYLE_SEMANTICS_ATTR:
                values_sem = self.get_mask_from_file(line[0], label=True)
                values_sem = values_sem.unique()
            values = line[1:]

            label = []

            no_show_attr = False
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
                else:
                    label.append(0)
                    self.NOTattr2filenames[attr].append(line[0])
                    no_show_attr = True

            if self.show_attr and no_show_attr:
                continue

            if 1 not in label and self.mode == 'train':
                continue

            self.filenames.append(line[0])
            self.mask.append(line[0])
            self.labels.append(label)

        self.num_data = len(self.filenames)

    def image_exist(self, name):
        if self.splits[name] not in self.mode_allowed:
            return False
        filename = os.path.abspath(os.path.join(self.data_dir, name))
        if not os.path.isfile(filename):
            return False
        return True

    def get_data(self):
        return self.filenames, self.labels

    # def get_mask_from_file(self, filename, label=False):
    #     maskname = filename.replace(self.data_dir, self.mask_dir)
    #     mask = Image.open(maskname).convert('RGB')
    #     mask = self.transform_mask(mask)[0] * 255.  # 0, 255
    #     if self.show_attr:
    #         labels_real = self.get_partial_mask(mask).unsqueeze(0)
    #     elif label:
    #         labels_real = mask
    #     else:
    #         labels_real = scatterMask(mask, num_channels=len(self.mask_label))
    #     # labels_real: C x size x size
    #     return labels_real  # 19 attrs

    def get_mask_from_file(self, filename, label=False):
        maskname = filename.replace(self.data_dir, self.mask_dir)
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

    # def __getitem__(self, index):
    #     filename = os.path.join(self.data_dir, self.filenames[index])
    #     image = self.file2img(filename)
    #     label = torch.FloatTensor(self.labels[index])
    #     mask = self.get_mask_from_file(filename)
    #     if self.config.TRAIN_MASK:
    #         _mask = image
    #         image = mask
    #         mask = _mask
    #     return image, label, mask, filename

    # def __getitem__(self, index):
    #     try:
    #         filename = os.path.join(self.data_dir, self.filenames[index])
    #         image = self.file2img(filename)
    #         label = torch.FloatTensor(self.labels[index])
    #         mask = self.get_mask_from_file(filename)
    #     except OSError:
    #         # image file truncated (?)
    #         filename_c = os.path.join(self.data_dir, self.filenames[index])
    #         print('Image file corrupted: ' + filename_c)
    #         filename = os.path.join(self.data_dir, self.filenames[0])
    #         image = self.file2img(filename)
    #         label = torch.FloatTensor(self.labels[0])
    #         mask = self.get_mask_from_file(filename)
    #     if self.config.TRAIN_MASK:
    #         _mask = image
    #         image = mask
    #         mask = _mask
    #     return image, label, mask, filename

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
        return image, label, mask, filename

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


def train_inception(batch_size=16, shuffling=False, num_workers=4, **kwargs):
    from torchvision.models import inception_v3
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    from misc.utils import to_cuda

    image_size = 299

    window = int(image_size / 10)
    transform_train = [
        transforms.Resize((image_size + window, image_size + window),
                          interpolation=Image.ANTIALIAS),
        transforms.RandomResizedCrop(image_size,
                                     scale=(0.7, 1.0),
                                     ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        transforms.Resize((image_size, image_size),
                          interpolation=Image.ANTIALIAS),
        transforms.ToTensor(),
    ]
    transform_test = transforms.Compose(transform_test)

    dataset_train = FFHQ(image_size,
                         'normal',
                         transform_train,
                         'train',
                         shuffling=True,
                         **kwargs)
    dataset_test = FFHQ(image_size,
                        'normal',
                        transform_test,
                        'val',
                        shuffling=False,
                        **kwargs)

    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    num_labels = len(train_loader.dataset.labels[0])
    n_epochs = 100
    net = inception_v3(pretrained=True, transform_input=True)
    net.aux_logits = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_labels)

    metadata_path = os.path.join('data', 'FFHQ', 'inception_v3')
    net_save = metadata_path + '/{}.pth'
    if not os.path.isdir(os.path.dirname(net_save)):
        os.makedirs(os.path.dirname(net_save))
    print("Model will be saved at: " + net_save)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5)
    # loss = F.cross_entropy(output, target)
    to_cuda(net)

    for epoch in range(n_epochs):
        net.eval()
        with torch.no_grad():
            loop(test_loader, net, 'Test')

        net.train()
        loop(train_loader, net, 'Train', optimizer=optimizer)
        torch.save(net.state_dict(), net_save.format(str(epoch).zfill(5)))
        train_loader.dataset.shuffle(epoch)


def loop(data_loader, net, mode='Train', optimizer=None):
    from misc.utils import to_cuda
    import torch
    import torch.nn.functional as F
    import tqdm
    LOSS = {mode: []}
    OUTPUT = {mode: []}
    LABEL = {mode: []}
    for i, (data, label,
            files) in tqdm.tqdm(enumerate(data_loader),
                                total=len(data_loader),
                                desc='{} Inception V3 | FFHQ'.format(mode)):
        data = to_cuda(data)
        label = to_cuda(label)
        out = net(data)
        loss = F.binary_cross_entropy_with_logits(
            out, label, reduction='sum') / label.size(0)
        if mode == 'Train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        LOSS[mode].append(loss.item())
        OUTPUT[mode].append(torch.sigmoid(out).detach().cpu())
        LABEL[mode].append(label.detach().cpu())
    acc = calculate_scores(OUTPUT[mode], LABEL[mode])
    print('[{:}] Loss: {:.4f} Acc: {:.3}%'.format(mode,
                                                  np.array(LOSS[mode]).mean(),
                                                  acc))


def calculate_scores(output, target):
    from sklearn.metrics import f1_score as f1s
    # from sklearn.metrics import precision_score, recall_score
    import numpy as np
    output = torch.cat(output).numpy()
    target = torch.cat(target).numpy()
    output = (output > 0.5) * 1.0
    F1 = []
    for i in range(output.shape[0]):
        F1.append(f1s(target[i], output[i]))
    #     print('Class {:2d}: {:.2f}%% F1'.format(i, F1[-1] * 100))
    F1_mean = np.array(F1).mean() * 100
    # print('Mean F1: {:.2f}%%'.format(F1_mean))
    return F1_mean


def show_me(args):
    from data_loader import get_transformations
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from misc.utils import denorm
    import numpy as np
    from misc.mask_utils import label2mask
    import matplotlib.pyplot as plt
    attrs = args.attr.split(',')
    assert len(attrs) >= 1, "Please provide at least one attribute"
    mode_data = 'faces'
    mode = 'train'
    img_size = 256
    transform = get_transformations(img_size, mode_data, mode='test')
    data = FFHQ(img_size,
                mode_data,
                transform,
                mode,
                show_attr=attrs,
                verbose=True)
    data_loader = DataLoader(dataset=data,
                             batch_size=64,
                             shuffle=False,
                             num_workers=4)
    for i, (data, label, mask, _) in enumerate(data_loader):
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='')
    parser.add_argument('--MASK', action='store_true', default=False)
    args = parser.parse_args()
    show_me(args)
