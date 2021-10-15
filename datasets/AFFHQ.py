import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
from glob import glob
# from misc.utils import timeit
module_path = os.path.abspath(os.getcwd())

if module_path not in sys.path:
    sys.path.append(module_path)

# ==================================================================#
# == CelebA_AFHQ
# ==================================================================#


class AFFHQ(Dataset):
    def __init__(self,
                 image_size,
                 mode_data,
                 transform,
                 mode,
                 shuffling=False,
                 all_attr=0,
                 verbose=False,
                 sampled=100,
                 show_attr='',
                 **kwargs):
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        # self.name = self.__class__.__name__
        self.name = 'AFFHQ'
        self.all_attr = all_attr
        self.mode_data = mode_data
        self.verbose = verbose
        self.show_attr = show_attr
        self.sampled = sampled  # How much data to train (percentage)
        self.data_dir = 'data/FFHQ/FFHQ-{}'.format(self.image_size)
        mode_animal = 'train' if self.mode == 'train' else 'test'
        self.animal_dir = 'data/AFHQ/afhq-{}/{}'.format(
            self.image_size, mode_animal)
        if 'config' in kwargs.keys():
            self.config = kwargs['config']
        else:
            self.config = None

        self.lines = [
            line.strip().split(',') for line in open(
                os.path.abspath('data/FFHQ/list_attr_ffhq.txt')).readlines()
        ]

        self.animal_lines = sorted(
            glob(os.path.join(self.animal_dir, '*', '*.jpg')))

        self.modes = [50000, 60000, 70000]
        files = [lines[0] for lines in self.lines[1:]]
        self.splits = {}
        self.MODES = {'train': 0, 'val': 1, 'test': 2}
        for idx, line in enumerate(files):
            if idx < self.modes[0]:
                self.splits[line] = self.MODES['train']
            elif idx > self.modes[1] and idx < self.modes[2]:
                self.splits[line] = self.MODES['val']
            else:
                self.splits[line] = self.MODES['test']

        self.mode_allowed = [self.MODES[self.mode]]
        # self.mode_allowed = [0, 1, 2]
        self.all_attr2idx = {}
        self.all_idx2attr = {}
        self.attr2idx = {}
        self.idx2attr = {}

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
        values = np.array([int(i) for i in self.lines[2][1:]]) * 0
        for line in self.lines[2:]:
            value = np.array([int(i) for i in line[1:]]).clip(min=0)
            values += value
        dict_ = {}
        for key, value in zip(self.lines[1], values):
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

    def subsample(self, filenames, labels):
        if self.sampled == 100 or self.mode not in ['train', 'val']:
            return filenames, labels
        else:
            self.num_data = int(self.num_data * self.sampled / 100)
            new_filenames = filenames[:self.num_data]
            new_labels = labels[:self.num_data]
            return new_filenames, new_labels

    def get_value(self, values, attr, animal=False):

        if animal:
            label = values.split('/')[-2]
            if attr.lower() == label:
                return 1
            else:
                return 0

        def str2int(v):
            return int(values[self.all_attr2idx[v]])

        value = 0
        NOT = False
        if 'NOT_' in attr:
            NOT = True
            attr = attr.replace('NOT_', '')

        if attr in self.all_attr2idx.keys():
            value = str2int(attr)
        else:
            if attr in ['General_Style']:
                value = 1

            elif attr in ['Female', 'Genre']:
                if str2int('Male') != 1:
                    value = 1

            elif attr in ['Aged']:
                if str2int('Young') != 1:
                    value = 1

            elif attr in ['Color_Hair', 'Color_Bangs']:
                value = 0
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
                if str2int('No_Beard') != 1:
                    value = 1
                else:
                    value = 0
                    _attrs = ['5_o_Clock_Shadow', 'Mustache', 'Goatee']
                    for _attr in _attrs:
                        if str2int(_attr) == 1:
                            value = 1
                            break
            elif attr == 'Hair':
                if str2int('Bald') != 1:
                    value = 1

            else:
                value = 0

        if NOT:
            value = int(not (max(value, 0)))

        return value

    def preprocess(self):
        attrs = self.lines[0][1:]
        assert len(attrs) == 40
        # if self.verbose:
        #     self.histogram()

        for i, attr in enumerate(attrs):
            self.all_attr2idx[attr] = i
            self.all_idx2attr[i] = attr

        if self.show_attr != '':
            self.selected_attrs = self.show_attr

        elif self.all_attr == 1:
            self.selected_attrs = [
                '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                'Wearing_Necktie', 'Young'
            ]  # Total: 40

        else:
            if self.config is not None:
                if self.config.ATTR == 'beings':
                    self.selected_attrs = [
                        'Male', 'Female', 'Cat', 'Dog', 'Wild'
                    ]
                elif self.config.ATTR == 'single':
                    self.selected_attrs = ['General_Style']
                else:
                    self.selected_attrs = [
                        'Eyeglasses',
                        'Smiling',
                        'Bangs',
                        # 'Hair',
                        # 'Color_Hair',
                        'Male',
                        'Female',
                        'Cat',
                        'Dog',
                        'Wild',
                        # 'Young',
                        # 'Aged',
                        # 'Attractive',
                        # 'Facial_Hair'
                    ]

        self.parent_attrs = {
            'Eyeglasses': ['Eyeglasses'],
            'Smile': ['Smiling'],
            'Bangs': ['Bangs'],
            # 'Color_Hair': ['Color_Hair'],
            'Beings': ['Male', 'Female', 'Cat', 'Dog', 'Wild'],
            # 'Makeup': ['Heavy_Makeup'],
            # 'Age': ['Young', 'Aged'],
        }
        self.children_attrs = self.selected_attrs

        for i, attr in enumerate(self.selected_attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
        self.filenames = []
        self.labels = []
        # lines = self.lines[2:]
        lines = self.lines[1:]
        # lines = self.subsample(lines)
        # random.shuffle(lines)
        # if self.shuffling:
        #     random.shuffle(lines)
        # import ipdb; ipdb.set_trace()
        for i, line in enumerate(lines):
            if not self.image_exist(line[0]):
                continue
            values = line[1:]
            label = []

            no_show_attr = False
            for attr in self.selected_attrs:
                selected_value = self.get_value(values, attr)
                if selected_value >= 1:
                    label.append(selected_value)
                else:
                    label.append(0)
                    no_show_attr = True

            if self.show_attr and no_show_attr:
                continue

            filename = os.path.abspath(os.path.join(self.data_dir, line[0]))
            self.filenames.append(filename)
            self.labels.append(label)
        for i, line in enumerate(self.animal_lines):
            label = []
            for attr in self.selected_attrs:
                selected_value = self.get_value(line, attr, animal=True)
                if selected_value >= 1:
                    label.append(selected_value)
                else:
                    label.append(0)
            self.filenames.append(line)
            self.labels.append(label)

        self.shuffle(11)
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

    def __getitem__(self, index):
        # filename = os.path.join(self.data_dir, self.filenames[index])
        filename = self.filenames[index]
        image = Image.open(filename).convert('RGB')
        label = torch.FloatTensor(self.labels[index])
        return self.transform(image), label, filename

    def __len__(self):
        return self.num_data

    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.filenames)
        random.seed(seed)
        random.shuffle(self.labels)


def train_inception(batch_size=16, shuffling=False, num_workers=4, **kwargs):
    from torchvision.models import inception_v3
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    from misc.utils import to_cuda
    from datasets.CelebA import CelebA

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

    dataset_train = CelebA(image_size,
                           'normal',
                           transform_train,
                           'train',
                           shuffling=True,
                           **kwargs)
    dataset_test = CelebA(image_size,
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

    metadata_path = os.path.join('data', 'CelebA', 'inception_v3')
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
                                desc='{} Inception V3 | CelebA'.format(mode)):
        data = to_cuda(data)
        label = to_cuda(label)
        out = net(data)
        loss = F.binary_cross_entropy_with_logits(
            out, label, reduction='sum') / label.size(0)
        # import ipdb; ipdb.set_trace()
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
    # import ipdb
    # ipdb.set_trace()
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
    import matplotlib.pyplot as plt
    attrs = args.attr.split(',')
    assert len(attrs) >= 1, "Please provide at least one attribute"
    mode_data = 'faces'
    mode = 'train'
    transform = get_transformations(256, mode_data, mode)
    data = AFFHQ(256,
                 mode_data,
                 transform,
                 mode,
                 show_attr=attrs,
                 verbose=True)
    data_loader = DataLoader(dataset=data,
                             batch_size=64,
                             shuffle=False,
                             num_workers=4)
    for i, (data, label, _) in enumerate(data_loader):
        data = denorm(data)
        data = make_grid(data).numpy()
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(data, (1, 2, 0)), interpolation='nearest')
        plt.show(block=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='')
    args = parser.parse_args()
    # train_inception()
    show_me(args)
