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
# == AFHQ
# ==================================================================#


class AFHQ(Dataset):
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
                 zero_domain=False,
                 COLORS=False,
                 **kwargs):
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        self.name = self.__class__.__name__
        self.all_attr = all_attr
        self.mode_data = mode_data
        self.verbose = verbose
        self.show_attr = show_attr
        self.sampled = sampled  # How much data to train (percentage)
        mode_animal = 'train' if self.mode == 'train' else 'test'
        self.animal_dir = 'data/AFHQ/afhq-{}/{}'.format(
            self.image_size, mode_animal)

        self.animal_lines = sorted(
            glob(os.path.join(self.animal_dir, '*', '*.jpg')))

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

    def subsample(self, filenames, labels):
        if self.sampled == 100 or self.mode not in ['train', 'val']:
            return filenames, labels
        else:
            self.num_data = int(self.num_data * self.sampled / 100)
            new_filenames = filenames[:self.num_data]
            new_labels = labels[:self.num_data]
            return new_filenames, new_labels

    def get_value(self, values, attr):
        label = values.split('/')[-2]
        if attr.lower() == label:
            return 1
        else:
            return 0

    def preprocess(self):
        self.selected_attrs = [
            i.capitalize() for i in os.listdir(self.animal_dir)
        ]
        self.parent_attrs = {i: [i] for i in self.selected_attrs}
        self.children_attrs = self.selected_attrs
        self.filenames = []
        self.labels = []

        for i, line in enumerate(self.animal_lines):
            label = []
            for attr in self.selected_attrs:
                # import ipdb; ipdb.set_trace()
                selected_value = self.get_value(line, attr)
                if selected_value >= 1:
                    label.append(selected_value)
                else:
                    label.append(0)
            self.filenames.append(line)
            self.labels.append(label)

        self.shuffle(123)
        self.num_data = len(self.filenames)

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
    dataset = 'CelebA_HQ'
    mode_data = 'faces'
    mode = 'train'
    transform = get_transformations(256, mode_data, mode, dataset)
    data = AFHQ(256,
                mode_data,
                transform,
                mode,
                show_attr=attrs,
                COLORS=True,
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
