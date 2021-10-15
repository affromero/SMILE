from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import importlib
import torch
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
import numpy as np
import random
from munch import Munch

# ==================================================================#
# ==                           LOADER                             ==#
# ==================================================================#


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """
    def __init__(self, labels, indices=None, class_choice="random"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)

            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from.

            class_choice: a string indicating how class will be selected for every
            sample.
                "random": class is chosen uniformly at random.
                "cycle": the sampler cycles through the classes sequentially.
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))
        self.map = []
        for class_ in range(self.labels.shape[1]):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.map.append(lst)

        assert class_choice in ["random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class +
                                  1) % self.labels.shape[1]
        class_indices = self.map[class_]
        return np.random.choice(class_indices)

    def __len__(self):
        return len(self.indices)


def _make_balanced_sampler(labels, attr=''):
    # Only for binary attributes
    if attr != '':
        _labels = [np.argmax(l) for l in labels]
    else:
        _labels = [[idx] * i for idx, i in enumerate(np.sum(labels, 0))]
        _labels = np.concatenate(_labels)
    class_counts = np.bincount(_labels)
    class_weights = 1. / class_counts
    weights = class_weights[_labels]
    return WeightedRandomSampler(weights, len(weights))


def get_transformations(mode, image_size):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    common = []
    # if mode == 'train':
    #     common += [transforms.RandomHorizontalFlip()]
    common += [transforms.ToTensor()]
    common = transforms.Compose(common)
    resize_img = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=Image.ANTIALIAS)
    ])
    resize_mask = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=Image.NEAREST)
    ])
    norm_img = transforms.Compose([transforms.Normalize(mean, std)])
    transform = Munch(common=common,
                      resize_rgb=resize_img,
                      resize_mask=resize_mask,
                      norm=norm_img)
    return transform


def get_loader(config,
               batch_size=0,
               mode='',
               shuffling=False,
               verbose=False,
               **kwargs):
    dist = config.dist
    batch_size = config.batch_size if batch_size == 0 else batch_size
    dataset = str(config.dataset)
    mode = config.mode if mode == '' else mode
    if mode == 'test':
        image_size = config.image_size_test
    else:
        image_size = config.image_size
    num_workers = config.num_workers if mode != 'val' else 0
    transform = get_transformations(mode, image_size)
    dataset_module = getattr(
        importlib.import_module('datasets.{}'.format(dataset)), dataset)
    dataset = dataset_module(image_size,
                             transform,
                             mode,
                             shuffling=shuffling or mode == 'train',
                             verbose=verbose and dist.rank() == 0,
                             config=config,
                             **kwargs)
    if dist.size() == 1:
        sampler = None
        # import ipdb; ipdb.set_trace()
        if mode == 'train':
            # if config.ATTR:
            #     sampler = _make_balanced_sampler(dataset.labels)
            # else:
            sampler = MultilabelBalancedRandomSampler(np.array(dataset.labels),
                                                      class_choice='cycle')
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True,
                                 sampler=sampler,
                                 pin_memory=torch.cuda.is_available(),
                                 num_workers=num_workers)
    elif dist.size() != 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=dist.size(), rank=dist.rank())
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 drop_last=True,
                                 pin_memory=torch.cuda.is_available(),
                                 sampler=sampler)
    return data_loader
