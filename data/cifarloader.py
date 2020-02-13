from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import random
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler
from .concat import ConcatDataset
import torchvision.transforms as transforms

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, split='train+test',
                 transform=None, target_transform=None,
                 download=False, target_list = range(5)):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        downloaded_list = []
        if split=='train':
            downloaded_list = self.train_list
        elif split=='test':
            downloaded_list = self.test_list
        elif split=='train+test':
            downloaded_list.extend(self.train_list)
            downloaded_list.extend(self.test_list)

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    #  self.targets.extend(entry['coarse_labels'])
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        ind = [i for i in range(len(self.targets)) if self.targets[i] in target_list]

        self.data = self.data[ind]
        self.targets = np.array(self.targets)
        self.targets = self.targets[ind].tolist()



    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        #  x = self.class_to_idx
        #  sorted_x = sorted(x.items(), key=lambda kv: kv[1])
        #  print(sorted_x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        #  'key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

def CIFAR10Data(root, split='train', aug=None, target_list=range(5)):
    if aug==None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    dataset = CIFAR10(root=root, split=split, transform=transform, target_list=target_list)
    return dataset

def CIFAR10Loader(root, batch_size, split='train', num_workers=2,  aug=None, shuffle=True, target_list=range(5)):
    dataset = CIFAR10Data(root, split, aug,target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR10LoaderMix(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10), new_labels=None):
    if aug==None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    dataset_labeled = CIFAR10(root=root, split=split, transform=transform, target_list=labeled_list)
    dataset_unlabeled = CIFAR10(root=root, split=split, transform=transform, target_list=unlabeled_list)
    if new_labels is not None:
        dataset_unlabeled.targets = new_labels
    dataset_labeled.targets = np.concatenate((dataset_labeled.targets,dataset_unlabeled.targets))
    dataset_labeled.data = np.concatenate((dataset_labeled.data,dataset_unlabeled.data),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR10LoaderTwoStream(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10), unlabeled_batch_size=64):
    dataset_labeled = CIFAR10Data(root, split, aug, labeled_list)
    dataset_unlabeled =  CIFAR10Data(root, split, aug, unlabeled_list)
    dataset = ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled))
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size)
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader


def CIFAR100Data(root, split='train', aug=None, target_list=range(80)):
    if aug==None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ]))
    dataset = CIFAR100(root=root, split=split, transform=transform, target_list=target_list)
    return dataset

def CIFAR100Loader(root, batch_size, split='train', num_workers=2,  aug=None, shuffle=True, target_list=range(80)):
    dataset = CIFAR100Data(root, split, aug,target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR100LoaderMix(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(80), unlabeled_list=range(90, 100)):
    dataset_labeled = CIFAR100Data(root, split, aug, labeled_list)
    dataset_unlabeled = CIFAR100Data(root, split, aug, unlabeled_list)
    dataset_labeled.targets = np.concatenate((dataset_labeled.targets,dataset_unlabeled.targets))
    dataset_labeled.data = np.concatenate((dataset_labeled.data,dataset_unlabeled.data),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR100LoaderTwoStream(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(80), unlabeled_list=range(90, 100), unlabeled_batch_size=32):
    dataset_labeled = CIFAR100Data(root, split, aug, labeled_list)
    dataset_unlabeled = CIFAR100Data(root, split, aug, unlabeled_list)
    dataset = ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled))
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size)
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader