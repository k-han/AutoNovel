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

import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader
from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler 
from .concat import ConcatDataset 

def find_classes_from_folder(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def find_classes_from_file(file_path):
    with open(file_path) as f:
            classes = f.readlines()
    classes = [x.strip() for x in classes] 
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, classes, class_to_idx):
    samples = []
    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                if 'JPEG' in path or 'jpg' in path:
                    samples.append(item)
    
    return samples 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None, loader=pil_loader):
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 images in subfolders \n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.samples=samples 
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)

def ImageNet882(aug=None, subfolder='train', path='./data/datasets/ImageNet/'):
    img_split = 'images/'+subfolder
    classes_118, class_to_idx_118 = find_classes_from_file(os.path.join(path, 'imagenet_rand118/imagenet_118.txt'))
    samples_118 = make_dataset(path+img_split, classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(os.path.join(path, img_split))
    classes_882 = list(set(classes_1000) - set(classes_118))
    class_to_idx_882 = {classes_882[i]: i for i in range(len(classes_882))}
    samples_882 = make_dataset(path+img_split, classes_882, class_to_idx_882)
    if aug==None:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='ktimes':
        transform = TransformKtimes(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), k=10)
    dataset = ImageFolder(transform=transform, samples=samples_882)
    return dataset 
    
def ImageNet30(path='./data/datasets/ImageNet/', subset='A', aug=None, subfolder='train'):
    classes_30, class_to_idx_30 = find_classes_from_file(os.path.join(path, 'imagenet_rand118/imagenet_30_{}.txt'.format(subset)))
    samples_30 = make_dataset(path+'images/{}'.format(subfolder), classes_30, class_to_idx_30)
    if aug==None:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='ktimes':
        transform = TransformKtimes(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), k=10)

    dataset = ImageFolder(transform=transform, samples=samples_30)
    return dataset

def ImageNetLoader30(batch_size, num_workers=2, path='./data/datasets/ImageNet/', subset='A', aug=None, shuffle=False, subfolder='train'):
    dataset = ImageNet30(path, subset, aug, subfolder)
    dataloader_30 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) 
    return dataloader_30

def ImageNetLoader882(batch_size, num_workers=2, path='./data/datasets/ImageNet/', aug=None, shuffle=False, subfolder='train'):
    dataset = ImageNet882(aug=aug, subfolder=subfolder, path=path)
    dataloader_882 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) 
    return dataloader_882

def ImageNetLoader882_30Mix(batch_size, num_workers=2, path='./data/datasets/ImageNet/',  unlabeled_subset='A', aug=None, shuffle=False, subfolder='train', unlabeled_batch_size=64):
    dataset_labeled = ImageNet882(aug=aug, subfolder=subfolder, path=path)
    dataset_unlabeled= ImageNet30(path, unlabeled_subset, aug, subfolder)
    dataset= ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled)) 
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size) 
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader 

