import logging
import math

import os
import sys
import pickle

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch
import torchvision
import torch.utils.data as data
import random
from PIL import ImageFilter
from .randaugment import RandAugmentMC
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader

from torch.utils.data import Dataset

import json
import math
import PIL.Image
import copy
import csv

from .iNatDataset import iNatDataset

logger = logging.getLogger(__name__)


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)


dset_root = {}
dset_root['cub'] = '../data/cub/images'
dset_root['semi_fungi'] = '../data/semi_fungi'
dset_root['semi_aves'] = '../data/semi_aves'
dset_root['semi_inat'] = '../data/semi_inat'



def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])


def get_cifar10(cfg):
    resize_dim = 256
    crop_dim = 224
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    base_dataset = datasets.CIFAR10(cfg.DATA.DATAPATH, train=True, download=True)

    l_samples = make_imb_data(cfg.DATA.NUM_L, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_L)
    u_samples = make_imb_data(cfg.DATA.NUM_U, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_U, cfg.flag_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, cfg)

    train_labeled_dataset = CIFAR10SSL(
        cfg.DATA.DATAPATH, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        cfg.DATA.DATAPATH, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch_ws(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        cfg.DATA.DATAPATH, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(cfg):
    resize_dim = 256
    crop_dim = 224
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim*0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        cfg.DATA.DATAPATH, train=True, download=True)

    l_samples = make_imb_data(cfg.DATA.NUM_L, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_L)
    u_samples = make_imb_data(cfg.DATA.NUM_U, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_U, cfg.flag_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, cfg)

    train_labeled_dataset = CIFAR100SSL(
        cfg.DATA.DATAPATH, train_labeled_idxs, train=True,
        transform=transform_labeled, cls_list=l_samples)

    train_unlabeled_dataset = CIFAR100SSL(
        cfg.DATA.DATAPATH, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch_ws(mean=cifar100_mean, std=cifar100_std), cls_list=u_samples)

    test_dataset = datasets.CIFAR100(
        cfg.DATA.DATAPATH, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_semi_fungi(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.466, 0.471, 0.380)
    dataset_std = (0.195, 0.194, 0.192)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])


    l_train = 'l_train'
    if cfg.dataset == 'Semi_Fungi_out':
        u_train = 'u_train_inout'
    else:
        u_train = 'u_train_in'

    train_labeled_dataset = iNatDataset(dset_root['semi_fungi'], l_train, 'semi_fungi', transform= transform_labeled)

    train_unlabeled_dataset = iNatDataset(dset_root['semi_fungi'], u_train, 'semi_fungi', transform= TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = iNatDataset(dset_root['semi_fungi'], 'test', 'semi_fungi', transform= transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_semi_inat(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.466, 0.471, 0.380)
    dataset_std = (0.195, 0.194, 0.192)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])


    l_train = 'l_train'
    if cfg.dataset == 'Semi_Inat_out':
        u_train = 'u_train'
    else:
        u_train = 'u_train_in'

    train_labeled_dataset = iNatDataset(dset_root['semi_inat'], l_train, 'semi_inat', transform= transform_labeled)

    train_unlabeled_dataset = iNatDataset(dset_root['semi_inat'], u_train, 'semi_inat', transform= TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = iNatDataset(dset_root['semi_inat'], 'test', 'semi_inat', transform= transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cub(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    l_train = 'l_train'
    if cfg.dataset == 'Cub_out':
        u_train = 'u_train'
    else:
        u_train = 'u_train_in'

    train_labeled_dataset = iNatDataset(dset_root['cub'], l_train, 'cub', transform= transform_labeled)

    train_unlabeled_dataset = iNatDataset(dset_root['cub'], u_train, 'cub', transform= TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = iNatDataset(dset_root['cub'], 'test', 'cub', transform= transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_semi_aves(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    l_train = 'l_train'
    if cfg.dataset == 'Semi_Aves_out':
        u_train = 'u_train'
    else:
        u_train = 'u_train_in'

    train_labeled_dataset = iNatDataset(dset_root['semi_aves'], l_train, 'semi_aves', transform= transform_labeled)

    train_unlabeled_dataset = iNatDataset(dset_root['semi_aves'], u_train, 'semi_aves', transform= TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = iNatDataset(dset_root['semi_aves'], 'test', 'semi_aves', transform= transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(cfg.DATA.NUMBER_CLASSES):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs


def train_split_l(labels, n_labeled_per_class, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(cfg.DATA.NUMBER_CLASSES):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs


def make_imbalance(dataset, indexs):
    dataset.data = dataset.data[indexs]
    dataset.labels = dataset.labels[indexs]
    return dataset


def make_imb_data(max_num, class_num, gamma, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    
    if flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    print(class_num_list)
    return list(class_num_list)


class GaussianBlur(object):
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    


class TransformFixMatch_ws(object):
    def __init__(self, mean, std, img_size=32):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])

        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)


class TransformFixMatchSTL(object):
    def __init__(self, mean, std):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None, cls_list=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.cls_list = cls_list
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_stl10(cfg):
    resize_dim = 256
    crop_dim = 224
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim*0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    train_labeled_dataset = STL10SSL(cfg.DATA.DATAPATH, split="train", transform=transform_labeled, download=True)
    train_unlabeled_dataset = STL10SSL(cfg.DATA.DATAPATH, split="unlabeled",
                                             transform=TransformFixMatchSTL(mean=stl10_mean, std=stl10_std),
                                             download=True)
    test_dataset = STL10SSL(cfg.DATA.DATAPATH, split="test", transform=transform_val, download=True)

    l_samples = make_imb_data(cfg.DATA.NUM_L, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_L, cfg.flag_LT)
    train_labeled_idxs = train_split_l(train_labeled_dataset.labels, l_samples, cfg)
    train_labeled_dataset = make_imbalance(train_labeled_dataset, train_labeled_idxs)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class STL10SSL(datasets.STL10):
    def __init__(self, root, split, transform=None, target_transform=None, cls_list=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.cls_list = cls_list

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index




DATASET_GETTERS = {'CIFAR10': get_cifar10,
                   'CIFAR100': get_cifar100,
                   'Semi_iNat': get_semi_inat,
                   'Semi_Fungi': get_semi_fungi,
                   'Semi_Aves': get_semi_aves,
                   'CUB': get_cub}

