import logging
import os

import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C
import random
from augmentations import *
from common import get_logger
from imagenet import ImageNet
#from CIFAR10_cut import CIFAR10_PC

from augmentations import Lighting, RandAugment, RWAug_Search,RWAug_Train,RandAugment_th

logger = get_logger('DDAS')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    logger.debug('augmentation: %s' % C.get()['aug'])
    if C.get()['aug'] == 'randaugment':
        transform_train.transforms.insert(0, RandAugment(C.get()['randaug']['N'], C.get()['randaug']['M']))
    elif C.get()['aug'] == 'randaugment_th':
        transform_train.transforms.insert(0, RandAugment_th(C.get()['randaug']['N'], C.get()['randaug']['M']))
    elif C.get()['aug'] == 'randaugment_ohl':
        # May 6th add for baseline "Total Random!"
        transform_train.transforms.insert(0, RandAugment_ohl(C.get()['randaug']['N']))
    elif C.get()['aug'] == 'curriculum_aug':
        # May 13th add for baseline circul aug
        transform_train.transforms.insert(0, Curriculum_Aug(C.get()['curriculum_aug']['N'], C.get()['curriculum_aug']['T']))
    elif C.get()['aug'] == 'randaugment_G':
        transform_train.transforms.insert(0, RandAugment_G(C.get()['randaug']['N'], C.get()['randaug']['M']))
    elif C.get()['aug'] == 'randaugment_C':
        transform_train.transforms.insert(0, RandAugment_C(C.get()['randaug']['N'], C.get()['randaug']['M']))
    elif C.get()['aug'] == 'rwaug_t':
        transform_train.transforms.insert(0, RWAug_Train(C.get()['rwaug']['n']))
    elif C.get()['aug'] in ['default', 'inception', 'inception320','mix']:
        pass
    else:
        raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=transform_test)
        # total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet'), transform=transform_train)
        # testset = ImageNet(root=os.path.join(dataroot, 'imagenet'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers = 16, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers = 16, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers = 16, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
def get_val_test_dataloader(dataset, batch, dataroot, split = 0.1):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    if dataset == 'cifar10' or dataset ==  'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_test)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_test)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'svhn' or dataset == 'svhn_core':
        
        total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_test)
        total_trainset.targets = [lb for lb in total_trainset.labels]
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=transform_test)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=transform_test)
        
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    else:
        raise ValueError('invalid dataset name=%s' % dataset)
    train_sampler = None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=0)
    if dataset ==  'reduced_cifar10':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split/10, random_state=0)
    sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
    train_idx, valid_idx = next(sss)
    #assuming that train idx is smaller than train index!
    valid_idx = valid_idx[0:len(train_idx)]
    train_sampler = SubsetSampler(train_idx)
    if dataset ==  'reduced_cifar10':
        train_idx = train_idx[0:4000]
        train_sampler = SubsetSampler(train_idx)
    train_target = [total_trainset.targets[idx] for idx in train_idx]
    
    train_target = [total_trainset.targets[idx] for idx in train_idx]
    label_freq = {}
    for lb in set(train_target):
        label_freq[lb] = train_target.count(lb)
    print(label_freq)
    print("length of train idx")
    print(len(train_idx))
    print(len(valid_idx))
    print(len(train_sampler.indices))
    valid_sampler = SubsetSampler(valid_idx)

    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
        drop_last=False
    )
    return train_sampler, validloader, testloader

#fixed batch size for each data loader!
def Get_DataLoaders_Epoch_s(dataset, batch, dataroot, random_sampler, AugTypes, loader_num = 4):
    loaders = []
    idx_epoch = []
    assert len(AugTypes) == loader_num
    for idx in random_sampler:
        idx_epoch.append(idx)
    #turn random sample to fixed id sampler!
    SubsetSampler_epoch = SubsetSampler(idx_epoch)
    for i in range(loader_num):
        loaders.append(get_dataloader_epoch(dataset, batch, dataroot, sampler = SubsetSampler_epoch, AugType = AugTypes[i]))
    #here to delet the augmentation in the 1st loader
    print(loaders[0].dataset.transform.transforms.pop(0))
    return loaders

def get_dataloader_epoch(dataset, batch, dataroot, sampler=None, AugType = (2,5)):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    #logger.debug('augmentation: %s' % C.get()['aug'])
    if C.get()['aug'] == 'randaugment':
        transform_train.transforms.insert(0, RandAugment(C.get()['randaug']['N'], C.get()['randaug']['M']))
    elif C.get()['aug'] == 'rwaug_s':
        transform_train.transforms.insert(0, RWAug_Search(C.get()['rwaug']['n'],AugType[1]))
    elif C.get()['aug'] == 'randaugment_G':
        transform_train.transforms.insert(0, RandAugment_G(AugType[0], AugType[1]))
    elif C.get()['aug'] == 'randaugment_C':
        transform_train.transforms.insert(0, RandAugment_C(AugType[0], AugType[1]))
    elif C.get()['aug'] in ['default', 'inception', 'inception320','mix']:
        pass
    else:
        raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset == 'cifar10' or dataset == 'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
    elif dataset == 'svhn_core':
        total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
    elif dataset == 'imagenet':
        total_trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=transform_train)
        # testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=transform_test)
        # total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = sampler

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=1, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return trainloader

if __name__ == '__main__':
    a=[1,2,3,4,5,6,7,8]
    sb=SubsetSampler(a)
    for i in sb:
        print(i)
