""" Dataset partitioning helper """
import argparse
import itertools
from random import Random

import torch
import torchvision
from torchvision.transforms import transforms
from torch import distributed as dist
import numpy as np
import random

np.random.seed(42)
import torchvision.datasets as dset


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, samples=50000, sizes=[0.7, 0.2, 0.1]):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    net_dataidx_map={}
    for i,client_idc in enumerate(client_idcs):
        np.random.shuffle(client_idc)
        net_dataidx_map[i] = client_idc.tolist()
    # data_ratio = [len(c) for c in client_idcs]
    #
    # data_ratio = np.array(data_ratio) / np.sum(data_ratio)

    return net_dataidx_map


class Non_IID_DataPartitioner(object):

    def __init__(self, data, dir=1, size=5, samples=50000):
        self.data = data
        self.partitions = []
        self.data_ratio = []
        train_labels = np.array(self.data.targets[0:samples])
        self.partitions, self.data_ratio = split_noniid(train_labels, alpha=dir, n_clients=size)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class class_DataPartitioner(object):
    def __init__(self, data, size=5):
        self.data = data
        data_len = len(self.data)
        indexes = [x for x in range(0, data_len)]
        data_dict = dict(zip(indexes, data.targets))
        data_dict_exchange = {}
        for key, value in data_dict.items():
            if value not in data_dict_exchange:
                data_dict_exchange[value] = [key]
            else:
                data_dict_exchange[value].append(key)

        values_list = list(data_dict_exchange.values())
        random.shuffle(values_list)
        values_list = list(itertools.chain(*values_list))
        result = np.array_split(values_list, size * 2)
        self.partitions = []
        my_list = random.sample(range(size * 2), size * 2)
        for i in range(size):
            self.partitions.append(list(result[my_list[i * 2]]) + list(result[my_list[i * 2 + 1]]))

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class IID_DataPartitioner(object):

    def __init__(self, data, frac_sizes, samples=50000):
        self.data = data
        self.partitions = []

        data_len = samples
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)
        for frac in frac_sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning """


def partition_dataset(clients,name, root,batchsize,data_split_type,dir, samples=50000):
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root, train=True, download=False,
                                               transform=transforms.Compose([
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   torchvision.transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ]))
        testset = torchvision.datasets.CIFAR10(root, train=False, download=False,
                                               transform=transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ]))
    elif name == "mnist":
        dataset = torchvision.datasets.MNIST(root, train=True, download=False,
                                             transform=transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                     (0.1307,), (0.3081,))
                                             ]))
        testset = torchvision.datasets.MNIST(root, train=False, download=False,
                                             transform=transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                     (0.1307,), (0.3081,))
                                             ]))
    elif name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root, train=True, download=False,
                                                transform=transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                ]))
        testset = torchvision.datasets.CIFAR100(root, train=False, download=False,
                                                transform=transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                ]))
    elif name == "tiny-imagenet":
        tinyimagenet_mean = (0.4802, 0.4481, 0.3975)
        tinyimagenet_std = (0.2302, 0.2265, 0.2262)

        dataset = dset.ImageFolder("./data/tiny-imagenet-200/train",
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(64),
                                       transforms.RandomHorizontalFlip(),
                                       torchvision.transforms.ToTensor(),
                                       transforms.Normalize(tinyimagenet_mean, tinyimagenet_std)]))
        testset = dset.ImageFolder("./data/tiny-imagenet-200/val",
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(64),
                                       torchvision.transforms.ToTensor(),
                                   transforms.Normalize(tinyimagenet_mean, tinyimagenet_std)]))

    idxs_clients = clients
    data_ratio = [1.0 / idxs_clients for _ in range(idxs_clients)]
    if data_split_type == "iid":
        partitioner = IID_DataPartitioner(dataset, data_ratio, samples)
    elif data_split_type == "niid_class":
        partitioner = class_DataPartitioner(dataset, idxs_clients)
    else:
        partitioner = Non_IID_DataPartitioner(dataset, dir=dir, size=idxs_clients, samples=samples)
        data_ratio = partitioner.data_ratio
    trainloader_list=[]
    data_size_list=[]
    for client_id in range(clients):
        partition = partitioner.use(client_id)
        data_size_list.append(len(partition))
        trainloader = torch.utils.data.DataLoader(partition,
                                                  batch_size=batchsize,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=1,
                                                  )
        trainloader_list.append(trainloader)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=False,
                                             # drop_last=True,
                                             # pin_memory=True,
                                             num_workers=1)
    data_distributed={}
    data_distributed["trainloader_list"]=trainloader_list
    data_distributed["testloader"] = testloader
    data_distributed["data_size_list"] = data_size_list
    data_distributed["clients"] = clients
    return data_distributed


def dataset():
    dataset = torchvision.datasets.CIFAR10('./data', train=True, download=False,
                                           transform=transforms.Compose([
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.RandomHorizontalFlip(),
                                               torchvision.transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ]))
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=False,
                                           transform=transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ]))
    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=256,
                                              shuffle=True,
                                              drop_last=True,
                                              pin_memory=True,
                                              num_workers=1)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=True,
                                             drop_last=True,
                                             pin_memory=True,
                                             num_workers=1)

    return trainloader, testloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--node_nums', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--batchsize', type=float, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--data_split_type', type=str, default="niid_class")
    parser.add_argument('--algorithm', type=str, default="fedavg")
    parser.add_argument('--dir', type=float, default=0.1)
    parser.add_argument('--gpu_nums', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.data_path = "../data"
    trainloader, testloader, data_ratio = partition_dataset(20, 'cifar10', '../../data',64,'niid',0.1)
    print(len(trainloader))
    for i, j in trainloader:
        print(j)
