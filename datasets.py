import torch
import torchvision
import torchvision.transforms as transforms


import numpy as np


def get_datasets():
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    validset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    return trainset, trainloader, validset, validloader


