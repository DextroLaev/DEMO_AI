
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import *

def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader
