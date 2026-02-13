
import torch
from config import *
from data import get_dataloaders
from model import resnet18_cifar

import torch
from quantize import *
import argparse

@torch.no_grad()
def test(model,testloader,device):
    correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total
    return acc

if __name__ == "__main__":
    
    trainloader, testloader = get_dataloaders()
    
    model = resnet18_cifar().to(device)
    model.load_state_dict(torch.load(ckpt_path,weights_only=True))
    model.eval()

    test_acc = test(model,testloader, device)
    print(f"\nTest Acc={test_acc:.2f}%\n")

    