# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import *
from data import get_dataloaders
from model import resnet18_cifar

def train():
    trainloader, testloader = get_dataloaders()
    model = resnet18_cifar().to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_acc = 0.0

    for epoch in range(epochs):
        
        model.train()
        correct, total, loss_sum = 0, 0, 0

        for x, y in tqdm(trainloader, leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.size(0)

        test_acc = 100.0 * correct / total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
