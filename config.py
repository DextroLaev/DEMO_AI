import torch
epochs = 200
batch_size = 256
test_batch_size = 256
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
label_smoothing = 0.1
num_classes = 10
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

data_root = "./data"
ckpt_path = "./resnet18_cifar10.pth"
