import arparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from model import LeNet5

def get_args():
    p = argparse.ArgumentParser(description="Train LeNet-5 on MNIST")
    p.add_argument("--epochs",     type=int,   default=200,  help="Training epochs (paper: 200)")
    p.add_argument("--batch_size", type=int,   default=500,  help="Batch size (paper: 500)")
    p.add_argument("--lr",         type=float, default=0.1,  help="Learning rate (paper: η=0.1)")
    p.add_argument("--data_dir",   type=str,   default="./data")
    p.add_argument("--save_dir",   type=str,   default="./checkpoints")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default=None,
                   help="cpu / cuda (default: auto-detect)")
    return p.parse_args()


def get_dataloaders(data_dir : str, batch_size : int):
    transform=transforms.Compose([
        transforms.ToTensor(),

    ])

    train_dataset=datasets.MNIST(
        root=data_dir,train=True,download=True,transform=transform
    )

    test_dataset=datasets.MNIST(
        root=data_dir,train=False,download=True,transform=transform
    )
