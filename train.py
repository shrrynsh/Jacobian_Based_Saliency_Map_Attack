"""
train.py
--------
Train LeNet-5 on MNIST to reproduce the model used in Papernot et al.

Paper target accuracy (Section V / Appendix A):
  - Training accuracy: 98.93%
  - Test accuracy:     99.41%

Usage:
    python train.py
    python train.py --epochs 200 --batch_size 500 --lr 0.1
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train LeNet-5 on MNIST")
    p.add_argument("--epochs",     type=int,   default=200,  help="Training epochs (paper: 200)")
    p.add_argument("--batch_size", type=int,   default=1000000,  help="Batch size (paper: 500)")
    p.add_argument("--lr",         type=float, default=0.1,  help="Learning rate (paper: η=0.1)")
    p.add_argument("--data_dir",   type=str,   default="./data")
    p.add_argument("--save_dir",   type=str,   default="./checkpoints")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default=None,
                   help="cpu / cuda (default: auto-detect)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_dataloaders(data_dir: str, batch_size: int):
    """MNIST dataset with normalisation to [0, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),   # → [0, 1] float32
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # Use softmax output for loss computation
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Seed
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Data
    print("Loading MNIST...")
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Model
    model = LeNet5(num_classes=10).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    # Paper uses gradient descent; SGD without momentum approximates this
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()   # NLLLoss with log-softmax ≡ CrossEntropyLoss

    # Wrap model output with log for NLLLoss
    import torch.nn.functional as F

    def compute_loss(model, images, labels):
        log_probs = torch.log(model(images) + 1e-10)
        return criterion(log_probs, labels)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, images, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = model.predict(images)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Evaluate
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                loss = compute_loss(model, images, labels)
                test_loss += loss.item() * images.size(0)
                preds = model.predict(images)
                test_correct += preds.eq(labels).sum().item()
                test_total += images.size(0)

        test_loss /= test_total
        test_acc = 100.0 * test_correct / test_total

        # Save best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "lenet_mnist_best.pth"),
            )

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%  "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%  "
                f"[{elapsed:.0f}s]"
            )

    # Final save
    final_path = os.path.join(args.save_dir, "lenet_mnist.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best test accuracy:  {best_test_acc:.2f}%")
    print(f"  Final test accuracy: {test_acc:.2f}%")
    print(f"  Paper target:        99.41% test, 98.93% train")
    print(f"  Model saved to:      {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()