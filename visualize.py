"""
visualize.py
------------
Generate all paper figures:

  Figure 1  — 10×10 grid of adversarial samples (source → target)
  Figure 7  — Saliency map visualization for a single sample
  Figure 9  — Adversarial samples generated from empty input
  Figure 10 — 10×10 grid using decrease strategy
  Figure 12 — Success rate matrix heatmap
  Figure 13 — Average distortion matrix heatmap
  Figure 14 — Hardness matrix heatmap
  Figure 15 — Adversarial distance matrix heatmap
  Extra     — Pixel modification overlay (original + delta + adversarial)

Usage:
    python visualize.py
    python visualize.py --model_path checkpoints/lenet_mnist.pth
"""

import argparse
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import LeNet5
from jsma import JSMAAttack, compute_jacobian


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str, default="./checkpoints/lenet_mnist.pth")
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--output_dir",  type=str, default="./figures")
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_one_sample_per_class(data_dir: str, device, seed: int = 42):
    """Return one correctly-classified image per digit class."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    samples = {}   # class → (img, label)
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))

    for idx in indices:
        img, label = dataset[idx]
        if label not in samples:
            samples[label] = img.unsqueeze(0)
        if len(samples) == 10:
            break

    return {k: v.to(device) for k, v in samples.items()}


def img_to_np(t: torch.Tensor) -> np.ndarray:
    """Convert (1,1,28,28) or (1,28,28) tensor to (28,28) numpy."""
    return t.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Figure 1 & 10: 10×10 adversarial sample grids
# ---------------------------------------------------------------------------

def plot_adversarial_grid(
    model,
    samples: dict,
    device,
    strategy: str = "increase",
    theta: float = 1.0,
    max_distortion: float = 0.145,
    save_path: str = "figures/fig1_adversarial_grid.png",
    title: str = "Adversarial Samples (increase strategy)",
):
    """
    Recreate Figure 1 / Figure 10 from the paper.
    Grid[i,j] = adversarial sample crafted from digit i → classified as j.
    Diagonal = original samples.
    """
    num_classes = 10
    attack = JSMAAttack(
        model, theta=theta,
        max_distortion=max_distortion,
        increase=(strategy == "increase"),
        device=device,
    )

    grid = np.zeros((num_classes, num_classes, 28, 28))
    distortions = np.zeros((num_classes, num_classes))

    for source in range(num_classes):
        x = samples[source]
        # Diagonal: original image
        grid[source, source] = img_to_np(x)

        for target in range(num_classes):
            if target == source:
                continue
            x_adv, stats = attack.craft(x, target_class=target)
            grid[source, target] = img_to_np(x_adv)
            distortions[source, target] = stats["distortion"] * 100

    # Plot
    fig, axes = plt.subplots(num_classes, num_classes, figsize=(14, 14))
    fig.suptitle(title, fontsize=14, y=1.01)

    for s in range(num_classes):
        for t in range(num_classes):
            ax = axes[s, t]
            ax.imshow(grid[s, t], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if s == 0:
                ax.set_title(str(t), fontsize=8)
            if t == 0:
                ax.set_ylabel(str(s), fontsize=8)

            if s != t:
                ax.set_xlabel(f"{distortions[s,t]:.1f}%", fontsize=5)

    # Labels
    fig.text(0.5, -0.01, "Target class", ha="center", fontsize=12)
    fig.text(-0.01, 0.5, "Source class", va="center", rotation="vertical", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 7: Saliency map visualization
# ---------------------------------------------------------------------------

def plot_saliency_map(
    model,
    x: torch.Tensor,
    target_class: int,
    device,
    save_path: str = "figures/fig7_saliency_map.png",
):
    """
    Visualise the adversarial saliency map S(X,t) for a given sample.
    """
    x = x.to(device)
    jacobian = compute_jacobian(model, x.view(1, 1, 28, 28))  # (10, 784)

    target_grad = jacobian[target_class]                        # (784,)
    other_grad  = jacobian.sum(dim=0) - target_grad             # (784,)

    # Compute saliency (increase strategy)
    saliency = torch.zeros(784)
    for i in range(784):
        alpha = target_grad[i].item()
        beta  = other_grad[i].item()
        if alpha > 0 and beta < 0:
            saliency[i] = alpha * (-beta)

    saliency_map = saliency.cpu().numpy().reshape(28, 28)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axes[0].imshow(img_to_np(x), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original image", fontsize=11)
    axes[0].axis("off")

    # Target gradient ∂Ft/∂X
    tg = target_grad.cpu().numpy().reshape(28, 28)
    im1 = axes[1].imshow(tg, cmap="RdBu_r")
    axes[1].set_title(f"∂F_{target_class}/∂X", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Saliency map
    im2 = axes[2].imshow(saliency_map, cmap="hot")
    axes[2].set_title(f"Saliency map S(X,t={target_class})", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Adversarial Saliency Map — Source class: {model.predict(x).item()}, "
        f"Target: {target_class}",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 9: Adversarial samples from empty input
# ---------------------------------------------------------------------------

def plot_empty_input_adversarials(
    model,
    device,
    save_path: str = "figures/fig9_empty_input.png",
    max_distortion: float = 1.0,
):
    """
    Reproduce Figure 9: craft adversarial samples starting from a blank
    (all-zero) input for each target class 0–9.
    """
    attack = JSMAAttack(
        model, theta=1.0, max_distortion=max_distortion,
        increase=True, device=device
    )

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(
        "Figure 9: Adversarial samples from empty input (all pixels=0)",
        fontsize=12,
    )

    for target in range(10):
        empty = torch.zeros(1, 1, 28, 28, device=device)
        x_adv, stats = attack.craft(empty, target_class=target)

        row, col = divmod(target, 5)
        ax = axes[row, col]
        ax.imshow(img_to_np(x_adv), cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"Target: {target}\n"
            f"{'✓' if stats['success'] else '✗'} ε={stats['distortion']*100:.1f}%",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figures 12-15: Matrix heatmaps
# ---------------------------------------------------------------------------

def plot_matrix_heatmap(
    matrix: np.ndarray,
    title: str,
    xlabel: str = "Target class",
    ylabel: str = "Source class",
    fmt: str = ".0f",
    cmap: str = "Blues",
    save_path: str = "figures/matrix.png",
    mask_diag: bool = True,
):
    """Generic heatmap for 10×10 class-pair matrices."""
    fig, ax = plt.subplots(figsize=(8, 7))

    if mask_diag:
        plot_matrix = matrix.copy()
        np.fill_diagonal(plot_matrix, np.nan)
    else:
        plot_matrix = matrix

    im = ax.imshow(plot_matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    # Annotate cells
    for i in range(10):
        for j in range(10):
            if i == j and mask_diag:
                continue
            val = matrix[i, j]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=7, color=text_color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Extra: Original + Perturbation + Adversarial triplets
# ---------------------------------------------------------------------------

def plot_perturbation_overlay(
    model,
    samples: dict,
    device,
    n_examples: int = 5,
    save_path: str = "figures/perturbation_overlay.png",
):
    """Show original / perturbation / adversarial side-by-side."""
    attack = JSMAAttack(model, theta=1.0, max_distortion=0.145, device=device)

    fig, axes = plt.subplots(n_examples, 3, figsize=(7, n_examples * 2))
    fig.suptitle("Original → Perturbation → Adversarial", fontsize=12)

    titles = ["Original", "Perturbation δX (×5)", "Adversarial X*"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=10)

    for row, source in enumerate(list(samples.keys())[:n_examples]):
        x = samples[source]
        target = (source + 1) % 10   # craft source → source+1

        x_adv, stats = attack.craft(x, target_class=target)
        delta = (x_adv - x).squeeze().cpu().numpy()
        orig  = img_to_np(x)
        adv   = img_to_np(x_adv)

        axes[row, 0].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{source}→{target}", fontsize=9)

        # Amplify perturbation for visibility
        delta_vis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
        axes[row, 1].imshow(delta_vis, cmap="RdBu_r")
        axes[row, 1].set_xlabel(f"ε={stats['distortion']*100:.1f}%", fontsize=8)

        axes[row, 2].imshow(adv, cmap="gray", vmin=0, vmax=1)
        status = "✓" if stats["success"] else "✗"
        axes[row, 2].set_xlabel(f"{status} pred={stats.get('final_pred','?')}", fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = LeNet5.load_model(args.model_path, device)

    print("Loading one sample per class...")
    samples = get_one_sample_per_class(args.data_dir, device, args.seed)

    # --- Figure 1 ---
    print("\n[Fig 1] Adversarial grid (increase strategy)...")
    plot_adversarial_grid(
        model, samples, device,
        strategy="increase", theta=1.0,
        save_path=os.path.join(args.output_dir, "fig1_adversarial_grid_increase.png"),
        title="Figure 1: JSMA Adversarial Samples (θ=+1, Υ=14.5%)",
    )

    # --- Figure 7 ---
    print("\n[Fig 7] Saliency map visualization...")
    x_demo = samples[7]   # use digit '7'
    plot_saliency_map(
        model, x_demo, target_class=1, device=device,
        save_path=os.path.join(args.output_dir, "fig7_saliency_map.png"),
    )

    # --- Figure 9 ---
    print("\n[Fig 9] Empty-input adversarials...")
    plot_empty_input_adversarials(
        model, device,
        save_path=os.path.join(args.output_dir, "fig9_empty_input.png"),
    )

    # --- Figure 10 ---
    print("\n[Fig 10] Adversarial grid (decrease strategy)...")
    plot_adversarial_grid(
        model, samples, device,
        strategy="decrease", theta=-1.0,
        save_path=os.path.join(args.output_dir, "fig10_adversarial_grid_decrease.png"),
        title="Figure 10: JSMA Adversarial Samples (θ=−1, decrease strategy)",
    )

    # --- Extra: Perturbation overlay ---
    print("\n[Extra] Perturbation overlay...")
    plot_perturbation_overlay(
        model, samples, device,
        save_path=os.path.join(args.output_dir, "extra_perturbation_overlay.png"),
    )

    # --- Figures 12-15 from saved matrices ---
    results_dir = args.results_dir
    matrix_files = {
        "success_matrix.npy": (
            "Figure 12: Success Rate τ per Source-Target Pair",
            "Blues", ".0%", 
            os.path.join(args.output_dir, "fig12_success_matrix.png"),
        ),
        "distortion_matrix.npy": (
            "Figure 13: Avg Distortion ε (%) per Source-Target Pair",
            "Reds", ".1f",
            os.path.join(args.output_dir, "fig13_distortion_matrix.png"),
        ),
        "hardness_matrix.npy": (
            "Figure 14: Hardness H(s,t)",
            "Purples", ".3f",
            os.path.join(args.output_dir, "fig14_hardness_matrix.png"),
        ),
        "adversarial_distance_matrix.npy": (
            "Figure 15: Adversarial Distance A(X,t)",
            "Greens", ".3f",
            os.path.join(args.output_dir, "fig15_adversarial_distance.png"),
        ),
    }

    print("\n[Figs 12-15] Matrix heatmaps from saved results...")
    for fname, (title, cmap, fmt, save_path) in matrix_files.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            matrix = np.load(path)
            if "success" in fname:
                matrix_pct = matrix * 100
                plot_matrix_heatmap(matrix_pct, title, cmap=cmap, fmt=".0f",
                                    save_path=save_path)
            else:
                plot_matrix_heatmap(matrix, title, cmap=cmap, fmt=fmt,
                                    save_path=save_path)
        else:
            print(f"  Skipping {fname} — not found (run evaluate.py / attack.py first)")

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()