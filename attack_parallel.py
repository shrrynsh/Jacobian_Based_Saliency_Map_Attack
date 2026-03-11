import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from jsma import JSMAAttack
from model import LeNet5


def get_args():
    p = argparse.ArgumentParser(description="Run parallel JSMA attack on MNIST")
    p.add_argument("--model_path", type=str, default="./checkpoints/lenet_mnist.pth")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--n_samples", type=int, default=10000, help="Number of source samples")
    p.add_argument(
        "--max_distortion",
        type=float,
        default=0.145,
        help="Max distortion ratio (paper: 0.145 = 14.5%)",
    )
    p.add_argument("--theta", type=float, default=1.0, help="Pixel change per iteration")
    p.add_argument(
        "--strategy",
        type=str,
        default="increase",
        choices=["increase", "decrease"],
        help="Saliency map strategy",
    )
    p.add_argument("--source_class", type=int, default=None, help="Restrict source class")
    p.add_argument("--target_class", type=int, default=None, help="Restrict target class")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--pin_memory", action="store_true", help="Enable pinned host memory")
    p.add_argument("--parallel_workers", type=int, default=8000, help="Concurrent attack workers")
    p.add_argument("--no_benchmark", action="store_true", help="Disable cuDNN benchmark autotuning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_test_data(
    data_dir: str,
    n_samples: int,
    source_class: int | None = None,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    if source_class is not None:
        indices = [i for i, (_, label) in enumerate(dataset) if label == source_class]
    else:
        indices = list(range(len(dataset)))

    sample_count = min(n_samples, len(indices))
    selected = rng.choice(indices, size=sample_count, replace=False).tolist()
    subset = Subset(dataset, selected)

    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


class AttackResults:
    def __init__(self):
        self.records = []

    def add(self, source: int, target: int, success: bool, distortion: float, n_iter: int):
        self.records.append(
            {
                "source": source,
                "target": target,
                "success": bool(success),
                "distortion": float(distortion),
                "n_iter": int(n_iter),
            }
        )

    def summary(self) -> dict:
        if not self.records:
            return {
                "total_attacks": 0,
                "n_success": 0,
                "success_rate_pct": 0.0,
                "avg_distortion_all_pct": 0.0,
                "avg_distortion_success_pct": 0.0,
            }

        total = len(self.records)
        successes = [r for r in self.records if r["success"]]
        n_success = len(successes)
        success_rate = 100.0 * n_success / total
        avg_distortion_all = 100.0 * float(np.mean([r["distortion"] for r in self.records]))
        avg_distortion_success = (
            100.0 * float(np.mean([r["distortion"] for r in successes])) if successes else 0.0
        )

        return {
            "total_attacks": total,
            "n_success": n_success,
            "success_rate_pct": round(success_rate, 2),
            "avg_distortion_all_pct": round(avg_distortion_all, 2),
            "avg_distortion_success_pct": round(avg_distortion_success, 2),
        }

    def per_class_summary(self) -> dict:
        pairs = defaultdict(list)
        for record in self.records:
            pairs[(record["source"], record["target"])].append(record)

        result = {}
        for (source, target), recs in pairs.items():
            successes = [r for r in recs if r["success"]]
            result[(source, target)] = {
                "n": len(recs),
                "n_success": len(successes),
                "success_rate": (len(successes) / len(recs)) if recs else 0.0,
                "avg_distortion_success": (
                    float(np.mean([r["distortion"] for r in successes])) if successes else 0.0
                ),
            }
        return result

    def to_numpy_matrices(self, num_classes: int = 10):
        pair_data = self.per_class_summary()
        success_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
        distortion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

        for (source, target), data in pair_data.items():
            success_matrix[source, target] = data["success_rate"]
            distortion_matrix[source, target] = data["avg_distortion_success"] * 100.0

        return success_matrix, distortion_matrix

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": self.records, "summary": self.summary()}, f, indent=2)
        print(f"Results saved to {path}")


def _chunked(items: list[int], chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def run_attack_parallel(args):
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.no_benchmark
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)

    parallel_workers = max(1, int(args.parallel_workers))
    worker_models = []
    worker_attacks = []
    worker_streams = []

    for _ in range(parallel_workers):
        worker_model = LeNet5().to(device)
        worker_model.load_state_dict(state_dict)
        worker_model.eval()
        worker_models.append(worker_model)
        worker_attacks.append(
            JSMAAttack(
                model=worker_model,
                theta=args.theta,
                max_distortion=args.max_distortion,
                increase=(args.strategy == "increase"),
                device=device,
            )
        )
        if device.type == "cuda":
            worker_streams.append(torch.cuda.Stream(device=device))
        else:
            worker_streams.append(None)

    loader = load_test_data(
        args.data_dir,
        args.n_samples,
        args.source_class,
        args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    target_classes = [args.target_class] if args.target_class is not None else list(range(10))

    results = AttackResults()
    os.makedirs(args.save_dir, exist_ok=True)

    total_attacks = len(loader.dataset) * len(target_classes)
    print("\nRunning parallel JSMA attack:")
    print(f"  Strategy:         {args.strategy} (theta={args.theta})")
    print(f"  Max distortion:   {args.max_distortion * 100:.1f}%")
    print(f"  Source samples:   {len(loader.dataset)}")
    print(f"  Target classes:   {target_classes}")
    print(f"  Parallel workers: {parallel_workers}")
    print(f"  Total attacks:    {total_attacks:,}")

    start_time = time.time()
    attack_count = 0
    pbar = tqdm(total=total_attacks, desc="Crafting adversarial samples (parallel)")

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        for images, labels in loader:
            image = images.to(device, non_blocking=args.pin_memory)
            source = int(labels.item())

            # Use worker 0 for source sanity check.
            pred_source = int(worker_models[0].predict(image).item())
            if pred_source != source:
                pbar.update(len(target_classes))
                continue

            valid_targets = [target for target in target_classes if target != source]
            skipped = len(target_classes) - len(valid_targets)
            if skipped > 0:
                pbar.update(skipped)

            for target_chunk in _chunked(valid_targets, parallel_workers):
                futures = []
                for worker_id, target in enumerate(target_chunk):
                    attack_obj = worker_attacks[worker_id]
                    stream = worker_streams[worker_id]
                    image_copy = image.detach().clone()

                    def _run_one(aobj=attack_obj, x=image_copy, tgt=target, s=stream):
                        if s is not None:
                            with torch.cuda.stream(s):
                                _, stats = aobj.craft(x, target_class=tgt, verbose=False)
                            s.synchronize()
                        else:
                            _, stats = aobj.craft(x, target_class=tgt, verbose=False)
                        return tgt, stats

                    futures.append(executor.submit(_run_one))

                for fut in futures:
                    target, stats = fut.result()
                    results.add(
                        source=source,
                        target=target,
                        success=bool(stats["success"]),
                        distortion=float(stats["distortion"]),
                        n_iter=int(stats["n_iter"]),
                    )
                    attack_count += 1
                    pbar.update(1)

                    if args.verbose:
                        status = "success" if stats["success"] else "fail"
                        print(
                            f"  {status} {source}->{target}: "
                            f"distortion={float(stats['distortion']) * 100:.1f}%, "
                            f"iters={int(stats['n_iter'])}"
                        )

    pbar.close()

    elapsed = time.time() - start_time
    summary = results.summary()

    print(f"\n{'=' * 60}")
    print("PARALLEL ATTACK RESULTS")
    print(f"{'=' * 60}")
    print(f"Total attacks:          {summary['total_attacks']:,}")
    print(f"Successful attacks:     {summary['n_success']:,}")
    print(f"Success rate (tau):     {summary['success_rate_pct']:.2f}%")
    print(f"Avg distortion (all):   {summary['avg_distortion_all_pct']:.2f}%")
    print(f"Avg distortion (eps):   {summary['avg_distortion_success_pct']:.2f}%")
    print(f"Time elapsed:           {elapsed:.1f}s")
    print(f"Time per attack:        {elapsed / max(attack_count, 1):.2f}s")
    print(f"{'=' * 60}")

    results_path = os.path.join(
        args.save_dir,
        f"results_parallel_{args.strategy}_{args.n_samples}samples_w{parallel_workers}.json",
    )
    results.save(results_path)

    success_matrix, distortion_matrix = results.to_numpy_matrices()
    np.save(os.path.join(args.save_dir, "success_matrix_parallel.npy"), success_matrix)
    np.save(os.path.join(args.save_dir, "distortion_matrix_parallel.npy"), distortion_matrix)

    print("\nPer-class success rates (rows=source, cols=target):")
    header = "    " + "  ".join(f"{target:4d}" for target in range(10))
    print(header)
    for source in range(10):
        row = f"{source:2d}  " + "  ".join(
            f"{success_matrix[source, target] * 100:3.0f}%" if source != target else "  --  "
            for target in range(10)
        )
        print(row)


if __name__ == "__main__":
    run_attack_parallel(get_args())
