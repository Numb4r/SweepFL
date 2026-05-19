#!/usr/bin/env python3
"""
Standalone MobileNetV2 + CIFAR-10 FL benchmark.

Simulates FedAvg with N_CLIENTS clients: training data is split IID into
N_CLIENTS shards; each round all clients train locally for LOCAL_EPOCHS and
their weights are averaged (FedAvg). The global model is then evaluated on the
full test set. This mirrors the structure used in the actual FL experiments.

Usage:
    # All batch sizes sequentially:
    python3 benchmark/train_mobilenetv2_cifar10.py

    # Single batch size (used by SLURM job array):
    python3 benchmark/train_mobilenetv2_cifar10.py --batch-size 32

    # Quick smoke-test:
    python3 benchmark/train_mobilenetv2_cifar10.py --batch-size 32 --rounds 5

Results are saved to benchmark/results/:
    batch_<N>.csv  — per-round log for each batch size
    summary.csv    — best/final val_acc and total wall time per batch size
"""

import argparse
import csv
import gc
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf
from client.models import create_mobilenetv2

BATCH_SIZES  = [16, 32, 64, 128]
ROUNDS       = 400   # FL rounds
N_CLIENTS    = 9     # simulated clients (matches FL experiments)
LOCAL_EPOCHS = 1     # local epochs per client per round (matches fit_*.py)
NUM_CLASSES  = 10
INPUT_SHAPE  = (32, 32, 3)
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
_LOCAL_TAR   = Path(__file__).resolve().parent / "cifar-10-python.tar.gz"


def _load_from_tar(tar_path: Path):
    """Load CIFAR-10 directly from the official tar.gz without extracting to disk."""
    x_train_parts, y_train_parts = [], []
    with tarfile.open(tar_path, "r:gz") as tar:
        for i in range(1, 6):
            f = tar.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            batch = pickle.load(f, encoding="bytes")
            x_train_parts.append(batch[b"data"])
            y_train_parts.extend(batch[b"labels"])
        f = tar.extractfile("cifar-10-batches-py/test_batch")
        batch = pickle.load(f, encoding="bytes")
        x_test = np.array(batch[b"data"])
        y_test = np.array(batch[b"labels"])

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.array(y_train_parts)

    # Raw layout is (N, 3072) in CHW order → convert to (N, 32, 32, 3) HWC
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def load_cifar10():
    if _LOCAL_TAR.exists():
        print(f"  Using local tar.gz: {_LOCAL_TAR}")
        return _load_from_tar(_LOCAL_TAR)
    print("  Local tar.gz not found — downloading via tf.keras ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def _partition_iid(x, y, n_clients: int, seed: int = 42):
    """Split (x, y) into n_clients equal IID shards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    size = len(x) // n_clients
    return [
        (x[idx[i * size:(i + 1) * size]], y[idx[i * size:(i + 1) * size]])
        for i in range(n_clients)
    ]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def train_one(batch_size: int, x_train, y_train, x_test, y_test,
              rounds: int = ROUNDS) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  batch_size={batch_size}  |  rounds={rounds}"
          f"  |  clients={N_CLIENTS}  |  local_epochs={LOCAL_EPOCHS}", flush=True)
    print(f"  log → {RESULTS_DIR / f'batch_{batch_size}.csv'}", flush=True)
    print(f"{'='*60}", flush=True)

    model = create_mobilenetv2(INPUT_SHAPE, NUM_CLASSES)
    print(f"  model params: {model.count_params():,}", flush=True)

    # IID partition — fixed seed for reproducibility
    client_data = _partition_iid(x_train, y_train, N_CLIENTS)
    print(f"  shard size: {len(client_data[0][0])} samples/client", flush=True)

    global_weights = model.get_weights()

    log_path = RESULTS_DIR / f"batch_{batch_size}.csv"
    best_val_acc = 0.0
    best_round   = 0
    final_val_acc = 0.0
    t_total = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["round", "train_loss", "train_acc", "val_loss", "val_acc", "round_time_s"]
        )

        for rnd in range(1, rounds + 1):
            t0 = time.perf_counter()

            # ── Local training (each client) ──────────────────────────────
            aggregated = [np.zeros_like(w) for w in global_weights]
            client_losses, client_accs = [], []
            for cx, cy in client_data:
                model.set_weights(global_weights)
                hist = model.fit(
                    cx, cy,
                    batch_size=batch_size,
                    epochs=LOCAL_EPOCHS,
                    verbose=0,
                )
                for i, w in enumerate(model.get_weights()):
                    aggregated[i] += w
                client_losses.append(hist.history["loss"][-1])
                client_accs.append(hist.history["accuracy"][-1])

            # ── FedAvg aggregation ────────────────────────────────────────
            global_weights = [w / N_CLIENTS for w in aggregated]
            del aggregated

            # ── Global evaluation ─────────────────────────────────────────
            model.set_weights(global_weights)
            vl, va = model.evaluate(x_test, y_test, verbose=0)

            elapsed = time.perf_counter() - t0
            t_total += elapsed

            tl = float(np.mean(client_losses))
            ta = float(np.mean(client_accs))
            final_val_acc = va

            new_best = va > best_val_acc
            if new_best:
                best_val_acc = va
                best_round   = rnd

            writer.writerow(
                [rnd, f"{tl:.6f}", f"{ta:.6f}", f"{vl:.6f}", f"{va:.6f}", f"{elapsed:.2f}"]
            )
            f.flush()

            remaining = (rounds - rnd) * (t_total / rnd)
            marker = " *" if new_best else ""
            print(
                f"  [{rnd:3d}/{rounds}]  "
                f"loss={tl:.4f}  acc={ta:.4f}  "
                f"val_loss={vl:.4f}  val_acc={va:.4f}  "
                f"{elapsed:.1f}s/rnd  eta={_fmt_time(remaining)}"
                f"{marker}",
                flush=True,
            )

            if rnd % 50 == 0:
                gc.collect()

    print(
        f"\n  [done] total={_fmt_time(t_total)}  "
        f"best_val_acc={best_val_acc:.4f} @ round {best_round}",
        flush=True,
    )
    return {
        "batch_size":    batch_size,
        "best_val_acc":  best_val_acc,
        "best_round":    best_round,
        "final_val_acc": final_val_acc,
        "total_time_s":  t_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description=f"MobileNetV2 CIFAR-10 FL benchmark — FedAvg, {ROUNDS} rounds"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=f"Run only this batch size. Default: all {BATCH_SIZES}",
    )
    parser.add_argument(
        "--rounds", type=int, default=ROUNDS, metavar="N",
        help=f"Number of FL rounds (default: {ROUNDS})",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(f"  train: {x_train.shape}  test: {x_test.shape}")
    print(f"  GPU visible: {tf.config.list_physical_devices('GPU')}")

    batch_sizes = [args.batch_size] if args.batch_size else BATCH_SIZES

    summary_path = RESULTS_DIR / "summary.csv"
    write_header = not summary_path.exists()

    with open(summary_path, "a", newline="") as sf:
        sw = csv.writer(sf)
        if write_header:
            sw.writerow(
                ["batch_size", "best_val_acc", "best_round", "final_val_acc", "total_time_s"]
            )

        for bs in batch_sizes:
            result = train_one(bs, x_train, y_train, x_test, y_test, rounds=args.rounds)
            sw.writerow([
                result["batch_size"],
                f"{result['best_val_acc']:.6f}",
                result["best_round"],
                f"{result['final_val_acc']:.6f}",
                f"{result['total_time_s']:.1f}",
            ])
            sf.flush()

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
