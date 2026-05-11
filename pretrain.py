#!/usr/bin/env python3
"""Pre-trains a model on a source dataset and saves the backbone weights.

The saved weights can be used as a starting point for FL experiments by
setting the PRETRAINED_WEIGHTS env var when launching clients.  The
classification head is named 'classifier' in the model and will be
automatically skipped when the downstream task has a different number of
classes (via by_name=True + skip_mismatch=True in Keras).

Saved files
-----------
  logs/models/pretrained/<model>/<dataset>/pretrained.weights.h5
  logs/models/pretrained/<model>/<dataset>/metadata.json

Usage examples
--------------
  # ViT-Tiny on CIFAR-100 (default)
  python3 pretrain.py

  # Explicit args
  python3 pretrain.py --model vit_tiny --dataset cifar100 --epochs 100

  # Quick smoke-test
  python3 pretrain.py --epochs 3 --batch-size 256
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Allow importing from client/ without installing as a package
sys.path.insert(0, str(Path(__file__).parent / "client"))
from models import (
    create_vit_tiny,
    create_resnet20,
    create_mobilenetv2,
    create_lenet5,
    create_dnn_tiny,
)

# ── Available datasets ────────────────────────────────────────────

DATASET_INFO = {
    "cifar100":      {"num_classes": 100, "input_shape": (32, 32, 3)},
    "cifar10":       {"num_classes": 10,  "input_shape": (32, 32, 3)},
    "fashion_mnist": {"num_classes": 10,  "input_shape": (28, 28, 1)},
    "mnist":         {"num_classes": 10,  "input_shape": (28, 28, 1)},
}

MODEL_FACTORIES = {
    "vit_tiny":    create_vit_tiny,
    "resnet20":    create_resnet20,
    "mobilenetv2": create_mobilenetv2,
    "lenet5":      create_lenet5,
    "dnn_tiny":    create_dnn_tiny,
}


# ── Data loading ──────────────────────────────────────────────────

def load_dataset(name: str):
    loaders = {
        "cifar100":      tf.keras.datasets.cifar100.load_data,
        "cifar10":       tf.keras.datasets.cifar10.load_data,
        "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
        "mnist":         tf.keras.datasets.mnist.load_data,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders)}")

    (x_train, y_train), (x_test, y_test) = loaders[name]()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    # Add channel dim for grayscale
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
        x_test  = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test


# ── Augmentation pipeline ─────────────────────────────────────────

def build_augmentation(input_shape):
    h, w = input_shape[:2]
    # Random horizontal flip + random translation (simulates random crop with padding).
    # No brightness/contrast: those can push pixel values outside [0,1] and cause
    # NaN in the attention softmax when training ViTs from scratch.
    pad = max(2, h // 8)
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(pad / h, pad / w, fill_mode="reflect"),
    ], name="augmentation")


def make_dataset(x, y, batch_size: int, augment: bool, augmentation_fn=None):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if augment and augmentation_fn is not None:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        ds = ds.map(
            lambda imgs, labels: (augmentation_fn(imgs, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


# ── LR schedule ───────────────────────────────────────────────────

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps`, then cosine decay to near-zero."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step    = tf.cast(step, tf.float32)
        warmup  = step / self.warmup_steps * self.peak_lr
        cosine  = self.peak_lr * 0.5 * (
            1.0 + tf.cos(
                np.pi * (step - self.warmup_steps) /
                tf.maximum(self.total_steps - self.warmup_steps, 1.0)
            )
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps}


# ── Training ──────────────────────────────────────────────────────

def pretrain(
    model_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    out_dir: Path,
):
    info = DATASET_INFO[dataset_name]
    num_classes  = info["num_classes"]
    input_shape  = info["input_shape"]

    print(f"\n{'='*60}")
    print(f"  Pre-training : {model_name}")
    print(f"  Dataset      : {dataset_name}  ({num_classes} classes, {input_shape})")
    print(f"  Epochs       : {epochs}  |  batch={batch_size}  |  lr={lr}")
    print(f"  Output       : {out_dir}")
    print(f"{'='*60}\n")

    # ── Build model ──────────────────────────────────────────────
    factory = MODEL_FACTORIES[model_name]
    model = factory(input_shape, num_classes)
    model.summary(line_length=80)

    # Re-compile with linear warmup + cosine decay and gradient clipping.
    # clipnorm=1.0 prevents the gradient explosions that cause NaN loss in ViT
    # training from scratch (attention softmax is sensitive to large weight updates).
    n_train         = 50_000
    steps_per_epoch = max(1, n_train // batch_size)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = max(1, 5 * steps_per_epoch)   # 5-epoch linear warmup

    schedule = WarmupCosineDecay(
        peak_lr=lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(schedule, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Load data ────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_dataset(dataset_name)
    augmentation = build_augmentation(input_shape)
    train_ds = make_dataset(x_train, y_train, batch_size, augment=True,  augmentation_fn=augmentation)
    val_ds   = make_dataset(x_test,  y_test,  batch_size, augment=False)

    # ── Train ────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
    )
    elapsed = time.time() - t0

    # ── Save weights ─────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "pretrained.weights.h5"
    model.save_weights(str(weights_path))
    print(f"\nWeights saved → {weights_path}")

    # ── Save metadata ─────────────────────────────────────────────
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc   = float(history.history["val_accuracy"][-1])
    metadata = {
        "model":           model_name,
        "pretrain_dataset": dataset_name,
        "num_classes":     num_classes,
        "input_shape":     list(input_shape),
        "epochs":          epochs,
        "batch_size":      batch_size,
        "lr":              lr,
        "train_acc":       round(final_train_acc, 4),
        "val_acc":         round(final_val_acc, 4),
        "training_time_s": round(elapsed, 1),
        "weights_file":    "pretrained.weights.h5",
        "transfer_note": (
            "Load with model.load_weights(path, by_name=True, skip_mismatch=True). "
            "The 'classifier' head layer will be skipped when num_classes differs."
        ),
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved → {meta_path}")

    print(f"\nFinal train acc: {final_train_acc:.4f}  |  val acc: {final_val_acc:.4f}")
    print(f"Training time  : {elapsed:.0f}s")
    print(f"\nTo use in FL experiments, set:")
    print(f"  PRETRAINED_WEIGHTS={weights_path}")

    return weights_path


# ── CLI ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      default="vit_tiny",  choices=list(MODEL_FACTORIES), help="Model architecture")
    p.add_argument("--dataset",    default="cifar100",  choices=list(DATASET_INFO),    help="Pre-training dataset")
    p.add_argument("--epochs",     default=100, type=int,   help="Number of training epochs")
    p.add_argument("--batch-size", default=128, type=int,   help="Batch size")
    p.add_argument("--lr",         default=3e-4, type=float, help="Initial learning rate (cosine decay)")
    p.add_argument("--output",     default=None,             help="Custom output directory (default: logs/models/pretrained/<model>/<dataset>)")
    p.add_argument("--gpu",        default="0",              help="GPU index(es) to use, e.g. '0' or '0,1'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_dir = Path(args.output) if args.output else \
              Path("logs/models/pretrained") / args.model / args.dataset

    pretrain(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=out_dir,
    )
