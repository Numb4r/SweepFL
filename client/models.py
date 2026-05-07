import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compile(model, lr: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def _glorot(seed: int):
    return tf.keras.initializers.GlorotUniform(seed)


# ══════════════════════════════════════════════════════════════════════════════
#  ResNet-20  (CIFAR)   ~270 k params
# ══════════════════════════════════════════════════════════════════════════════

def _residual_block(x, filters: int, stride: int = 1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([x, shortcut]))


def create_resnet20(input_shape, num_classes: int):
    """ResNet-20 for CIFAR — ~270 k parameters."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for _ in range(3):
        x = _residual_block(x, 16)
    x = _residual_block(x, 32, stride=2)
    for _ in range(2):
        x = _residual_block(x, 32)
    x = _residual_block(x, 64, stride=2)
    for _ in range(2):
        x = _residual_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return _compile(models.Model(inputs, outputs))


# ══════════════════════════════════════════════════════════════════════════════
#  DNN-Tiny   ~111 k params on MNIST (28×28)
#  DNN-Base   ~309 k params on MNIST (28×28)
# ══════════════════════════════════════════════════════════════════════════════

def create_dnn_tiny(input_shape: Tuple, num_classes: int, seed: int = 42):
    """DNN-Tiny — 3 hidden layers, ~111 k parameters on MNIST."""
    return _compile(tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(64,  activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(32,  activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(num_classes, activation='softmax'),
    ]))


def create_dnn_base(input_shape: Tuple, num_classes: int, seed: int = 42):
    """DNN-Base — 4 hidden layers, ~309 k parameters on MNIST."""
    return _compile(tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(256, activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(128, activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(64,  activation='relu', kernel_initializer=_glorot(seed)),
        layers.Dense(num_classes, activation='softmax'),
    ]))


def create_dnn(input_shape: Tuple, num_classes: int, seed: int = 42):
    """Alias for create_dnn_base — kept for backward compatibility."""
    return create_dnn_base(input_shape, num_classes, seed)


# ══════════════════════════════════════════════════════════════════════════════
#  LeNet-5   (FashionMNIST)   ~44 k params
# ══════════════════════════════════════════════════════════════════════════════

def create_lenet5(input_shape, num_classes: int):
    """LeNet-5 — ~44 k parameters."""
    return _compile(tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(6,  (5, 5), activation='tanh', padding='same'),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh', padding='valid'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84,  activation='tanh'),
        layers.Dense(num_classes, activation='softmax'),
    ]))


# ══════════════════════════════════════════════════════════════════════════════
#  MobileNetV2   ~460 k params (alpha=0.5, 32×32)
# ══════════════════════════════════════════════════════════════════════════════

def create_mobilenetv2(input_shape, num_classes: int, alpha: float = 0.5):
    """MobileNetV2 with width multiplier alpha.

    Approximate parameter counts (exclude head, no pretrained weights):
      alpha=0.35 → ~250 k   |   alpha=0.5 → ~460 k   |   alpha=1.0 → ~2.3 M

    Handles grayscale / 2-D inputs by converting to 3-channel internally.
    Resizes inputs smaller than 32×32 to 32×32 (MobileNetV2 minimum).
    """
    h = input_shape[0]
    w = input_shape[1]
    c = input_shape[2] if len(input_shape) == 3 else 1
    eff_h, eff_w = max(h, 32), max(w, 32)

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 2-D input → (H, W, 1)
    if len(input_shape) == 2:
        x = layers.Reshape((h, w, 1))(x)

    # Grayscale → RGB by repeating the channel
    if c == 1:
        x = layers.Concatenate(axis=-1)([x, x, x])

    # Upsample if below MobileNetV2's minimum
    if h < 32 or w < 32:
        x = layers.Resizing(eff_h, eff_w)(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(eff_h, eff_w, 3),
        alpha=alpha,
        include_top=False,
        weights=None,
    )
    x = backbone(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return _compile(models.Model(inputs, outputs))


# ══════════════════════════════════════════════════════════════════════════════
#  ViT-Tiny   ~206 k params (patch 4, embed 64, depth 6, MNIST 28×28)
# ══════════════════════════════════════════════════════════════════════════════

class _PosEmbedding(layers.Layer):
    """Adds a learnable (1, num_patches, embed_dim) position encoding."""

    def build(self, input_shape):
        _, n, d = input_shape
        self.pos = self.add_weight(
            name='pos_emb', shape=(1, n, d),
            initializer='zeros', trainable=True,
        )

    def call(self, x):
        return x + self.pos


def _transformer_block(x, embed_dim: int, num_heads: int, mlp_dim: int):
    # Pre-norm multi-head self-attention
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads,
    )(y, y)
    x = layers.Add()([x, y])

    # Pre-norm MLP
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(mlp_dim, activation='gelu')(y)
    y = layers.Dense(embed_dim)(y)
    return layers.Add()([x, y])


def create_vit_tiny(
    input_shape,
    num_classes: int,
    patch_size: int = 4,
    embed_dim: int = 64,
    depth: int = 6,
    num_heads: int = 4,
    mlp_ratio: int = 2,
):
    """ViT-Tiny — patch-based Vision Transformer.

    Default config (patch=4, embed=64, depth=6):
      MNIST  28×28×1  →  49 patches  →  ~206 k params
      CIFAR  32×32×3  →  64 patches  →  ~209 k params

    Uses global-average-pool over patch tokens (no CLS token).
    Accepts grayscale or 2-D inputs by adding a channel dim internally.
    """
    h = input_shape[0]
    w = input_shape[1]
    c = input_shape[2] if len(input_shape) == 3 else 1

    assert h % patch_size == 0 and w % patch_size == 0, (
        f"Input ({h}, {w}) must be divisible by patch_size={patch_size}"
    )

    num_patches = (h // patch_size) * (w // patch_size)
    mlp_dim     = embed_dim * mlp_ratio

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 2-D input → (H, W, C)
    if len(input_shape) == 2:
        x = layers.Reshape((h, w, 1))(x)

    # Patch embedding: Conv2D with stride=patch_size is a learned linear projection
    # Input (B, H, W, C) → (B, H/P, W/P, embed_dim)
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
    )(x)
    x = layers.Reshape((num_patches, embed_dim))(x)   # (B, N, D)

    # Learnable position encoding
    x = _PosEmbedding()(x)

    # Transformer encoder
    for _ in range(depth):
        x = _transformer_block(x, embed_dim, num_heads, mlp_dim)

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Global average pooling over the sequence dimension
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return _compile(models.Model(inputs, outputs), lr=3e-4)


# ══════════════════════════════════════════════════════════════════════════════
#  Simple CNN  (reference / smoke-test)
# ══════════════════════════════════════════════════════════════════════════════

def create_simple_cnn(input_shape, num_classes: int):
    return _compile(tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ]))


# ══════════════════════════════════════════════════════════════════════════════
#  Parameter utilities  (used by FL encryption layer)
# ══════════════════════════════════════════════════════════════════════════════

def flat_parameters(parameters):
    flat = []
    for p in parameters:
        flat.extend(p.flatten())
    return flat


def reshape_parameters(self, decrypted_parameters):
    reshaped = []
    for layer in self.model.get_weights():
        reshaped.append(np.reshape(decrypted_parameters[:layer.size], layer.shape))
        decrypted_parameters = decrypted_parameters[layer.size:]
    return reshaped


def reshape_model(flatted_packs, a, sentinel=0):
    model = []
    if type(a) not in (int, float):
        for item in a:
            m, sentinel = reshape_model(flatted_packs, item, sentinel)
            model.append(m)
    else:
        sentinel += 1
        return flatted_packs[sentinel - 1], sentinel
    return model, sentinel
