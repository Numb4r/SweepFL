"""Single source of style (colors, markers, line styles, hatches, labels) for every paper figure."""

# Legend order: full-model references first, then the four selection policies.
ORDER = ["plaintext", "ckks", "bfv", "batchcrypt", "fedphe", "rw", "nvnl", "sw"]

COLORS = {"plaintext": "#1a1a1a", "ckks": "#6b3fa0", "bfv": "#008300", "batchcrypt": "#c2185b",
          "fedphe": "#eb6834", "rw": "#8c8c8c", "nvnl": "#2a78d6", "sw": "#1baf7a"}
MARKERS = {"plaintext": None, "ckks": "s", "bfv": "^", "batchcrypt": "D",
           "fedphe": "v", "rw": "X", "nvnl": "o", "sw": "P"}
LINESTYLES = {"plaintext": "--", "ckks": "--", "bfv": "--", "batchcrypt": "--",
              "fedphe": "-", "rw": "-", "nvnl": "-", "sw": "-"}
HATCHES = {"plaintext": "", "ckks": "//", "bfv": "//", "batchcrypt": "//",
           "fedphe": "", "rw": "", "nvnl": "", "sw": ""}

LABELS = {"plaintext": "FedAvg", "ckks": "CKKS", "bfv": "BFV", "batchcrypt": "BatchCrypt",
          "fedphe": "Top-$k$ (FedPHE)", "rw": "Random", "nvnl": "Round Robin", "sw": "Sliding Window"}

# Model pairs in packet order.
PAIR_ORDER = ["lenet5", "dnn", "vit", "resnet20", "mobilenetv2"]
PAIR_MARKERS = {"lenet5": "o", "dnn": "s", "vit": "^", "resnet20": "D", "mobilenetv2": "v"}
# Ordinal ramp of the Top-k orange: more packets = darker.
FEDPHE_RAMP = {"lenet5": "#f29a63", "dnn": "#e8743a", "vit": "#cc5520",
               "resnet20": "#a13f14", "mobilenetv2": "#6e2a0b"}

TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#1a1a1a", "#52514e", "#e6e6e6"
MARKERSIZE = 3.5
LINEWIDTHS = {"plaintext": 0.9, "ckks": 0.75, "bfv": 0.75, "batchcrypt": 0.75,
              "fedphe": 1.0, "rw": 0.85, "nvnl": 1.1, "sw": 1.1}
DASHES = {"plaintext": (0, (6, 2)), "ckks": (0, (3, 1.5)), "bfv": (0, (3, 1.5)),
          "batchcrypt": (0, (3, 1.5))}
# Full-model references sit behind and thin; Round Robin and Sliding Window on top.
ZORDER = {"plaintext": 2, "ckks": 1, "bfv": 1, "batchcrypt": 1,
          "fedphe": 3, "rw": 2, "nvnl": 5, "sw": 4}
# Staggered first marker so markers of different curves do not stack (per-round curves).
MARKEVERY_OFFSET = {"ckks": 0, "bfv": 5, "batchcrypt": 10, "fedphe": 15, "rw": 20, "nvnl": 25, "sw": 30}
MARKEVERY_STEP = 40   # about five markers on a 200-round curve
# Neutral shading used for annotation bands (not a technique color).
BAND = "#e6e6e6"

# Neutral swatches for the per-round time legend (components are drawn in the technique color).
TIME_SWATCH = {"training": "#d0d0d0", "encryption": "#6e6e6e", "decryption": "#6e6e6e"}


def tint(color, frac=0.35):
    """`color` mixed linearly with white; frac is the white share."""
    from matplotlib.colors import to_rgb
    return tuple((1 - frac) * c + frac for c in to_rgb(color))


def ordered(keys):
    """Keys present in `keys`, in legend order; unknown keys are appended sorted."""
    keys = set(keys)
    return [k for k in ORDER if k in keys] + sorted(keys - set(ORDER))


def line_style(key, rounds=False):
    """rounds=True: sparse staggered markers for per-round curves; False: a marker on every point."""
    markevery = (MARKEVERY_OFFSET.get(key, 0), MARKEVERY_STEP) if rounds else None
    return dict(color=COLORS[key], marker=MARKERS[key], linestyle=DASHES.get(key, LINESTYLES[key]),
                label=LABELS[key], linewidth=LINEWIDTHS[key], markersize=MARKERSIZE,
                markeredgewidth=0.4, markeredgecolor="white", zorder=ZORDER[key],
                markevery=markevery)


def bar_style(key):
    return dict(color=COLORS[key], hatch=HATCHES[key], edgecolor="white", linewidth=0.8,
                label=LABELS[key])


def partition_style(base, partition):
    """non-IID: the technique line with filled markers; IID: dotted, hollow markers, same color."""
    if partition == "iid":
        return {**base, "linestyle": ":", "markerfacecolor": "white", "markeredgecolor": base["color"]}
    return dict(base)


def apply_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def legend(ax, handles=None, labels=None, **kw):
    """Frameless legend, entries sorted by ORDER (by label) when handles are taken from the axes."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
        rank = {LABELS[k]: i for i, k in enumerate(ORDER)}
        pairs = sorted(zip(handles, labels), key=lambda p: rank.get(p[1], len(rank)))
        handles, labels = [p[0] for p in pairs], [p[1] for p in pairs]
    kw.setdefault("frameon", False)
    return ax.legend(handles, labels, **kw)
