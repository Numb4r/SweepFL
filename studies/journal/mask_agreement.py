"""Per-round mask agreement from mask.csv (no rerun).

Usage:
    python3 mask_agreement.py [p]            # table on stdout
    python3 mask_agreement.py [p] --plot     # also writes figures/mask-agreement.pdf

Round 1 is dropped, as in starvation_vs_packets_en.py and paper_numbers.py: it ran
with 2 of the 9 clients, so it enters neither the Jaccard index nor the starved count.
"""
import collections
import csv
import glob
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps  # noqa: E402

_ROOTS = [os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "logs", "sparsification_percentage")]
ROOT = os.environ.get("LOGS") or next((p for p in _ROOTS if os.path.isdir(p)), _ROOTS[0])
OUT = os.path.join(os.environ.get("FIG_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures"), "mask-agreement.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PAIRS = {"lenet5_FASHION_MNIST": "LeNet-5", "dnn_tiny_FASHION_MNIST": "DNN",
         "vit_tiny_tanganke_gtsrb": "ViT-Tiny", "resnet20_CIFAR10": "ResNet-20",
         "mobilenetv2_CIFAR10": "MobileNetV2"}
POL = {"fedphe": "Top-k", "rw": "Random", "sw": "SW", "sweepfl": "RR"}
# Folder -> paper label (author's convention, Alterações.md §0.1).
PARTS = {"niid1": "non-IID", "niid0.1": "IID"}
FIRST_ROUND = 2   # round 1 ran with 2 clients instead of 9


def stats(d):
    rounds = collections.defaultdict(dict)
    P = None
    for r in csv.DictReader(open(d + "/mask.csv")):
        P = int(r["n_packs"])
        sel = r["selected_packs"]
        rounds[int(r["round"])][r["cid"]] = frozenset(map(int, sel.split(";"))) if sel else frozenset()
    sent, part, tot, dist, jac = set(), 0, 0, [], []
    for rnd, m in rounds.items():
        if rnd < FIRST_ROUND:
            continue
        for s in m.values():
            sent |= s
        if len(m) < 2:
            continue
        cnt = collections.Counter(j for s in m.values() for j in s)
        tot += len(cnt)
        part += sum(c < len(m) for c in cnt.values())
        dist.append(len(set(m.values())))
        jac += [len(a & b) / len(a | b) for a, b in itertools.combinations(m.values(), 2)]
    mean = lambda x: sum(x) / len(x)
    return dict(P=P, masks=mean(dist), jaccard=mean(jac), partial=part / tot, starved=1 - len(sent) / P)


def collect(p):
    rows = []
    for pair, pname in PAIRS.items():
        for folder, plabel in PARTS.items():
            for pol, polname in POL.items():
                ds = sorted(glob.glob(f"{ROOT}/{pair}_{pol}_none_*_{folder}_percentage{p}_*"))
                if ds:
                    rows.append(dict(pair=pname, part=plabel, pol=polname, **stats(ds[-1])))
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8,
                         "savefig.dpi": 300})
    topk = ps.COLORS["fedphe"]
    fig, ax = plt.subplots(figsize=(3.45, 2.5), constrained_layout=True)
    ax.axvspan(0.0, 0.7, color=ps.BAND, alpha=0.6, lw=0)
    ax.text(0.27, 57, "masks diverge", fontsize=7, style="italic", color=ps.TEXT_SECONDARY, va="top")
    ax.text(0.72, 57, "masks agree", fontsize=7, style="italic", color=ps.TEXT_SECONDARY, va="top")

    offs = {"LeNet-5": (6, 4), "DNN": (6, 4), "ViT-Tiny": (-22, -14),
            "ResNet-20": (-60, 2), "MobileNetV2": (-32, 8)}
    for r in rows:
        x, y = r["jaccard"], 100 * r["starved"]
        if r["pol"] == "Top-k":
            filled = r["part"] == "non-IID"
            ax.scatter(x, y, s=34, marker=ps.MARKERS["fedphe"], zorder=3, edgecolors=topk, linewidths=1.0,
                       facecolors=topk if filled else "white")
            if filled:
                ax.annotate(r["pair"], (x, y), xytext=offs[r["pair"]], textcoords="offset points", fontsize=7)
    rnd = [r["jaccard"] for r in rows if r["pol"] == "Random"]
    ax.scatter(sum(rnd) / len(rnd), 0, s=34, marker=ps.MARKERS["rw"], color=ps.COLORS["rw"], zorder=5)
    ax.annotate("Random", (sum(rnd) / len(rnd), 0), xytext=(-12, 7), textcoords="offset points", fontsize=7)
    ax.scatter(1.0, 0, s=70, marker=ps.MARKERS["sweepfl"], color=ps.COLORS["sweepfl"], zorder=3, clip_on=False)
    ax.scatter(1.0, 0, s=34, marker=ps.MARKERS["sw"], color=ps.COLORS["sw"], zorder=4, clip_on=False)
    ax.annotate("RR, SW", (1.0, 0), xytext=(-30, 7), textcoords="offset points", fontsize=7)

    ax.set_xlim(0.25, 1.03)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("Mean pairwise Jaccard index of client masks")
    ax.set_ylabel("Starved packets (% of $P$)")
    cue = [plt.Line2D([], [], ls="", marker=ps.MARKERS["fedphe"], color=topk, label="Top-$k$ (FedPHE), non-IID"),
           plt.Line2D([], [], ls="", marker=ps.MARKERS["fedphe"], mec=topk, mfc="white", label="Top-$k$ (FedPHE), IID")]
    ps.apply_axes(ax)
    ps.legend(ax, cue, [h.get_label() for h in cue], loc="upper left", fontsize=6, borderpad=0.3, labelspacing=0.25, handletextpad=0.2)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = collect(args[0] if args else "0.1")
    print("pair | partition | policy | distinct masks/round | mean Jaccard | partial | starved | P")
    for r in rows:
        print(f"{r['pair']} | {r['part']} | {r['pol']} | {r['masks']:.2f} | {r['jaccard']:.3f} | "
              f"{r['partial']:.3f} | {r['starved']:.3f} | {r['P']}")
    if "--plot" in sys.argv:
        plot(rows)
