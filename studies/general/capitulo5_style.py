"""
Shared matplotlib style for the dissertation Chapter 5 figures.

Extracted from the reference "estilo acadêmico Cap. 4" script (large serif
fonts tuned for full-width \\includegraphics figures): font.family serif,
big title/label/tick/legend sizes, dotted grid, dpi=300.

Two variants:
  CAP5_RCPARAMS         — full size (title 30 / label 28 / tick 24 /
                           legend 22), used by every wide-panel figure.
  CAP5_RCPARAMS_COMPACT — same family/grid/axisbelow/dpi, but with the small
                           sizes already tuned for the "faixa" (heatmap-strip)
                           figures (studies 20/22), used for their v1 variant.

Usage (same importlib pattern as solution_style.py / load_data.py in this
package, since these scripts are not an installable module):

    spec = importlib.util.spec_from_file_location("capitulo5_style", HERE / "capitulo5_style.py")
    cap5_style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap5_style)
    cap5_style.apply_style()
"""

import matplotlib.pyplot as plt

CAP5_RCPARAMS = {
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 30,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Same family/grid/dpi conventions, small sizes for the compact heatmap-strip
# figures (studies 20/22, variant v1) that keep today's footprint.
CAP5_RCPARAMS_COMPACT = {
    **CAP5_RCPARAMS,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def apply_style(compact: bool = False) -> None:
    """Apply the Chapter 5 rcParams globally (affects every figure created after this call)."""
    plt.rcParams.update(CAP5_RCPARAMS_COMPACT if compact else CAP5_RCPARAMS)
