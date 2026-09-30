# Analysis Studies — Per-Script Reference

Analysis scripts live in `studies/sparsification_percentage/`. See CLAUDE.md for the run commands and `load_data.py` API summary.

## Shared utility — `load_data.py`

```python
from load_data import load_all, load_masks, last_round, rounds_to_threshold, auc_accuracy, discover_dims
dfs = load_all()          # reads logs/sparsification_percentage/
dfs = load_all("/custom/path")
# Keys: 'train', 'evaluate', 'server_eval', 'server_agg'
# percentage=None means plaintext/no-packing baseline (was -1 in folder name)

masks = load_masks()      # reads mask.csv from every experiment folder (NOT included in load_all)
masks = load_masks("/custom/path")
# Columns: experiment_name, round, cid, solution, model, dataset, distribution,
#          n_packs, packs_sent, selected_packs, selected (list[int])
# Meta injected from folder name: model, dataset, solution, niid, percentage, seed, rounds, clients
# 'selected' = list[int] parsed from selected_packs (';'-separated pack indices)
```

Key details:
- Folder regex strips timestamp; `_NVNL_` uppercase is normalized to `nvnl`; erroneous `nvnl + percentage=-1` folders are skipped.
- `groupby` calls use `dropna=False` throughout to preserve plaintext rows (where `percentage=None/NaN`).
- Plaintext appears as a "baseline" column in all comparison plots, separated by a dashed vertical line.

## Script 10 — Full model coverage rounds

Answers: how many rounds until the server has received every model pack at least once?

| Solution | Formula | ~Rounds at p=0.1 | ~Rounds at p=0.9 |
|----------|---------|-----------------|-----------------|
| nvnl | `ceil(n_packs / packs_sent)` | 14 | 2 |
| sw | `ceil(n_packs / ceil(packs_sent × 0.5))` | 28 | 3 |
| rw | `(n_packs/packs_sent) × ln(n_packs)` | ~47 | ~4 |
| fedphe | top-k fixed → same packs every round | NaN | NaN |
| topkrr | phase 1 (top-k): NaN; phase 2 (RR): `ceil(n_packs / packs_sent)` after `RR_START_ROUND` | NaN / 14 | NaN / 2 |
| srr | `n // k` rounds when `n % k == 0`; partial coverage otherwise | ~10 | ~2 |
| plaintext | 1 | 1 | 1 |

SW costs roughly 2× more rounds than NVNL due to 50% stride overlap.

## Script 11 — Data transferred vs model coverage (NVNL vs SW)

Answers: given a fixed byte or round budget, how much of the model has each strategy exposed?

Key insight: NVNL and SW send the same bytes per round, but SW overlaps 50% with the previous window → SW covers only half as much new territory per MB.

| Metric | NVNL | SW |
|--------|------|----|
| New coverage/round | `packs_sent / n_packs` | `ceil(packs_sent×0.5) / n_packs` |
| MB/round (raw float32) | `packs_sent × pack_size × 4 / 1e6` | same |
| Marginal cost (MB per 1% cov.) | `n_params × 4 / (1e6 × 100)` | 2× NVNL |

Plots: (A) coverage % vs round, (B) coverage % vs cumulative MB — the key efficiency plot, (C) coverage at fixed budgets (5,10,20,50 rounds), (D) marginal cost bar chart.

Bytes estimate uses raw float32 pre-encryption (`pack_size × packs_sent × 4`) since `payload_bytes` in packing solutions only records mask metadata.

## Script 12 — Cumulative bytes vs accuracy (all techniques, per percentage)

Answers: how much data does each technique need to transmit to reach 80% accuracy?

**Layout**: one figure per percentage. Each figure row = (model, niid).
- **Left panel**: X = cumulative upload MB, Y-left = accuracy %, Y-right = round (approx.).
  Curves stop at 80% (★ marker). Round annotations along each curve.
- **Right panel**: horizontal bar chart — rounds and MB cost to reach 80% per technique.

**Bytes metric**: raw float32 model data (pre-encryption).
- Packing: `packs_sent × pack_size × 4 bytes` per client per round
- Plaintext/other: `n_params × 4 bytes` per client per round

**Output**: `12_bytes_acc_p{pct}.{png,pdf}` (one per percentage), `12_bytes_acc_plaintext.{png,pdf}`.
**Table**: `12_bytes_to_80pct.{csv,tex}`.

## Script 13 — Communication time overhead vs accuracy

Answers: how much time do HE/communication operations consume to reach 80% accuracy?

**Communication overhead** = `pack_time_s + mask_select_time_s + encrypt_time_s + serialize_time_s + decrypt_time_s` (training time excluded — focus is on the privacy/communication cost).

**Layout**: one figure per percentage. Each figure row = (model, niid).
- **Left panel** (2/3 width): X = cumulative communication overhead (s), Y-left = accuracy %, Y-right = round (approx.). Round annotations every ~4 points. ★ marker when 80% is reached.
- **Right panel** (1/3 width): stacked horizontal bars per technique — overhead split by `Pack+Mask | Encrypt | Serialize | Decrypt` (median per round).

Key insight: complements study 12 (bytes axis) with a time axis, revealing how much compute each technique spends on privacy operations per unit of accuracy gained.

**Output**: `13_overhead_acc_p{pct}.{png,pdf}` (one per percentage), `13_overhead_acc_plaintext.{png,pdf}`.
**Table**: `13_overhead_summary.{csv,tex}` — columns: model, solution, percentage, niid, rounds_to_80pct, seconds_to_80pct, comm_overhead_s_median, final_acc.

## Script 14 — Cumulative neurons transmitted vs accuracy

Answers: how many model parameters does each technique need to transmit to reach 80% accuracy?

**Neurons metric** (parameters actually sent per client per round):
- Packing (`nvnl`, `sw`, `rw`, `fedphe`, `topkrr`, `srr`): `packs_sent × pack_size`
- Other (`plaintext`, `ckks`, `bfv`): `n_params`

**Layout**: one figure per percentage. Each figure row = (model, niid).
- **Left panel** (2/3 width): X = cumulative neurons transmitted (M), Y-left = accuracy %, Y-right = round (approx.). Round annotations every ~4 points. ★ marker when 80% is reached.
- **Right panel** (1/3 width): horizontal bars per technique — median neurons/round, labelled in M (e.g. `73.7k`, `12.5M`).

Key insight: complements study 12 (bytes axis) and 13 (time axis) with a parameter-count axis, revealing how much of the model each technique exposes per unit of accuracy gained — independent of encoding size or hardware speed.

**Output**: `14_neurons_acc_p{pct}.{png,pdf}` (one per percentage), `14_neurons_acc_plaintext.{png,pdf}`.
**Table**: `14_neurons_summary.{csv,tex}` — columns: model, solution, percentage, niid, rounds_to_80pct, neurons_to_80pct_M, neurons_per_round_median, final_acc.

## Script 15 — Cumulative bytes vs accuracy (techniques only, max-acc target)

Like script 12, but **techniques only** (plaintext fully excluded — no baseline curve, no plaintext figure) and targeting the **maximum accuracy reached by the best technique** of each panel instead of a fixed 80%.

**Layout**: one figure per percentage. Each figure row = (model, niid).
- **Left panel** (3/4 width): X = cumulative upload MB (raw float32 pre-encryption), Y-left = accuracy %, Y-right = round (approx.). Dashed reference line = best technique's max accuracy in the panel (labelled `best: <sol> X%`). Each curve is clipped at its own peak-accuracy round with a ★ marker.
- **Right panel** (1/4 width): horizontal bars per technique — max accuracy reached, sorted best-on-top, labelled `acc% · R{peak round} · MB`.

Helpers (replace the 80%-target helpers of scripts 12/13): `_peak_row`, `_max_acc`, `_peak_round`, `_cost_at_peak`, `_clip_at_peak`.

**Output**: `15_bytes_maxacc_p{pct}.{png,pdf}` (one per percentage).
**Table**: `15_bytes_maxacc.{csv,tex}` — columns: model, solution, percentage, niid, max_acc, round_at_max, mb_at_max, final_acc (best `max_acc` per model×niid×pct bolded in LaTeX).

## Script 16 — Communication overhead vs accuracy (techniques only, max-acc target)

Same max-acc design as script 15, but X = cumulative communication overhead (s) = `pack + mask_select + encrypt + serialize + decrypt` (as in script 13). Techniques only; right-panel bars labelled `acc% · R{peak round} · overhead s`.

**Caveat (decrypt)**: `decrypt_time_s` bundles deserialize + decrypt + re-pack of the full model + reshape into one value and is backend-dependent (ckks_python_simplify ≫ tenseal), so it dominates the overhead and is **not** directly comparable to `encrypt_time_s` (encryption only). Logs produced before the timing fix also counted the snapshot/checkpoint disk writes inside this window (now excluded). See the `decrypt_time_s` scope note in `docs/log_schema.md`.

**Output**: `16_overhead_maxacc_p{pct}.{png,pdf}` (one per percentage).
**Table**: `16_overhead_maxacc.{csv,tex}` — columns: model, solution, percentage, niid, max_acc, round_at_max, seconds_at_max, final_acc.

## Script 17 — Observed unique-pack coverage per round (sem duplicação)

Answers: as rounds progress, what fraction of the model's packs has the server received **at least once** (within the current cycle)?

**Metric**: server-side union of all clients' pack indices per round accumulated into a running set `covered`. Coverage = `|covered| / n_packs × 100%`. Resets when `|covered| >= n_packs` → sawtooth pattern restarts. Duplicate packs re-sent in different rounds do **not** advance the counter (only genuinely new packs do).

| Technique | Behaviour |
|---|---|
| nvnl | rises `packs_sent/n_packs` per round; resets at 100% (no overlap) |
| sw | ~50% stride overlap → new-pack contribution per round ≈ half of nvnl → longer cycle |
| rw | irregular slope (random); cycle length ≈ coupon-collector |
| srr | same as nvnl (spread, no intra-cycle overlap) |
| topkrr | top-k phase: saturates below 100% (never resets); RR phase: full cycle |
| fedphe | top-k fixed: curve saturates at `packs_sent/n_packs%`, never resets |

Source: `mask.csv` read via `load_data.load_masks`. Complements study 10 (theoretical rounds) with **observed** coverage.

**Output**: `output/figures/17_coverage_unique/{model}_{technique}.{png,pdf}` — one figure per (model, technique); each figure has subplots for iid/niid, lines = sparsification percentages (viridis colormap).
**Table**: `17_coverage_unique_summary.{csv,tex}` — columns: model, solution, niid, percentage, n_packs, packs_per_round, rounds_per_cycle (NaN if never completes), n_cycles, ever_full.

## Script 18 — Observed volume-based coverage per round (com duplicação)

Same sawtooth design as script 17, but every pack transmission counts (including re-sent packs).

**Metric**: per round, the server-side union size (intra-round deduplication across clients) is added to `vol`. Coverage = `vol / n_packs × 100%`. Resets when `vol >= n_packs`. Unlike study 17, duplicate packs re-sent across rounds **do** inflate the counter.

Key contrast with study 17:
- **nvnl/srr**: identical to study 17 (no overlap → no difference).
- **sw**: per-round volume ≈ nvnl (union size same); BUT same packs repeat across rounds → SW cycle length ≈ nvnl in study 18, while study 17 shows SW takes ~2× nvnl rounds.
- **fedphe**: here becomes a sawtooth (volume resets every `ceil(n_packs/packs_sent)` rounds), evidencing the retransmission cost of always sending the same top-k packs.

**Output**: `output/figures/18_coverage_volume/{model}_{technique}.{png,pdf}` — one figure per (model, technique); each figure has subplots for iid/niid, lines = sparsification percentages (viridis colormap).
**Table**: `18_coverage_volume_summary.{csv,tex}` — columns: model, solution, niid, percentage, n_packs, packs_per_round, rounds_per_cycle, n_cycles.

## Script 19 — Coverage vs accuracy per round (dual Y axes)

Same layout as study 17 (one figure per model×technique, grid rows=percentage × cols=niid, x=round),
but each subplot has **two Y axes**:
- **Y left (blue)** = unique-pack coverage % — sawtooth identical to study 17.
- **Y right (red)** = model accuracy % per round (`server_eval.csv` → `mean_acc × 100`, averaged across seeds).

Reveals how accuracy evolves as the server progressively covers the model: whether accuracy gains
correlate with coverage cycles or proceed independently of the packing pattern. Only for the
unique-coverage metric (study 17); volume metric (study 18) is not duplicated here.

**Output**: `output/figures/19_coverage_accuracy/{model}_{technique}.{png,pdf}`.
**Table**: `19_coverage_accuracy_summary.{csv,tex}` — columns: model, solution, niid, percentage, rounds_per_cycle, final_acc, max_acc.

## Script 20 — Pack transmission frequency

Answers: over all rounds, how often was each pack index transmitted?

**Metric**: server-side union of selected pack indices per round (same convention as studies 17–19). `send_pct[pack_k]` = % of rounds where pack k appeared in the union. Horizontal bar chart per subplot.

| Technique | Expected pattern |
|---|---|
| nvnl | nearly uniform bars at ~pct% |
| sw | bars at ~2×pct% (each pack covered by two overlapping windows) |
| rw | non-uniform; high-magnitude packs sent more often |
| fedphe | bimodal: ~pct% packs at 100%, rest at 0% |

Source: `mask.csv` only (no snapshot needed). **Output**: `output/figures/20_pack_frequency/{model}_{technique}.{png,pdf}`. **Table**: `20_pack_frequency_summary.{csv,tex}`.

## Script 21 — Layer transmission frequency

Same as script 20, but per model **layer** instead of per pack. A layer is "sent" in a round if any of its covering packs is in the server-union. Layer→pack mapping uses flat-offset arithmetic (`pack_size = poly_modulus_degree//2`); shapes come from one model snapshot (`logs/.../model/round_N.npz`). Bar annotations show the number of packs covering each layer ("N pk"). Small bias layers sharing a pack with adjacent weight layers always show near-100%.

Source: `mask.csv` + one snapshot per (model, solution) combo. **Output**: `output/figures/21_layer_frequency/{model}_{technique}.{png,pdf}`. **Table**: `21_layer_frequency_summary.{csv,tex}` — columns: model, solution, niid, percentage, layer_idx, layer_shape, n_packs_covering, send_count, send_pct.

## Duplicate detection

`studies/sparsification_percentage/move_duplicates.sh` strips timestamps from folder names and moves older runs (same config) to `logs/sparsification_percentage/duplicates/`.
