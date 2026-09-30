# CSV Log Schema & Model Compression

## train.csv (41 columns, one row per client per round)

| Group | Columns |
|---|---|
| Identity | `experiment_name`, `round`, `cid` |
| Config | `solution`, `schema`, `model`, `dataset`, `distribution`, `dirichlet_alpha` |
| Accuracy | `train_loss`, `train_acc` |
| Decrypt | `decrypt_time_s`, `decrypt_time_cpu` |
| Train | `train_time_s`, `train_time_cpu` |
| Flatten | `flatten_time_s`, `flatten_time_cpu` |
| Quantize | `quantize_time_s`, `quantize_time_cpu` (BFV / BatchCrypt only) |
| Batch padding | `batch_padding_time_s`, `batch_padding_time_cpu` (BatchCrypt only) |
| Pack | `pack_time_s`, `pack_time_cpu` (packing solutions only) |
| Mask select | `mask_select_time_s`, `mask_select_time_cpu` (packing solutions only) |
| Encrypt | `encrypt_time_s`, `encrypt_time_cpu` |
| Serialize | `serialize_time_s`, `serialize_time_cpu` |
| Payload | `payload_bytes`, `n_params` |
| Pack info | `n_packs`, `packs_sent`, `pack_size`, `sigma_ckks` |
| HE params | `poly_modulus_degree`, `coeff_mod_bits`, `scale_bits`, `plain_modulus` |
| Memory | `ram_rss_mb`, `ram_peak_mb` |

Columns unused by a technique are emitted as `0`. `coeff_mod_bits` uses semicolons as separator (e.g. `60;40;40;60`) to avoid CSV breakage. `scale_bits` stores the exponent (e.g. `40` for `2^40`).

**`decrypt_time_s` scope**: for HE solutions this column bundles deserialize + decryption + re-packing the full model + reshape into one value, so it is **not** directly comparable to `encrypt_time_s` (which is encryption only — `flatten_time_s` / `pack_time_s` / `serialize_time_s` are separate columns). The disk writes (model snapshot + checkpoint) are **excluded** from this timer — `he_*_to_model` stops timing before the `np.savez_compressed` calls. Magnitude is backend-dependent (ckks_python_simplify decryption ≫ tenseal). Note: logs produced before this timing fix still include the disk writes in `decrypt_time_s`.

## evaluate.csv (15 columns)

`experiment_name`, `round`, `cid`, `solution`, `schema`, `model`, `dataset`, `distribution`, `dirichlet_alpha`, `eval_loss`, `eval_acc`, `decrypt_time_s`, `poly_modulus_degree`, `coeff_mod_bits`, `scale_bits`.

## Server logs (in `logs/<LOGFOLDER>/`)

- `server_agg.csv`: `experiment_name`, `round`, `n_clients`, `total_examples`, `selection_time_s`, `aggregation_time_s`
- `server_eval.csv`: `experiment_name`, `round`, `n_clients`, `mean_loss`, `mean_acc`

## Model Compression

Per-round model snapshots are saved as `logs/<LOGFOLDER>/model/round_<N>.npz` using `np.savez_compressed` with float16 weights. This reduces storage by ~54% vs float32. Only client 0 saves snapshots. All rounds are kept (not just the last).

**Save timing**: for HE/packing solutions (`nvnl`, `fedphe`, `sw`, `rw`, `topkrr`, `srr`), saves happen inside `he_packs_to_model` / `he_parameters_to_model` in `client/literature/set_parameters.py`, called from both `fit()` and `evaluate()`. Round N's file reflects the server's aggregated weights sent back for that round. For `plaintext`, the save is at the top of `fit_plaintext()` in `client/literature/plaintext.py`. These `np.savez_compressed` writes run **after** the decrypt timer stops, so they are not counted in `decrypt_time_s`.

**Path convention**: Docker CWD is `/`; logs volume is mounted at `/logs`. Always use `f'logs/{self.log_folder}/model'` — never `f'{self.log_folder}/model'`. The latter writes to ephemeral container storage lost on exit. Both `set_parameters.py` and `plaintext.py` must use this prefix consistently.

Loading saved weights:
```python
data = np.load("round_N.npz")
weights = [data[f'arr_{i}'] for i in range(len(data.files))]  # numeric order, NOT sorted()
```

## Payload bytes caveat (packing solutions)

`payload_bytes` in `train.csv` for packing solutions (`nvnl`, `sw`, `rw`, `fedphe`, `topkrr`, `srr`) records only pack-index / mask metadata (a few hundred bytes), **not** the actual CKKS ciphertext size. For plaintext it correctly records the full model byte count. Analyses that compare byte costs between HE and plaintext must account for this — use rounds as the primary communication metric when comparing packing strategies.
