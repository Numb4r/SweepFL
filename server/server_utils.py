import numpy as np
import pickle as pk
from encryption.ckks_backend import get_backend, TenSEALBackend, CKKSSimplifyBackend


def _backend(server_self=None):
    """Return the active backend. Uses server's key_config when available."""
    if server_self is not None:
        name = getattr(server_self, 'key_config', {}).get("ckks_backend")
        return get_backend(name)
    return get_backend()


# ── Pack helpers ───────────────────────────────────────────────────────────────

def align_packs(self, packed_parameters, mask):
    """Deserialize packed bytes into ciphertext objects, aligned to the mask."""
    backend     = _backend(self)
    aligned_pack = [0] * len(mask)
    for idx, m in enumerate(mask):
        if m == 1:
            aligned_pack[idx] = backend.deserialize(self.context, packed_parameters.pop(0))
    return aligned_pack


def aggregate_packs_and_masks(aligned_pack, weighted_mask, aggregated_packs, aggregated_mask,
                               backend=None, ctx=None):
    """Accumulate weighted encrypted packs across clients.

    backend/ctx: optional — if None, falls back to TenSEAL for backward compat
    with existing test code that passes raw TenSEAL vectors.
    """
    if backend is None:
        backend = TenSEALBackend()

    is_empty = len(aggregated_packs) == 0

    for i in range(len(weighted_mask)):
        p = aligned_pack[i]
        w = weighted_mask[i]

        if p == 0:  # inactive pack position
            if is_empty:
                aggregated_packs.append(0)
                aggregated_mask.append(w)
            else:
                aggregated_mask[i] += w
            continue

        weighted_p = backend.mul(ctx, p, w)

        if is_empty:
            aggregated_packs.append(weighted_p)
            aggregated_mask.append(w)
        else:
            aggregated_packs[i] = backend.add(ctx, aggregated_packs[i], weighted_p)
            aggregated_mask[i] += w

    return aggregated_packs, aggregated_mask


def serialize_packs(aggregated_packs, backend=None):
    if backend is None:
        backend = TenSEALBackend()
    return [backend.serialize(pack) for pack in aggregated_packs if pack != 0]


def get_pack_mask_pair(self, results):
    backend      = _backend(self)
    pack_mask_pair = []
    total_examples = 0

    for _, fit_res in results:
        packed_parameters = pk.loads(fit_res.metrics['he'])
        mask              = pk.loads(fit_res.metrics['mask'])
        dataset_size      = int(fit_res.num_examples)

        aligned_pack  = align_packs(self, packed_parameters, mask)
        weighted_mask = np.array(mask) * dataset_size
        total_examples += dataset_size
        pack_mask_pair.append([aligned_pack, weighted_mask])

    return pack_mask_pair, total_examples


def aggregate_packs_and_mask(pack_mask_pair, total_examples, onlysum, backend=None, ctx=None):
    """Weighted homomorphic aggregation across all clients' pack-mask pairs."""
    n                = len(pack_mask_pair[0][1])
    aggregated_packs = [0] * n
    aggregated_masks = [0] * n

    # CKKSSimplifyBackend cannot do fractional scalar mul (single-level CKKS,
    # no rescaling).  Clients pre-divide by num_clients, so pure sum = FedAvg.
    _pure_sum = onlysum or isinstance(backend, CKKSSimplifyBackend)

    for packs, mask in pack_mask_pair:
        mask = mask / total_examples

        for idx_pack, p in enumerate(packs):
            if p == 0:  # inactive position — skip, stays 0
                continue

            if _pure_sum:
                new_term = p
                aggregated_masks[idx_pack] += mask[idx_pack] * total_examples
            else:
                new_term = backend.mul(ctx, p, float(mask[idx_pack])) if backend else p * mask[idx_pack]
                aggregated_masks[idx_pack] += mask[idx_pack]

            if aggregated_packs[idx_pack] == 0:
                aggregated_packs[idx_pack] = new_term
            else:
                if backend:
                    aggregated_packs[idx_pack] = backend.add(ctx, aggregated_packs[idx_pack], new_term)
                else:
                    aggregated_packs[idx_pack] = aggregated_packs[idx_pack] + new_term

    return aggregated_packs, aggregated_masks


# ── Non-packing (single-vector) helpers ───────────────────────────────────────

def get_cyphered_parameters(self, results):
    backend         = _backend(self)
    parameters_list = []
    total_examples  = 0

    for _, fit_res in results:
        if self.homomorphic_type == 'Paillier':
            parameters = pk.loads(fit_res.metrics['he'])
        elif self.homomorphic_type == 'BFV':
            # BFV always uses TenSEAL, regardless of the CKKS backend.
            import tenseal as ts
            parameters = ts.bfv_vector_from(self.context, fit_res.metrics['he'])
        else:
            parameters = backend.deserialize(self.context, fit_res.metrics['he'])

        parameters_list.append((parameters, int(fit_res.num_examples)))
        total_examples += int(fit_res.num_examples)

    return parameters_list, total_examples


def aggregated_cyphered_parameters(self, parameters_list, total_examples):
    backend    = _backend(self)
    agg        = None

    # CKKSSimplifyBackend cannot do fractional scalar mul (single-level CKKS,
    # no rescaling).  Clients pre-divide by num_clients, so pure sum = FedAvg.
    _pure_sum = self.onlysum or isinstance(backend, CKKSSimplifyBackend)

    for parameters, num_examples in parameters_list:
        if _pure_sum:
            if self.homomorphic_type == 'Paillier':
                if agg is None:
                    agg = parameters
                else:
                    agg = (agg * parameters) % pow(self.context.get_n(), 2)
            elif self.homomorphic_type == 'BFV':
                # TenSEAL BFV vectors: plain ciphertext addition.
                agg = parameters if agg is None else agg + parameters
            else:
                agg = parameters if agg is None else backend.add(self.context, agg, parameters)
        else:
            weight    = num_examples / total_examples
            weighted  = backend.mul(self.context, parameters, weight)
            agg       = weighted if agg is None else backend.add(self.context, agg, weighted)

    return agg
