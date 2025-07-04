# -*- coding: utf-8 -*-
"""crvq_final.py – orchestrator implementing CRVQ Algorithm 1 for all Linear layers.
     Requires modules.py utilities.  Fully debugged (safe indices, proper VQ encode/decode).
"""
from __future__ import annotations
import logging, numpy as np, torch
from modules import (
    partition_to_vectors, reassemble_from_vectors,
    vq_encode, vq_decode,
    importance_metric, reorder_channels, restore_order,
    beam_search_iterative, compression_ratio)

log = logging.getLogger("crvq")
log.addHandler(logging.NullHandler())

class CRVQ:
    """Channel‑Relaxed Vector Quantisation (Algorithm 1)."""

    def __init__(self, d=8, e=8, m=4, lam=0.05, eps=1e-3):
        self.d, self.e, self.m, self.lam, self.eps = d, e, m, lam, eps
        self.state = {}

    # ------------------------------------------------------------------ #
    def quantise_layer(self, name: str, W: torch.Tensor,
                       h_diag: np.ndarray | None = None) -> torch.Tensor:
        """Quantise a 2‑D weight matrix.  Returns quantised weights (same dtype)."""
        Wnp = W.detach().cpu().numpy()
        O, I = Wnp.shape

        # ───── Step 3  Pre‑quant (coarse VQ) ────────────────────────────
        V, pad = partition_to_vectors(Wnp, self.d)
        C_coarse, B_coarse = vq_encode(V, 2 ** min(6, self.e))
        Vq_coarse = vq_decode(C_coarse, B_coarse)
        Wq_coarse = reassemble_from_vectors(Vq_coarse, O, I, pad, self.d)

        # ───── Step 4  Importance  (max error × Hessian‑inv diag) ──────
        imp = importance_metric(Wnp, Wq_coarse, h_diag)

        # ───── Step 5  Reorder channels by importance ──────────────────
        W_sorted, perm = reorder_channels(Wnp, imp)

        # ───── Steps 6‑7  Base codebook VQ ─────────────────────────────
        V_sorted, pad = partition_to_vectors(W_sorted, self.d)
        C_base, B_base = vq_encode(V_sorted, 2 ** self.e)
        Vq_base        = vq_decode(C_base, B_base)
        C_list, B_list = [C_base], [B_base]

        # ───── Steps 8‑13  Extended codebooks on λ‑fraction ────────────
        v_per_row = I // self.d
        if v_per_row == 0:
            log.warning(f"Layer {name}: in_features({I}) < d({self.d}); skipping CRVQ on this layer.")
            return W  # return original tensor unchanged

        n_vectors = V_sorted.shape[0]
        crit_rows = max(1, int(self.lam * O))
        crit_vecs = min(crit_rows * v_per_row, n_vectors)
        idx_crit  = np.arange(crit_vecs)

        if crit_vecs > 0:
            for step in range(1, self.m):
                resid = V_sorted[idx_crit] - _recon(C_list, B_list)[idx_crit]
                if np.mean(resid ** 2) < self.eps:
                    break
                C_ext, B_ext = vq_encode(resid, 2 ** self.e, random_state=step)
                Vq_ext       = vq_decode(C_ext, B_ext)

                # store codes only for critical vectors, zeros elsewhere
                fill_codes = np.zeros_like(B_base)
                fill_codes[idx_crit] = B_ext
                C_list.append(C_ext)
                B_list.append(fill_codes)

        # ───── Step 16  Beam‑search refinement ─────────────────────────
        B_list = beam_search_iterative(V_sorted, C_list, B_list,
                                       beam=4, iters=4)

        # ───── Reconstruction & restore order ──────────────────────────
        V_final   = _recon(C_list, B_list)
        Wq_sorted = reassemble_from_vectors(V_final, O, I, pad, self.d)
        W_quant   = restore_order(Wq_sorted, perm)

        cr, bpp = compression_ratio(O, I, self.d, self.e, len(C_list), self.lam)
        log.info(f"Layer {name}: compression ≈{cr:.1f}×  ({bpp:.2f} bits/param)")

        self.state[name] = dict(codebooks=C_list, codes=B_list, perm=perm)
        return torch.tensor(W_quant, dtype=W.dtype)

# ----------------------------------------------------------------------
# helper (vector reconstruction)
# ----------------------------------------------------------------------

def _recon(C_list, B_list):
    v = vq_decode(C_list[0], B_list[0])
    for C, B in zip(C_list[1:], B_list[1:]):
        v += vq_decode(C, B)
    return v
