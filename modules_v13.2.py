# -*- coding: utf-8 -*-
"""
modules.py - low-level helpers for CRVQ
--------------------------------------
• partition_to_vectors / reassemble_from_vectors
• k-means codebook + VQ
• Hessian-diag capture for importance (activation hooks)
• beam_search_iterative  (multi-step index refinement)
• compression_ratio  (Eq. 8, CRVQ appendix)
"""
from __future__ import annotations
import logging, numpy as np, torch
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, vq as scipy_vq
from scipy.spatial.distance import cdist

log = logging.getLogger("modules")
log.addHandler(logging.NullHandler())

# --------------------------------------------------------------------- #
# Partition helpers
# --------------------------------------------------------------------- #
def partition_to_vectors(mat: np.ndarray, d: int):
    flat = mat.T.reshape(-1)
    pad = (-len(flat)) % d
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, flat.dtype)])
    log.info(f"after partition output: {flat.reshape(-1,d).shape}")
    return flat.reshape(-1, d), pad

def reassemble_from_vectors(vectors: np.ndarray,
                            O: int, I: int, pad: int, d: int):
    log.debug(f"Reassembling vectors to shape ({O}, {I}) with pad={pad} and d={d}")
    flat = vectors.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(I, O).T

def vq_encode(vectors: np.ndarray, k: int,
              iterations: int = 20,
              random_state: int | None = None):
    """
    Return (centroids, codes) where
      • centroids : (k_eff , d)   float32
      • codes     : (N     ,)     int32
    Handles tiny N by setting k_eff = min(k, N) (never > #samples).
    """
    # ensure 2-D
    if vectors.ndim != 2:
        print("error in reshaping of vq_encode")
        vectors = vectors.reshape(-1, vectors.size)
    N, d = map(int, vectors.shape)      # cast to plain ints
    k_eff = min(k, N) if N > 0 else 1

    if random_state is not None:
        np.random.seed(random_state)

    # k-means++ initialisation
    centroids, _ = kmeans2(
        vectors,
        k_eff,
        minit='++',
        iter=iterations)
    labels, _ = scipy_vq(vectors, centroids)
    codes = labels.astype(np.uint8)
    centroids = centroids.astype(np.float32)
    log.debug(f"VQ: {N} vectors, {k_eff} centroids, d={d}, "
              f"centroids shape {centroids.shape}, codes shape {codes.shape}")
    return centroids, codes

def vq_decode(centroids: np.ndarray, codes: np.ndarray):
    """Reconstruct vectors from centroids[ codes ]."""
    return centroids[codes]

# --------------------------------------------------------------------- #
# Importance (Eq.7 simplified: max error × Hessian-inv diag)
# --------------------------------------------------------------------- #
def importance_metric(W: np.ndarray, Wq: np.ndarray,
                      h_inv_diag: np.ndarray | None):
    log.debug(f"Computing importance metric for weights of shape {W.shape} "
              f"and quantised weights of shape {Wq.shape}...")
    err2 = (W - Wq)**2  # (O,I)
    sum_err = err2.sum(0)
    err_l2=np.sqrt(sum_err)
    # max_err = err2.max(axis=1)  # (O,)   DOUBT
    if h_inv_diag is None:
        print("taking hess diag as matrix with all elements as 1 due to error for imp calculation")
        h_inv_diag = np.ones_like(err_l2)
    elif len(h_inv_diag) != len(err_l2):
        print("taking hess diag as matrix with all elements as 1 due to error for imp calculation")
        h_inv_diag = np.ones_like(err_l2)
    return 0.5 * err_l2 * h_inv_diag

def reorder_channels(W: np.ndarray, imp: np.ndarray):
    log.debug(f"Reordering channels by importance metric of shape {imp.shape}...")
    perm = np.argsort(-imp)
    return W[:, perm], perm

def restore_order(W_sorted: np.ndarray, perm: np.ndarray):
    inv = np.argsort(perm)
    log.debug(f"Restoring original order of channels with shape {W_sorted.shape} "
              f"using permutation of shape {inv.shape}...")
    return W_sorted[:, inv]

# --------------------------------------------------------------------- #
# Beam search (multi-step)
# --------------------------------------------------------------------- #
def _recon(C_list, B_list):
    v = C_list[0][B_list[0]].copy()
    for C, B in zip(C_list[1:], B_list[1:]):
        v += C[B]
    return v

def beam_search_iterative(vecs, C_list, codes, beam=4, iters=4, tol=1e-4):
    prev = ((vecs - _recon(C_list, codes))**2).sum()
    for _ in range(iters):
        for l, (C, code) in enumerate(zip(C_list, codes)):
            for i in range(len(code)):
                best = code[i]
                best_err = ((vecs[i] - _recon(C_list, codes)[i])**2).sum()
                # Find beam candidates for codebook l
                cand = np.argpartition(((vecs[i] - (_recon(C_list, codes)[i] - C[code[i]] + C))**2).sum(1), beam)[:beam]
                for alt in cand:
                    old = code[i]; code[i] = alt
                    err = ((vecs[i] - _recon(C_list, codes)[i])**2).sum()
                    if err < best_err:
                        best, best_err = alt, err
                    else:
                        code[i] = old
                code[i] = best
        cur = ((vecs - _recon(C_list, codes))**2).sum()
        if (prev - cur)/(prev+1e-12) < tol:
            break
        prev = cur
    return codes

# --------------------------------------------------------------------- #
# Hessian-diag capture (activation hook)
# --------------------------------------------------------------------- #
def capture_h_inv_diag(model: torch.nn.Module, loader,
                       device="cpu") -> dict[str, np.ndarray]:
    acts = {}
    hooks = []
    def make_hook(name):
        def _hook(_,inp):
            x = inp[0].detach().to(device).flatten(0, -2)  # merge batch/seq
            acts[name] = acts.get(name, 0) + x.t() @ x  # (O,O)
        return _hook
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_pre_hook(make_hook(n)))
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device))
    for h in hooks: 
        h.remove()
    hinv = {}
    for n, H in acts.items():
        diag_inv = torch.diag(torch.inverse(H + 1e-6 * torch.eye(H.size(0), device=H.device)))
        hinv[n] = diag_inv.cpu().numpy().astype(np.float32)
    return hinv

# --------------------------------------------------------------------- #
# Compression formula  (Eq.8, CRVQ Appendix)
# --------------------------------------------------------------------- #
def compression_ratio(O, I, d, e, m, lam):
    n_vec_base = (O * I) // d
    base_bits = n_vec_base * e  # label bits for base codebook
    crit_rows = max(1, int(lam * O))
    n_vec_ext = crit_rows * (I / d)
    ext_bits = (m - 1) * n_vec_ext * e    # label bits for extended codebooks
    code_bits = m * (2 ** e) * d * 32     # centroids bits (m codebooks total)
    total_bits = base_bits + ext_bits + code_bits
    avg_bits = total_bits / (O * I)
    return 32 / avg_bits, avg_bits  # compression ratio (×), bits per parameter
