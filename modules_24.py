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
from datetime import datetime
import logging, numpy as np, torch
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, vq as scipy_vq
from scipy.spatial.distance import cdist
from IPython import embed
log = logging.getLogger("modules")
log.addHandler(logging.NullHandler())
file_handler = logging.FileHandler('logs_test.log')
log.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
file_handler.setFormatter(formatter)

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
    # codes=codes.astype(np.int32) 

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
    box=0.5 * (err2 @ h_inv_diag)
    return box



def reorder_channels(W: np.ndarray, imp: np.ndarray):
    log.debug(f"Reordering channels by importance metric of shape {imp.shape}...")

    perm = np.argsort(-imp)
    return W[perm], perm

def restore_order(W_sorted: np.ndarray, perm: np.ndarray):
    inv = np.argsort(perm)
    log.debug(f"Restoring original order of channels with shape {W_sorted.shape} "
              f"using permutation of shape {inv.shape}...")
    return W_sorted[inv]
# --------------------------------------------------------------------- #
# Beam search (multi-step)-----CUDA CHECK ---21 JULY
# --------------------------------------------------------------------- #

def _recon_torch(C_list, B_list):
    v = C_list[0][B_list[0]]
    for C, B in zip(C_list[1:], B_list[1:]):
        v = v + C[B]
    return v

def beam_search_iterative(vecs, C_list, codes,
                                beam=4, iters=4, tol=1e-4,
                                device="cuda"):
    """
    CUDA-aware rewrite of beam_search_iterative.
      vecs  : (N,d) NumPy or torch tensor, float32
      C_list: list of (K,d) centroid arrays / tensors, float32
      codes : list of (N,) code arrays / tensors, int
    Returns: list of NumPy uint16/uint8 codes (shape unchanged)
    """
    # --- to torch ---
    vecs  = torch.as_tensor(vecs,  dtype=torch.float32, device=device)
    C_list = [torch.as_tensor(C, dtype=torch.float32, device=device) for C in C_list]
    codes  = [torch.as_tensor(B, dtype=torch.int32,  device=device).clone() for B in codes]

    prev = torch.sum((vecs - _recon_torch(C_list, codes)) ** 2)

    for _ in range(iters):
        for C, code in zip(C_list, codes):
            recon = _recon_torch(C_list, codes)      # (N,d)
            # for i in range(code.numel()):
            #     best      = code[i].item()
            #     best_err  = torch.sum((vecs[i] - recon[i]) ** 2)

            #     # candidate centroids for the *current* book
            #     diff = vecs[i] - (recon[i] - C[code[i]] + C)   # (K,d)
            #     errs = torch.sum(diff ** 2, dim=1)
            #     _, cand = torch.topk(errs, beam, largest=False)

            #     for alt in cand:
            #         old = code[i].item()
            #         code[i] = alt
            #         err = torch.sum((vecs[i] - _recon_torch(C_list, codes)[i]) ** 2)
            #         if err < best_err:
            #             best, best_err = alt.item(), err
            #         else:
            #             code[i] = old
            #     code[i] = best
            ##################################################change1 23july########
            for i in range(code.numel()):
                best = code[i].item()
                # Current reconstruction for vector i
                recon_i = sum(C[B[i]] for C, B in zip(C_list, codes))
                best_err = torch.sum((vecs[i] - recon_i) ** 2)

                # Candidate centroids for current codebook
                diff = vecs[i] - (recon_i - C[code[i]] + C)  # (K, d)
                errs = torch.sum(diff ** 2, dim=1)
                _, cand = torch.topk(errs, beam, largest=False)

                for alt in cand:
                    err = torch.sum((vecs[i] - (recon_i - C[code[i]] + C[alt])) ** 2)
                    if err < best_err:
                        best, best_err = alt.item(), err

                code[i] = best
            ############################################################################
        cur = torch.sum((vecs - _recon_torch(C_list, codes)) ** 2)
        if (prev - cur) / (prev + 1e-12) < tol:
            break
        prev = cur

    # --- back to NumPy so downstream code stays unchanged ---
    return [B.cpu().numpy().astype('uint8') for B in codes]




def capture_h_inv_diag(model: torch.nn.Module, loader,
                       device="cpu") -> dict[str, np.ndarray]:
    acts = {}
    hooks = []
    def make_hook(name):
        def _hook(_,inp):
            x = inp[0].detach().to(device).flatten(0, -2)  # merge batch/seq
            acts[name] = acts.get(name, 0) + x.t() @ x  # (N,N)

           # embed()
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
        # diag_inv = torch.diag(torch.inverse(H + 1e-6 * torch.eye(H.size(0), device=H.device)))
        diag=torch.diag(H + 1e-6 * torch.eye(H.size(0), device=H.device))
        # embed()
        hinv[n] = diag.cpu().numpy().astype(np.float32)
        # hinv[n] = diag_inv.cpu().numpy().astype(np.float32)
        
    return hinv

# --------------------------------------------------------------------- #
# Compression formula  (Eq.8, CRVQ Appendix)
# --------------------------------------------------------------------- #
def compression_ratio(O, I, d, e, m, lam):
    n_vec_base = (O * I) // d
    base_bits = n_vec_base * 8  # label bits for base codebook
    crit_rows = max(1, int(lam * O))
    n_vec_ext = crit_rows * (I / d)
    ext_bits = (m - 1) * n_vec_ext * 8    # label bits for extended codebooks
    code_bits = m * (2 ** e) * d * 32     # centroids bits (m codebooks total)
    total_bits = base_bits + ext_bits + code_bits
    avg_bits = total_bits / (O * I)
    return 32 / avg_bits, avg_bits  # compression ratio (×), bits per parameter
