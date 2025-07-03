import numpy as np, math
from scipy.cluster.vq import kmeans2, vq as scipy_vq
import matplotlib.pyplot as plt



def vq_encode(weights: np.ndarray,
              k: int = 8,
              subvec_cols: int = 2,
              init: str = "++",
              iterations: int = 25,
              random_state: int | None = None):
    if random_state is not None:
        np.random.seed(random_state)

    rows, cols = weights.shape
    assert cols % subvec_cols == 0, "`cols` must be divisible by subvec_cols"

    n_subvectors = rows * (cols // subvec_cols)
    X = weights.reshape(rows, -1, subvec_cols).reshape(n_subvectors, subvec_cols)

    centroids, _ = kmeans2(X, k, minit=init, iter=iterations)
    codes, _     = scipy_vq(X, centroids)

    return dict(centroids=centroids,
                codes=codes,
                subvec_cols=subvec_cols,
                original_shape=X)

def analyse_vq(weights: np.ndarray,
               centroids: np.ndarray,
               codes: np.ndarray,
               subvec_cols: int,
               plot: bool = False):
    rows, cols = weights.shape
    n_subvectors = codes.size

    # --- reconstruction ---
    X_hat = centroids[codes]
    W_hat = X_hat.reshape(rows, cols)
    error_matrix = np.abs(W_hat - weights)
    error_shape=list[error_matrix.shape]
    # --- error ---
    rmse = float(np.sqrt(np.mean((weights - W_hat) ** 2)))

    # --- compression maths ---
    k = centroids.shape[0]
    bits_per_weight = 32
    bits_per_index  = int(math.ceil(np.log2(k)))
    orig_bits       = weights.size * bits_per_weight
    idx_bits        = n_subvectors * bits_per_index
    cen_bits        = centroids.size * bits_per_weight
    comp_bits       = idx_bits + cen_bits
    comp            = dict(original=orig_bits,
                           indices=idx_bits,
                           centroids=cen_bits,
                           compressed=comp_bits,
                           ratio=orig_bits / comp_bits)
    return dict(reconstructed=W_hat,
                rmse=rmse,
                compression_bits=comp,
                error_matrix=error_matrix, error_shape=error_shape)                       
