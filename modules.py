import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn


def Prequantize(W, d=8, e=8):
    """
    Step 3: Pre-quantizes the weight matrix W using Vector Quantization (VQ).
    This is an initial VQ step to estimate quantization error for importance computation.

    Args:
        W (np.array): Weight matrix to pre-quantize
        d (int): Vector dimension (default 8)
        e (int): Codebook bit-width (default 8, gives 2^8=256 vectors)

    Returns:
        np.array: Pre-quantized weight matrix
    """
    import numpy as np
    from sklearn.cluster import KMeans

    print("Prequantizing weight matrix W...")

    original_shape = W.shape
    is_conv = len(original_shape) == 4  ###special case of conv layers(optional)

    if is_conv:
        # Reshape convolutional weights for vectorization (flatten spatial and in_channels)
        # New shape will be (out_channels, in_channels * kernel_height * kernel_width)
        W_reshaped = W.reshape(original_shape[0], -1)
    else:
        W_reshaped = W.copy()

    # Reshape W_reshaped into d-dimensional vectors
    M, N_flat = W_reshaped.shape
    # Ensure N_flat is divisible by d, pad if necessary
    if N_flat % d != 0:
        padding = d - (N_flat % d)
        W_padded = np.pad(W_reshaped, ((0, 0), (0, padding)), mode='constant')
    else:
        W_padded = W_reshaped.copy()
        padding = 0
    # print[{W_padded.shape}]
    # Partition into d-dimensional vectors
    vectors = W_padded.reshape(-1, d)
    # print({vectors.shape})
    # Apply K-means clustering
    n_clusters = 2**e if 2**e <= len(vectors) else len(vectors)
    # Ensure n_clusters is not zero if vectors is empty
    if len(vectors) == 0:
        print("Warning: No vectors to cluster, returning original weights.")
        return W
    # kmeans2(X, k, minit=init, iter=iterations)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10,init="k-means++")
    cluster_labels = kmeans.fit_predict(vectors)

    # Replace each vector with its cluster center
    quantized_vectors = kmeans.cluster_centers_[cluster_labels]

    # Reshape back to original reshaped shape (before padding)
    W_quant_padded = quantized_vectors.reshape(W_padded.shape)

    # Remove padding if it was added
    if padding > 0:
        W_quant_reshaped = W_quant_padded[:, :-padding]
    else:
        W_quant_reshaped = W_quant_padded

    if is_conv:
        # Reshape back to original convolutional shape
        W_quant = W_quant_reshaped.reshape(original_shape)
    else:
        W_quant = W_quant_reshaped

    return W_quant

def ComputeImportance(W, Wquant, XXT):
    """
    Step 4: Computes the importance I for each weight channel using Hessian metric.

    The importance is formulated as: Ii = max_j [w_ji - VQ(w_ji)]^2 / 2 * [XXT]^-1_ii
    For convolutional layers, channels correspond to output channels.

    Args:
        W (np.array): Original weight matrix.
        Wquant (np.array): Pre-quantized weight matrix.
        XXT (np.array): Hessian proxy, derived from activations X.

    Returns:
        np.array: An array of importance scores for each channel (row in flattened view).
    """
    print("Computing channel importance...")

    original_shape = W.shape
    is_conv = len(original_shape) == 4

    if is_conv:
        # Reshape convolutional weights for calculation
        W_reshaped = W.reshape(original_shape[0], -1)
        Wquant_reshaped = Wquant.reshape(original_shape[0], -1)
        M, N_flat = W_reshaped.shape # M is number of output channels/rows
    else:
        W_reshaped = W.copy()
        Wquant_reshaped = Wquant.copy()
        M, N_flat = W_reshaped.shape # M is number of rows


    # Compute quantization error for each element in the reshaped matrix
    quant_error_reshaped = (W_reshaped - Wquant_reshaped) ** 2

    # Compute diagonal of XXT inverse (regularized to avoid numerical issues)
    # Ensure XXT is square and matches the second dimension of W_reshaped (N_flat x N_flat)
    if XXT.shape[0] != N_flat or XXT.shape[1] != N_flat:
        print(f"Warning: XXT shape {XXT.shape} does not match expected shape {N_flat}x{N_flat}. Cannot compute Hessian-based importance. Using simplified importance (mean error per row).")
        # Fallback to simplified importance if XXT shape is incorrect - mean error per row (output channel)
        importance = np.mean(quant_error_reshaped, axis=1)
        return importance

    reg_factor = 1e-6
    XXT_reg = XXT + reg_factor * np.eye(XXT.shape[0])

    try:
        XXT_inv_diag = np.diag(np.linalg.inv(XXT_reg))
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        print("Warning: XXT is singular, using pseudo-inverse for diagonal.")
        XXT_inv_diag = np.diag(np.linalg.pinv(XXT_reg))

    # Ensure XXT_inv_diag has the right shape (N_flat,)
    if len(XXT_inv_diag) != N_flat:
        print("Warning: XXT_inv_diag length doesn't match reshaped weight dimension. Using simplified metric (mean error per row).")
        importance = np.mean(quant_error_reshaped, axis=1)
    else:
        # Compute importance for each *row* (output channel) in the reshaped matrix:
        # I_i = sum_j [ (w_ij - VQ(w_ij))^2 * [XXT]^-1_jj ] / 2
        # where i iterates over rows (output channels) and j iterates over columns (input features).
        # We need to sum across the columns (axis=1) for each row.
        # quant_error_reshaped is M x N_flat
        # XXT_inv_diag is N_flat, needs to be broadcasted to M x N_flat
        importance = np.sum(quant_error_reshaped * XXT_inv_diag[np.newaxis, :], axis=1) / 2.0


    # The importance calculated here is for each *row* of the reshaped matrix,
    # which corresponds to output channels for both linear and convolutional layers.
    return importance


def ReorderChannels(W, I):

    """

    Step 5: Reorders channels of the weight matrix W based on their importance I.

    Critical channels (those with higher importance) are grouped together.

    For convolutional layers, this reorders the *output channels* based on the

    importance scores associated with them.

    Importance I is assumed to be for each output channel (row of the reshaped matrix).



    Args:

        W (np.array): Reshaped 2D weight matrix (M x N_flat).

        I (np.array): Importance scores for each output channel (length M).



    Returns:

        tuple: (W_sorted, original_indices_map) where W_sorted is the reordered matrix

               and original_indices_map allows restoring the original order.

    """

    print("Reordering channels...")

    # Importance I is assumed to be for each row (output channel)

    # Get the indices that would sort the importance scores in descending order

    sorted_indices = np.argsort(I)[::-1]

    # Ensure the number of importance scores matches the number of rows in W
    if len(I) != W.shape[0]:
        print(f"Error: Number of importance scores ({len(I)}) does not match number of rows in W ({W.shape[0]}). Cannot reorder.")
        # Return original W and identity mapping if shapes don't match
        return W.copy(), np.arange(W.shape[0])


    W_sorted = W[sorted_indices, :]



    # Store the mapping to restore original order later
    # This map allows original_W[original_indices_map[i], :] == W_sorted[i, :]
    # It maps the index in the sorted array back to the index in the original array.
    original_indices_map = np.argsort(sorted_indices)


    return W_sorted, original_indices_map

def VectorQuant(vectors, codebook):
    """
    Steps 7 & 11: Replaces each vector in a set with its closest vector from the given codebook.
    Returns the quantized vectors and their corresponding binary encodings (indices).

    Args:
        vectors (np.array): The d-dimensional vectors to quantize.
        codebook (np.array): The codebook (cluster centers).

    Returns:
        tuple: (quantized_vectors, binary_encodings)
               quantized_vectors (np.array): Vectors replaced by their closest codebook entry.
               binary_encodings (np.array): Indices of the closest codebook entries.
    """
    print("Performing Vector Quantization...")

    if len(vectors) == 0:
        print("Warning: No vectors to quantize.")
        return np.array([]).reshape(0, vectors.shape[1]), np.array([], dtype=int)
    if len(codebook) == 0:
        print("Warning: Empty codebook.")
        # Return original vectors if codebook is empty (no quantization)
        return vectors, np.zeros(len(vectors), dtype=int)

    # Compute distances from each vector to all codebook entries
    # vectors: (n_vectors, d), codebook: (n_codes, d)
    # distances: (n_vectors, n_codes)
    # Use optimized distance calculation if possible, or a loop for large codebooks/vectors
    try:
        # Attempt a direct broadcasted norm calculation
        distances = np.linalg.norm(vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :], axis=2)
    except MemoryError:
        print("MemoryError during distance calculation, using a loop...")
        distances = np.zeros((vectors.shape[0], codebook.shape[0]))
        for i in range(vectors.shape[0]):
            distances[i, :] = np.linalg.norm(vectors[i, :] - codebook, axis=1)


    # Find closest codebook entry for each vector
    binary_encodings = np.argmin(distances, axis=1)

    # Replace vectors with their closest codebook entries
    quantized_vectors = codebook[binary_encodings]

    return quantized_vectors, binary_encodings

def ComputeError(W_lambda, W_lambda_encoded):
    """
    Step 9: Computes the residual error for the important channels (rows).
    This is E_t = W_lambda - W_lambda_encoded, where W_lambda are the
    important channels (rows) and W_lambda_encoded is their current quantized representation.

    Args:
        W_lambda (np.array): Original values of the important channels (rows).
        W_lambda_encoded (np.array): Currently quantized values of the important channels (rows).

    Returns:
        np.array: The residual error (E_t).
    """
    print("Computing residual error for important channels...")
    return W_lambda - W_lambda_encoded



def Update(W_lambda_encoded, Et_encoded):

    """

    Step 12: Additively updates the quantized important channels (rows).

    W_lambda_encoded is updated by adding the newly quantized residual E_t_encoded.

    This is part of additive VQ [1].



    Args:

        W_lambda_encoded (np.array): Current quantized representation of important channels (rows).

        Et_encoded (np.array): Quantized residual error from the current extended codebook.



    Returns:

        np.array: Updated quantized representation of important channels (rows).

    """

    print("Updating quantized important channels additively...")

    return W_lambda_encoded + Et_encoded

def QuantLoss(W_original_reshaped, W_encoded_reshaped, X):
    """
    Step 14: Calculates the quantization loss, typically ∥W_reshaped X − W_encoded_reshaped X∥^2_2.
    This loss guides the fine-tuning process. Operates on reshaped 2D weights.

    Args:
        W_original_reshaped (np.array): Original weight matrix (reshaped to 2D).
        W_encoded_reshaped (np.array): Current encoded/quantized weight matrix (reshaped to 2D).
        X (np.array): Activation input (expected shape N_flat x O).

    Returns:
        float: The calculated quantization loss.
    """
    print("Calculating quantization loss...")

    # Ensure X has the correct shape for multiplication with reshaped weights (N_flat x O)
    # W_reshaped is M x N_flat
    # X is N_flat x O
    # Result is M x O
    if W_original_reshaped.shape[1] != X.shape[0]:
         print(f"Warning: Activation shape {X.shape} does not match reshaped weight dimension {W_original_reshaped.shape}. Skipping loss calculation.")
         return 0.0 # Cannot compute loss if dimensions don't match

    # Loss is ||WX - WencodedX||^2_2
    original_output = np.dot(W_original_reshaped, X)
    encoded_output = np.dot(W_encoded_reshaped, X)
    loss = np.linalg.norm(original_output - encoded_output, 'fro') ** 2

    return loss



def FineTuneCodebook(Cbase, Cext_list, W_original_reshaped, W_encoded_reshaped, X, lr=0.01, max_iter=10):
    """
    Step 15: Fine-tunes the basic and extended codebooks.
    This involves optimizing the codebook entries to minimize quantization error.
    Operates on reshaped 2D weights.

    Args:
        Cbase (np.array): Basic codebook
        Cext_list (list): List of extended codebooks
        W_original_reshaped (np.array): Original weight matrix (reshaped to 2D)
        W_encoded_reshaped (np.array): Encoded weight matrix (reshaped to 2D)
        X (np.array): Activation input (expected shape N_flat x O)
        lr (float): Learning rate
        max_iter (int): Maximum iterations
    """
    print("Fine-tuning codebooks...")

    # Simple gradient-based optimization
    # In practice, this would involve differentiating through the VQ assignment and reconstruction.
    # This simplified version applies perturbations based on the loss.
    for iteration in range(max_iter):
        # Compute current loss
        current_loss = QuantLoss(W_original_reshaped, W_encoded_reshaped, X)
        if current_loss == 0.0: # Skip if loss cannot be computed
             break

        # Compute gradients (simplified approach)
        # The gradient of the loss w.r.t. W_encoded_reshaped is -2 * (W_original_reshaped - W_encoded_reshaped) @ X.T @ X
        # This is a rough proxy. A proper gradient would consider the VQ mapping.
        grad_W_encoded = -2 * np.dot((W_original_reshaped - W_encoded_reshaped), np.dot(X, X.T))

        # How to relate grad_W_encoded back to codebook updates is complex.
        # For simplification, we'll use a perturbation-based approach scaled by the gradient magnitude.
        grad_scale = lr * np.linalg.norm(grad_W_encoded, 'fro') / (1 + iteration)

        if grad_scale < 1e-9: # Stop if gradient is too small
            break

        # Apply updates to codebooks based on gradient direction related to encoded weights
        # This is a highly simplified heuristic. A proper VQ fine-tuning method (e.g., LSQ) is needed.
        # For now, just apply small random perturbations scaled by the loss.
        Cbase += np.random.normal(0, grad_scale * 0.001, Cbase.shape)

        # Update extended codebooks
        for i, Cext in enumerate(Cext_list):
            Cext_list[i] += np.random.normal(0, grad_scale * 0.001, Cext.shape)


        if iteration % 5 == 0:
            print(f"  Fine-tuning iteration {iteration}, loss: {current_loss:.6f}")


def BeamSearchOptimize(W_original_reshaped, Cbase, Cext_list, Bbase, Bext_list, beam_width=4):
    """
    Step 16: Optimizes the binary codes (encodings) using beam search.
    This finds a better set of codebook indices that minimizes the approximation error.
    Operates on reshaped 2D weights.

    Args:
        W_original_reshaped (np.array): Original weight matrix (reshaped to 2D)
        Cbase (np.array): Basic codebook
        Cext_list (list): List of extended codebooks
        Bbase (np.array): Basic binary encodings
        Bext_list (list): List of extended binary encodings
        beam_width (int): Width of beam search
    """
    print("Performing Beam Search Optimization...")

    # This is a highly simplified placeholder. A proper beam search would
    # iteratively build up the best sequence of codebook assignments.

    # For this placeholder, we'll just return the original encodings.
    print("  Beam Search Optimization is a placeholder, returning original encodings.")

    return Bbase, Bext_list # Return original encodings


def FineTuneBlock(model, block_idx, X, y, lr=0.001, max_iter=50):
    """
    Step 19: Block-level fine-tuning.
    Adjusts a few parameters within the current block to mitigate model sensitivity to
    extreme compression.

    Args:
        model: The model containing the block to fine-tune
        block_idx: Index of the block to fine-tune
        X: Input data
        y: Target data
        lr: Learning rate
        max_iter: Maximum iterations
    """
    print(f"Fine-tuning block {block_idx}...")

    if hasattr(model, 'train'):
        model.train()

        # Simple fine-tuning loop
        # In a real implementation, you'd identify parameters specific to the block
        # and fine-tune only those. Accessing module parameters by index is fragile.
        # A better approach would be to pass the specific module to fine-tune.
        # For this placeholder, we'll skip actual fine-tuning if the model is not a nn.Module
        # or if parameters cannot be accessed easily.

        # Access parameters of the specific block/layer
        # This part needs to be more robust depending on how blocks are defined
        block_params = []
        # Example: assuming blocks are sequential layers
        # If model is nn.Sequential, model[block_idx] could be the block.
        # If blocks are within custom class, need to access them by name or attribute.

        # Placeholder: Try to get parameters from the whole model, this is NOT block-specific fine-tuning
        # This will fine-tune the whole model, which is not the intent of block fine-tuning.
        # TODO: Implement proper block-specific parameter identification.
        if isinstance(model, nn.Module) and hasattr(model, 'parameters'):
             # Find the parameters for the specific layer being quantized (this function is called after quantizing a layer)
            try:
                # Assuming block_idx somehow maps to a specific module in the model's named_modules()
                # This mapping is not provided, so this is a guess.
                module_to_finetune = None
                current_layer_idx = -1
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                         current_layer_idx += 1
                         if current_layer_idx == block_idx:
                             module_to_finetune = module
                             break

                if module_to_finetune is not None:
                     print(f"  Fine-tuning parameters for module: {name}")
                     block_params = list(module_to_finetune.parameters())
                else:
                     print(f"  Could not find module for block_idx {block_idx}. Skipping block fine-tuning.")
                     return

            except Exception as e:
                print(f"  Error identifying block parameters: {e}. Skipping block fine-tuning.")
                return


            if not block_params:
                print("  No trainable parameters found for block. Skipping block fine-tuning.")
                return

            import torch.optim as optim
            optimizer = optim.Adam(block_params, lr=lr)
            criterion = nn.CrossEntropyLoss() # Assuming classification task

            # Move data to the same device as the model
            device = next(model.parameters()).device
            X, y = X.to(device), y.to(device)


            for epoch in range(max_iter):
                optimizer.zero_grad()
                outputs = model(X) # Forward pass through the whole model
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f"  Block fine-tuning epoch {epoch}, loss: {loss.item():.6f}")
        else:
            print("  Model does not support training or parameters access, skipping block fine-tuning")


def E2E_FineTune(model, train_loader=None, lr=0.001, max_epochs=10):
    """
    Step 21: End-to-end fine-tuning of the entire quantized model.
    This is a final optional step to enhance overall quantization performance.

    Args:
        model: The quantized model to fine-tune
        train_loader: DataLoader for training data
        lr: Learning rate
        max_epochs: Maximum training epochs

    Returns:
        The fine-tuned model
    """
    print("Performing End-to-End Fine-tuning of the entire model...")

    if train_loader is None or not hasattr(model, 'train'):
        print("  No training data or model doesn't support training, skipping E2E fine-tuning")
        return model

    model.train()
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    device = next(model.parameters()).device
    model.to(device)

    for epoch in range(max_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  E2E Fine-tuning epoch {epoch}, avg loss: {avg_loss:.6f}")

    return model


def KMeansCodebook(vectors, n_clusters=256, random_state=42):
    """
    Creates a codebook using K-means clustering.

    Args:
        vectors (np.array): Vectors to cluster
        n_clusters (int): Number of clusters (codebook size)
        random_state (int): Random state for reproducibility

    Returns:
        np.array: Codebook (cluster centers)
    """
    if len(vectors) == 0:
        print("Warning: Cannot create codebook from empty vectors.")
        return np.array([]) # Return empty array if no vectors

    if len(vectors) < n_clusters:
        n_clusters = len(vectors)
        print(f"Warning: Number of vectors ({len(vectors)}) is less than n_clusters ({n_clusters}). Setting n_clusters to {n_clusters}.")

    # Ensure n_clusters is at least 1 if vectors are present
    if n_clusters == 0:
         print("Warning: n_clusters is 0, setting to 1.")
         n_clusters = 1

    # kmeans2(X, k, minit=init, iter=iterations)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10,init='k-means++')
    kmeans.fit(vectors)
    return kmeans.cluster_centers_


def partition_to_vectors(W_reshaped, d=8):
    """
    Partitions a reshaped 2D weight matrix into d-dimensional vectors.

    Args:
        W_reshaped (np.array): Reshaped 2D weight matrix (M x N_flat)
        d (int): Vector dimension

    Returns:
        tuple: (vectors, padding)
               vectors (np.array): Reshaped vectors
               padding (int): Amount of padding added to the last dimension
    """
    M, N_flat = W_reshaped.shape

    # Ensure N_flat is divisible by d, pad if necessary
    if N_flat % d != 0:
        padding = d - (N_flat % d)
        W_padded = np.pad(W_reshaped, ((0, 0), (0, padding)), mode='constant')
    else:
        W_padded = W_reshaped.copy()
        padding = 0

    # Partition into d-dimensional vectors
    # Reshape from (M, N_flat + padding) to (M * (N_flat + padding) / d, d)
    vectors = W_padded.reshape(-1, d)

    return vectors, padding


def vectors_to_matrix(vectors, original_reshaped_shape, padding=0):
    """
    Converts vectors back to reshaped 2D matrix format.

    Args:
        vectors (np.array): Vector array (num_vectors x d)
        original_reshaped_shape (tuple): Original 2D reshaped matrix shape (M x N_flat)
        padding (int): Amount of padding to remove from the last dimension

    Returns:
        np.array: Reconstructed 2D reshaped matrix
    """
    M, N_flat = original_reshaped_shape
    d = vectors.shape[1]

    # Calculate padded shape for the second dimension
    N_padded = N_flat + padding if padding > 0 else N_flat

    # Reshape vectors back to padded 2D matrix
    # Reshape from (num_vectors, d) to (M, N_padded)
    # Ensure num_vectors * d matches M * N_padded
    expected_num_vectors = M * N_padded // d
    if vectors.shape[0] != expected_num_vectors:
        print(f"Error: Vector count {vectors.shape[0]} does not match expected {expected_num_vectors} for reshaping to ({M}, {N_padded}).")
        # Attempt to reshape anyway, might result in error or incorrect shape
        W_reconstructed_padded = vectors.reshape(M, N_padded)
    else:
        W_reconstructed_padded = vectors.reshape(M, N_padded)


    # Remove padding if it was added
    if padding > 0:
        W_reconstructed_reshaped = W_reconstructed_padded[:, :-padding]
    else:
        W_reconstructed_reshaped = W_reconstructed_padded

    return W_reconstructed_reshaped
