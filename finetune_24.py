import torch
import logging
from tqdm import trange


def fine_tune_codebooks(codebooks, assignments, original_vectors, 
                        steps=5000, lr=1e-4, optimizer_type='Adam', device=None):
    """
    Fine-tune codebook centroids via gradient descent to minimize reconstruction loss.

    Parameters:
        codebooks (list of np.array or torch.Tensor): List of codebook matrices (each shape [K, D]).
        assignments (list of np.array or torch.Tensor): List of assignment arrays, one per codebook (shape [N]).
        original_vectors (np.array or torch.Tensor): Original unquantized vectors, shape [N, D].
        steps (int): Number of optimization steps.
        lr (float): Learning rate.
        optimizer_type (str): 'Adam' or 'SGD'.
        device (str or torch.device): 'cuda', 'cpu', or None to auto-select.

    Returns:
        None. Codebooks are updated in-place.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    logger = logging.getLogger("fine_tune_codebooks")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('logs_test.log')
    logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.info(f"Starting fine-tuning on {len(codebooks)} codebooks | Device: {device}")

    # Convert inputs to tensors
    orig = torch.as_tensor(original_vectors, dtype=torch.float32, device=device)
    orig.requires_grad = False

    codebook_params = [
        torch.tensor(cb, dtype=torch.float32, device=device, requires_grad=True)
        for cb in codebooks
    ]

    assignment_tensors = [
        torch.as_tensor(B, dtype=torch.long, device=device)
        for B in assignments
    ]

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(codebook_params, lr=lr)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(codebook_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    N = orig.size(0)
    for step in range(steps):
        optimizer.zero_grad()
        recon = torch.zeros_like(orig)
        for cb_tensor, B_tensor in zip(codebook_params, assignment_tensors):
            recon += cb_tensor[B_tensor]

        loss = torch.nn.functional.mse_loss(recon, orig, reduction='mean')
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == steps - 1:
            logger.info(f"Step {step+1}/{steps} - Loss: {loss.item():.6f}")

    # Copy back refined centroids
    for i, cb_tensor in enumerate(codebook_params):
        refined = cb_tensor.detach().cpu().numpy()
        if isinstance(codebooks[i], torch.Tensor):
            codebooks[i].data.copy_(cb_tensor.data)
        else:
            codebooks[i][...] = refined

    logger.info("Codebook fine-tuning complete.\n")
