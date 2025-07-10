import os
import copy
import logging
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import utilities from provided modules
from main import load_architecture, accuracy, train_epochs
from crvq import CRVQ
from modules import capture_h_inv_diag, compression_ratio

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analysis")

# Hyperparameter grids to evaluate
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
ms      = [1, 2, 3, 4]
ds      = [4, 8, 16, 32]
es      = [6, 8, 10, 12]

# Prepare output CSV
csv_filename = "crvq_analysis_results.csv"
csv_fields = [
    "lambda", "m", "d", "e",
    "baseline_acc", "no_ft_acc", "block_ft_acc", "e2e_ft_acc",
    "accuracy_drop",  # (baseline - no_ft)
]
# Add per-layer compression columns dynamically (we will fill in after seeing model layers)
# We will name them as layer0_comp, layer1_comp, ... and also overall model compression.
# (compression defined as ratio: original model size / compressed size).
layer_comp_fields = []
model_comp_field = "model_comp"

# Load model architecture and weights
ARCH_FILE = "model.py"             # ensure this is present in working directory
CKPT_FILE = "model.pth"            # pre-trained weights checkpoint path
device = torch.device("cpu")       # change to "cuda" if GPU available for speed

ModelClass = load_architecture(ARCH_FILE)
model = ModelClass().to(device)
model.load_state_dict(torch.load(CKPT_FILE, map_location=device))
model.eval()

# Get list of Linear layer names to create CSV columns for each layer's compression
linear_layer_names = [name for name,module in model.named_modules() if isinstance(module, nn.Linear)]
for idx, name in enumerate(linear_layer_names):
    layer_comp_fields.append(f"{name}_comp_ratio")
csv_fields.extend(layer_comp_fields)
csv_fields.append(model_comp_field)

# Prepare data loaders (using MNIST as in main.py, replace with appropriate dataset if needed)
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                          batch_size=256, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                          batch_size=256)

# Compute baseline accuracy once
baseline_acc = accuracy(model, test_loader, device)
log.info(f"Baseline accuracy: {baseline_acc:.2f}%")

# Save a copy of the original weights to reset after each run
original_state = copy.deepcopy(model.state_dict())

# Capture Hessian inverse diagonal for all linear layers (importance metric)
log.info("Capturing Hessian inverse diagonal for importance metric...")
h_diag = capture_h_inv_diag(model, train_loader, device)
log.info("Hessian diagonal captured for layers: " + ", ".join(h_diag.keys()))

# Open CSV and write header
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_fields)

    # Iterate over all combinations of hyperparameters
    for lam in lambdas:
        for m in ms:
            for d in ds:
                for e in es:
                    log.info(f"Testing λ={lam}, m={m}, d={d}, e={e}")
                    # Reset model to original weights for this combination
                    model.load_state_dict(original_state)
                    model.eval()

                    # Apply CRVQ quantization
                    crvq = CRVQ(d=d, e=e, m=m, lam=lam)
                    # Quantize each linear layer and record compression
                    layer_compressions = {}
                    total_bits = 0
                    total_params = 0
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            W = module.weight.data  # original weights
                            # Quantize this layer (provide Hessian diag if available, else None)
                            quantized_W = crvq.quantise_layer(name, W, h_diag.get(name))
                            module.weight.data = quantized_W
                            # Compute compression ratio for this layer
                            O, I = W.shape
                            # If layer was skipped (CRVQ returns original weights if not applicable)
                            if name not in crvq.state:
                                # no quantization applied (e.g., if I<d), treat compression as 1x (no reduction)
                                comp_ratio = 1.0
                            else:
                                # Use the actual number of codebooks used (len(C_list) stored in state)
                                used_codebooks = len(crvq.state[name]['codebooks'])
                                cr, bpp = compression_ratio(O, I, d, e, used_codebooks, lam)
                                comp_ratio = cr
                            layer_compressions[name] = comp_ratio
                            # Accumulate for overall model compression (bits count)
                            # Calculate bits for this layer using same formula as compression_ratio
                            if name not in crvq.state:
                                # All weights 32 bits each
                                layer_bits = O * I * 32
                            else:
                                # Derive bits count from returned comp_ratio and original size
                                # comp_ratio = 32 / avg_bits_per_weight -> avg_bits = 32/comp_ratio
                                avg_bits = 32.0 / comp_ratio
                                layer_bits = avg_bits * O * I
                            total_bits += layer_bits
                            total_params += O * I

                    # Compute overall model compression ratio
                    avg_bits_model = total_bits / total_params
                    model_comp_ratio = 32.0 / avg_bits_model

                    # Evaluate accuracy after quantization (no fine-tuning)
                    no_ft_acc = accuracy(model, test_loader, device)
                    acc_drop = baseline_acc - no_ft_acc

                    # Fine-tune linear layers (block-wise) for a few epochs
                    if len(linear_layer_names) > 0:
                        lin_params = [p for n,m in model.named_modules() if isinstance(m, nn.Linear) for p in m.parameters()]
                    else:
                        lin_params = model.parameters()  # in case model has no Linear layers (unlikely here)
                    train_epochs(model, train_loader, lr=1e-4, epochs=1, params=lin_params)
                    block_ft_acc = accuracy(model, test_loader, device)

                    # End-to-end fine-tuning for a few epochs
                    train_epochs(model, train_loader, lr=1e-4, epochs=1, params=None)
                    e2e_ft_acc = accuracy(model, test_loader, device)

                    log.info(f"Results λ={lam}, m={m}, d={d}, e={e} | "
                             f"Accuracies: orig={baseline_acc:.2f}%, noFT={no_ft_acc:.2f}%, "
                             f"blockFT={block_ft_acc:.2f}%, e2eFT={e2e_ft_acc:.2f}% | "
                             f"Compression: {model_comp_ratio:.2f}x overall")

                    # Write results to CSV
                    row = [lam, m, d, e,
                           f"{baseline_acc:.2f}", f"{no_ft_acc:.2f}", f"{block_ft_acc:.2f}", f"{e2e_ft_acc:.2f}",
                           f"{acc_drop:.2f}"]
                    # layer-wise compression (as ratio x)
                    for name in linear_layer_names:
                        comp = layer_compressions.get(name, 1.0)
                        row.append(f"{comp:.2f}")
                    row.append(f"{model_comp_ratio:.2f}")
                    writer.writerow(row)

log.info(f"Analysis complete. Results saved to {csv_filename}")
