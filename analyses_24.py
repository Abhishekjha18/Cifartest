import copy
import logging
import csv
import torch
import os, logging, torch
import numpy as np
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import time
from IPython import embed
# Import utilities from provided modules
from main import accuracy, train_epochs
from crvq import CRVQ
from modules import capture_h_inv_diag, compression_ratio
from data_loader import data_load
from model_loader import model_load
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analysis")
file_handler = logging.FileHandler('logs_test.log')
log.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
file_handler.setFormatter(formatter)

''' --------------------------------- HARD CODE DEVICE(CPU,CUDA), MODEL_NAME (VIT,CNN), ARCH_FILE, CKPT_FILE AND HYPERPARAMETERS, csv file name -------------------'''
# Hyperparameter grids to evaluate
lambdas = [0.1]
ms      = [4]
ds      = [8]
es      = [8]
################################################################
model_name= "vit"   #chose between 'vit' or 'cnn'
ARCH_FILE = "modeling.py"                      #vit arch file
CKPT_FILE = "cifar100_run_checkpoint.bin"        #vit model file       
# -------choose either above or below-------#
# model_name="cnn"
# ARCH_FILE="model_2.py"
# CKPT_FILE="model_2.pth"
# # ------------------------------------------#
#################################################################
device = torch.device("cuda")         # change to "cuda" if GPU available for speed
csv_filename = "results_vit.csv"
'''--------------------------------------------------------------------------------------------------------------------------------------------------------------'''
# Prepare output CSV
csv_fields = [
    "lambda", "m", "d", "e",
    "baseline_acc", "no_ft_acc", "block_ft_acc", "e2e_ft_acc",
    "accuracy_drop",  # (baseline - no_ft)
]
layer_comp_fields = []
model_comp_field = "model_comp"


model=model_load(ARCH_FILE,CKPT_FILE,model_name,device)
print("model setup done")
model.load_state_dict(torch.load(CKPT_FILE, map_location=device))
model.eval()
print("model in evel state")

# Get list of Linear layer names to create CSV columns for each layer's compression
linear_layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
for idx, name in enumerate(linear_layer_names):
    layer_comp_fields.append(f"{name}_comp_ratio")
csv_fields.extend(layer_comp_fields)
csv_fields.append(model_comp_field)

tr_loader, te_loader= data_load(model_name)

# Compute baseline accuracy once

baseline_acc = accuracy(model, te_loader, device, model_name)
print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))

log.info(f"Baseline accuracy: {baseline_acc:.2f}%")

# Save a copy of the original weights to reset after each run
original_state = copy.deepcopy(model.state_dict())

# Capture Hessian inverse diagonal for all linear layers (importance metric)
print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
log.info("Capturing Hessian inverse diagonal for importance metric...")
h_diag = capture_h_inv_diag(model, tr_loader, device)
log.info("Hessian diagonal captured for layers: ")
print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
for n, d in h_diag.items():
        log.info(f"Layer {n}: Hessian diag shape {d.shape}, dtype {d.dtype}")

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
                    print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
                    # Reset model to original weights for this combination
                    model.load_state_dict(original_state)
                    model.eval()

                    # Apply CRVQ quantization
                    crvq = CRVQ(d=d, e=e, m=m, lam=lam)
                    print("TIME STAMP CRVQ STEPs: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
                    # Quantize each linear layer and record compression
                    layer_compressions = {}
                    total_bits = 0
                    total_params = 0
                    
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            W = module.weight.data  # original weights
                            # Quantize this layer (provide Hessian diag if available, else None)
                            quantized_W = crvq.quantise_layer(name, W, h_diag.get(name))
                            print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
                            module.weight.data = quantized_W
                            model.to(device)
                            log.info(f"Model accuracy after layer {name} quantization: {accuracy(model, te_loader, device , model_name)}")
                            #--------------ATTENTION 1 , BELOW LINES ARE FOR EXPERIMENT TO SEE WHICH LAYER IMPACTS THE MODEL MOST---------
                            
                            # log.info(f"Resoring Original Weights")
                            # module.weight.data=W
                            # log.info(f"Model accuracy after layer {name} restoring(should be baseline): {accuracy(model, te_loader, device , model_name)}")
                            # log.info(f"MODEL HAS RETAINED ORIGNAL SHAPE, GOOD FOR EXPERIMENT BUT NOT GOOD FOR CRVQ")
                            
    
                            #--------------------------------------------------------------------------------------------------------------------------#
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
                    no_ft_acc = accuracy(model, te_loader, device,model_name)
                    acc_drop = baseline_acc - no_ft_acc
                    
                    # Fine-tune linear layers (block-wise) for a few epochs
                    if len(linear_layer_names) > 0:
                        lin_params = [p for n, m in model.named_modules() if isinstance(m, nn.Linear) for p in m.parameters()]
                    else:
                        lin_params = model.parameters()  # in case model has no Linear layers (unlikely here)
                    train_epochs(model, tr_loader, lr=1e-4, epochs=1, params=lin_params,device=device,model_name=model_name)
                    block_ft_acc = accuracy(model, te_loader, device,model_name)
                    
                    # End-to-end fine-tuning for a few epochs
                    train_epochs(model, tr_loader, lr=1e-4, epochs=1, params=None,device=device,model_name=model_name)
                    e2e_ft_acc = accuracy(model, te_loader, device,model_name)

                    log.info(f"Results λ={lam}, m={m}, d={d}, e={e} | "
                             f"Accuracies: orig={baseline_acc:.2f}%, noFT={no_ft_acc:.2f}%, "
                             f"blockFT={block_ft_acc:.2f}%, e2eFT={e2e_ft_acc:.2f}%")
                    
                    # Write results to CSV
                    row = [lam, m, d, e,
                           baseline_acc, no_ft_acc, block_ft_acc, e2e_ft_acc,
                           acc_drop]
                    # Append per-layer compression ratios in order and overall model compression
                    for lname in linear_layer_names:
                        row.append(layer_compressions.get(lname, 1.0))
                    row.append(model_comp_ratio)
                    writer.writerow(row)
