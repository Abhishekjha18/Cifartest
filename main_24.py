# -*- coding: utf-8 -*-
# LOGLEVEL=INFO python main.py --arch original_model.py --ckpt original_model.pth --out-dir results --model_type "cnn"
"""
main.py - end-to-end pipeline:
 • load architecture + weights
 • baseline accuracy
 • CRVQ quantise all Linear layers
 • no-FT, block-FT, E2E-FT accuracies
 • compression summary
"""
import os, logging, importlib.util, argparse, torch
import numpy as np
import torch.nn as nn, torch.optim as optim

from tqdm import tqdm
from modules import capture_h_inv_diag
from crvq import CRVQ
from modeling import VisionTransformer, CONFIGS
from data_loader import data_load
from model_loader import model_load
from IPython import embed
from datetime import datetime


logging.basicConfig(level=os.environ.get("LOGLEVEL","DEBUG"),
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")
file_handler = logging.FileHandler('logs_test.log')
log.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
file_handler.setFormatter(formatter)



def accuracy(net, loader, device, model_name):
    logging.info("Computing accuracy...")
    net.eval(); correct = tot = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader):
            xb, yb, xb[0]= xb.to(device), yb.to(device), xb[0].to(device)   
            if model_name == "vit" :
                # xb = tuple(x.to(device) for x in xb)
                pred = net(xb)[0].argmax(1) #fot vit
            elif model_name == "cnn":
                pred = net(xb).argmax(1) # for cnn
            correct += (pred == yb).sum().item()
            tot += yb.size(0)
    return 100 * correct / tot

def train_epochs(net, loader, lr, epochs, params=None, device="cpu",model_name="vit"):
    logging.info(f"Training for {epochs} epochs with lr={lr}")
    if params is None: 
        params = net.parameters()
    opt = optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad()
            if model_name == "vit" :
                ce(net(xb)[0], yb).backward()
                
            elif model_name == "cnn":
                ce(net(xb), yb).backward()
                
            opt.step()

# ---------- main --------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="Python file with nn.Module")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pth or bin")
    ap.add_argument("--out-dir", default=".", help="Output folder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model_type", default="cnn", help="choose between 'vit' or 'cnn'")
    args = ap.parse_args()

    device = torch.device(args.device)
    model_name=args.model_type
    model = model_load(args.arch, args.ckpt, model_name,device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print("model in evel state")

    tr_loader,te_loader=data_load(model_name)
    # -------- baseline --------------------------------------------------- #
    acc_orig = accuracy(model, te_loader, device,model_name)
    log.info(f"Baseline accuracy {acc_orig:.2f}%")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "baseline.pth"))
    

    # -------- capture Hessian diag -------------------------------------- #
    log.info("Capturing Hessian diagonal (inv) for all Linear layers...")
    print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    h_diag = capture_h_inv_diag(model, tr_loader, device)
    log.info("Hessian diagonal captured for all Linear layers.")
    print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
    for n, d in h_diag.items():
        log.info(f"Layer {n}: Hessian diag shape {d.shape}, dtype {d.dtype}")

    # -------- CRVQ quantisation ---------------------------------------- #
    crvq = CRVQ(d=8, e=8, m=2, lam=0.2)
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            print("TIME STAMP CRVQ STEP start: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
            m.weight.data = crvq.quantise_layer(n, m.weight.data,
                                                h_diag.get(n))
            log.info(f"n:{n}, m: {m}, m.weight.data.shape:{m.weight.data.shape}, h_diag:{(h_diag.get(n).shape if h_diag.get(n) is not None else None)}")
            print("TIME STAMP: ",datetime.now().strftime("-%m-%d %H:%M:%S"))
    model=model.to(device)
    acc_nft = accuracy(model, te_loader, device,model_name)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "crvq_nft.pth"))
    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"No-FT accuracy {acc_nft:.2f}%")

    #-------- block fine-tune (all Linear) ------------------------------ #
    lin_params = [p for m in model.modules() if isinstance(m, nn.Linear)
                  for p in m.parameters()]

    train_epochs(model, tr_loader, lr=1e-4, epochs=1, params=lin_params,device=device,model_name=model_name)

    acc_blk = accuracy(model, te_loader, device,model_name)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "crvq_block.pth"))
    log.info(f"Block-FT accuracy {acc_blk:.2f}%")

    # -------- end-to-end fine-tune -------------------------------------- #
    train_epochs(model, tr_loader, lr=1e-4, epochs=1, device=device,model_name=model_name)
    acc_e2e = accuracy(model, te_loader, device,model_name)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "crvq_e2e.pth"))
    log.info(f"E2E-FT accuracy {acc_e2e:.2f}%")

    # -------- summary ---------------------------------------------------- #
    print("***********************************************************************************************")
    log.info(f"Summary | orig {acc_orig:.2f} → nft {acc_nft:.2f} → blk {acc_blk:.2f} → e2e {acc_e2e:.2f}")
    print("***********************************************************************************************")

if __name__ == "__main__":
    main()
