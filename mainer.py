# -*- coding: utf-8 -*-
"""
main.py - end-to-end pipeline:
 • load architecture + weights
 • baseline accuracy
 • CRVQ quantise all Linear layers
 • no-FT, block-FT, E2E-FT accuracies
 • compression summary
"""

import os, logging, importlib.util, argparse, torch
import torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules import capture_h_inv_diag
from crvq import CRVQ

logging.basicConfig(level=os.environ.get("LOGLEVEL","DEBUG"),
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

# ---------- utilities ---------------------------------------------------- #
def load_architecture(py_file: str):
    spec = importlib.util.spec_from_file_location("arch", py_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for obj in vars(mod).values():
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return obj
    raise RuntimeError("No nn.Module subclass found in architecture file.")

def accuracy(net, loader, device):
    net.eval(); correct=tot=0
    with torch.no_grad():
        for xb,yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb).argmax(1)
            correct += (pred==yb).sum().item()
            tot += yb.size(0)
    return 100*correct/tot

def train_epochs(net, loader, lr, epochs, params=None):
    if params is None: params = net.parameters()
    opt = optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for xb,yb in loader:
            opt.zero_grad(); ce(net(xb), yb).backward(); opt.step()

# ---------- main --------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="Python file with nn.Module")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pth")
    ap.add_argument("--out-dir", default=".", help="Output folder")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ModelCls = load_architecture(args.arch)
    model = ModelCls().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # MNIST loaders (default demo – replace for other datasets)
    tf = transforms.ToTensor()
    tr_loader = DataLoader(datasets.MNIST('./data',train=True,download=True,transform=tf),
                           batch_size=256, shuffle=True)
    te_loader = DataLoader(datasets.MNIST('./data',train=False,transform=tf),
                           batch_size=256)

    # -------- baseline --------------------------------------------------- #
    acc_orig = accuracy(model, te_loader, device)
    log.info(f"Baseline accuracy {acc_orig:.2f}%")
    os.makedirs(args.out_dir,exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"baseline.pth"))

    # -------- capture Hessian diag -------------------------------------- #
    h_diag = capture_h_inv_diag(model, tr_loader, device)

    # -------- CRVQ quantisation ----------------------------------------- #
    crvq = CRVQ(d=8,e=8,m=2,lam=0.02)
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = crvq.quantise_layer(n, m.weight.data,
                                                h_diag.get(n))
    acc_nft = accuracy(model, te_loader, device)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"crvq_nft.pth"))
    os.makedirs(args.out_dir,exist_ok=True)
    log.info(f"No-FT accuracy {acc_nft:.2f}%")

    #-------- block fine-tune (all Linear) ------------------------------ #
    lin_params = [p for m in model.modules() if isinstance(m,nn.Linear)
                  for p in m.parameters()]
    train_epochs(model, tr_loader, lr=1e-4, epochs=1, params=lin_params)
    acc_blk = accuracy(model, te_loader, device)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"crvq_block.pth"))
    log.info(f"Block-FT accuracy {acc_blk:.2f}%")

    # -------- end-to-end fine-tune -------------------------------------- #
    train_epochs(model, tr_loader, lr=1e-4, epochs=1)
    acc_e2e = accuracy(model, te_loader, device)
    torch.save(model.state_dict(), os.path.join(args.out_dir,"crvq_e2e.pth"))
    log.info(f"E2E-FT accuracy {acc_e2e:.2f}%")

    # -------- summary ---------------------------------------------------- #
    log.info(f"Summary | orig {acc_orig:.2f} → nft {acc_nft:.2f} → blk {acc_blk:.2f} → e2e {acc_e2e:.2f}")

if __name__ == "__main__":
    main()
