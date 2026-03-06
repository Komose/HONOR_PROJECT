#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_attack_pgd.py

End-to-end example runner for your refactored unified-API framework:
  Data (HDF5 + CSV labels) -> Model -> FMAttackWrapper -> PGD(Linf) -> Metrics/Save

Expected project layout (as in your screenshot):
refactored_code/
  attacks/
    __init__.py
    pgd_linf.py
    fgsm_linf.py
    deepfool_l2.py
    ...
  fm_attack_wrapper.py
  run_attack_pgd.py   <-- put this file here

Example:
  python run_attack_pgd.py ^
    --model_path models/chexzero.pt ^
    --h5_path data/test_cxr.h5 ^
    --labels_csv data/final_paths.csv ^
    --pretrained false ^
    --epsilon 8 --step_size 2 --steps 10 --random_start true ^
    --batch_size 16 ^
    --out_dir outputs/pgd_eps8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import Dataset, DataLoader

# ---- your modules ----
from model import CLIP  # your CheXzero-style CLIP implementation
import clip as openai_clip  # your local clip.py (OpenAI-CLIP loader/tokenize)

from fm_attack_wrapper import FMAttackWrapper, InputDomain
from attacks.pgd_linf import PGDLinf


# 14-label CheXpert-style list (must match your evaluation)
CXR_LABELS: List[str] = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema',
    'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
    'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
    'Pneumothorax','Support Devices'
]


class CXRH5Dataset(Dataset):
    """
    Reads:
      - images from an HDF5 dataset named 'cxr' (shape: N x H x W, grayscale)
      - labels from a CSV (same format as your final_paths.csv usage in zero_shot.py)
    Returns:
      x: FloatTensor [3,H,W]  (pixel domain by default)
      y: FloatTensor [14]     (0/1 multi-label)
    """
    def __init__(self, h5_path: str, labels_csv: str, *, labels: List[str] = CXR_LABELS):
        super().__init__()
        self.h5_path = str(h5_path)
        self.labels_csv = str(labels_csv)
        self.labels = list(labels)

        # load labels once
        df = pd.read_csv(self.labels_csv)

        # Many versions of final_paths.csv have a first column as path/index; drop it if needed.
        # Keep only the 14 columns in CXR_LABELS if present.
        if all(l in df.columns for l in self.labels):
            y = df[self.labels].to_numpy()
        else:
            # fallback: drop the first column and assume remaining columns are the labels
            df2 = df.copy()
            df2.drop(df2.columns[0], axis=1, inplace=True)
            y = df2.to_numpy()

        self.y = y.astype(np.float32)

        # open HDF5 lazily per worker
        self._h5 = None
        self._dset = None

    def _lazy_init(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._dset = self._h5["cxr"]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        self._lazy_init()
        img = self._dset[idx]  # (H,W) grayscale, usually uint8/float in pixel domain

        # Convert to float32 tensor, channelize: (1,H,W) -> (3,H,W)
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        x = torch.from_numpy(img)  # [3,H,W]

        y = torch.from_numpy(self.y[idx])  # [14]
        return x, y


def str2bool(x: str) -> bool:
    return x.lower() in ("1", "true", "t", "yes", "y")


def load_clip_model(model_path: str, *, pretrained: bool, context_length: int, device: torch.device) -> torch.nn.Module:
    """
    Mirrors your zero_shot.py load_clip().
    - pretrained=False: instantiate CLIP with CheXzero params (image_resolution=320).
    - pretrained=True:  load OpenAI CLIP ViT-B/32 (jit=False).
    """
    if not pretrained:
        params = dict(
            embed_dim=768,
            image_resolution=320,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            context_length=context_length,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
        model = CLIP(**params)
    else:
        model, _ = openai_clip.load("ViT-B/32", device=device, jit=False)

    sd = torch.load(model_path, map_location=device)
    # Some checkpoints are stored as {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def batch_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Mean AUROC across labels. Skips labels with all-0 or all-1 ground truth.
    """
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if np.all(col == 0) or np.all(col == 1):
            continue
        aucs.append(roc_auc_score(col, y_prob[:, j]))
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--h5_path", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)

    ap.add_argument("--pretrained", type=str, default="false")
    ap.add_argument("--context_length", type=int, default=77)

    # Attack params (pixel domain by default; e.g., epsilon=8 means 8 gray-levels)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--step_size", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--random_start", type=str, default="true")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    # Wrapper preprocessing params: match your CLIP preprocess + 14-label eval
    ap.add_argument("--mean", type=float, nargs=3, default=(101.48761, 101.48761, 101.48761))
    ap.add_argument("--std", type=float, nargs=3, default=(83.43944, 83.43944, 83.43944))
    ap.add_argument("--input_resolution", type=int, default=320)
    ap.add_argument("--resize_mode", type=str, default="none", choices=["none", "bilinear", "bicubic"])

    ap.add_argument("--out_dir", type=str, default="outputs/pgd")
    ap.add_argument("--save_adv", type=str, default="false", help="Whether to save x_adv to out_dir (pt file).")

    args = ap.parse_args()

    pretrained = str2bool(args.pretrained)
    random_start = str2bool(args.random_start)
    save_adv = str2bool(args.save_adv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load model
    model = load_clip_model(args.model_path, pretrained=pretrained, context_length=args.context_length, device=device)

    # 2) Build wrapper (this caches text features; forward_logits becomes plug-and-play)
    # input_domain defaults to pixel [0,255] which matches your current Normalize(mean=101, std=83) pipeline.
    wrapper = FMAttackWrapper(
        model,
        class_names=CXR_LABELS,
        pos_template="{}",
        neg_template="no {}",
        context_length=args.context_length,
        device=device,
        input_resolution=args.input_resolution,
        resize_mode=args.resize_mode,
        mean=tuple(args.mean),
        std=tuple(args.std),
        input_domain=InputDomain(value_range=(0.0, 255.0), space="pixel", layout="NCHW"),
        clamp=True,
        tokenizer=None,  # use clip.tokenize from your local clip.py
    )

    # 3) DataLoader
    dset = CXRH5Dataset(args.h5_path, args.labels_csv, labels=CXR_LABELS)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4) Attack instance
    attack = PGDLinf(
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        random_start=random_start,
    )

    # 5) Run clean + adv
    all_y = []
    all_prob_clean = []
    all_prob_adv = []
    all_success = []
    all_linf = []
    all_l2 = []

    adv_batches = [] if save_adv else None

    from tqdm import tqdm
    for x, y in tqdm(loader, desc="PGD attack"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # clean probs
        with torch.no_grad():
            logits_clean = wrapper.forward_logits(x)              # [B,14]
            prob_clean = torch.sigmoid(logits_clean)              # [B,14]

        # adversarial
        out = attack.perturb(x, y, wrapper)
        x_adv = out["x_adv"]

        with torch.no_grad():
            logits_adv = wrapper.forward_logits(x_adv)
            prob_adv = torch.sigmoid(logits_adv)

        # collect
        all_y.append(y.detach().cpu().numpy())
        all_prob_clean.append(prob_clean.detach().cpu().numpy())
        all_prob_adv.append(prob_adv.detach().cpu().numpy())

        meta = out.get("meta", {})
        if "success" in meta:
            all_success.append(np.asarray(meta["success"], dtype=np.float32))
        if "linf" in meta:
            all_linf.append(np.asarray(meta["linf"], dtype=np.float32))
        if "l2" in meta:
            all_l2.append(np.asarray(meta["l2"], dtype=np.float32))

        if save_adv:
            adv_batches.append(x_adv.detach().cpu())

    y_np = np.concatenate(all_y, axis=0)
    p_clean = np.concatenate(all_prob_clean, axis=0)
    p_adv = np.concatenate(all_prob_adv, axis=0)

    auc_clean = batch_auc(y_np, p_clean)
    auc_adv = batch_auc(y_np, p_adv)

    # success rate: label-set changed (multi-label). If not present, leave as NaN.
    if len(all_success):
        succ = np.concatenate(all_success, axis=0)
        asr = float(np.mean(succ))
    else:
        asr = float("nan")

    linf = float(np.mean(np.concatenate(all_linf))) if len(all_linf) else float("nan")
    l2 = float(np.mean(np.concatenate(all_l2))) if len(all_l2) else float("nan")

    summary = {
        "attack": "PGD_Linf",
        "epsilon": args.epsilon,
        "step_size": args.step_size,
        "steps": args.steps,
        "random_start": random_start,
        "pretrained": pretrained,
        "auc_clean_mean": auc_clean,
        "auc_adv_mean": auc_adv,
        "auc_drop": float(auc_clean - auc_adv) if np.isfinite(auc_clean) and np.isfinite(auc_adv) else float("nan"),
        "asr_label_set_changed": asr,
        "mean_linf": linf,
        "mean_l2": l2,
        "n": int(y_np.shape[0]),
    }

    (out_dir / "summary.json").write_text(
        str(summary).replace("'", '"'), encoding="utf-8"
    )
    np.save(out_dir / "probs_clean.npy", p_clean)
    np.save(out_dir / "probs_adv.npy", p_adv)
    np.save(out_dir / "y_true.npy", y_np)

    if save_adv and adv_batches is not None:
        x_adv_all = torch.cat(adv_batches, dim=0)  # [N,3,H,W]
        torch.save(x_adv_all, out_dir / "x_adv.pt")

    print("\n=== PGD run done ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
