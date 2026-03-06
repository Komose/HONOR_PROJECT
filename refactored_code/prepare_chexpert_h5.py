#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_chexpert_h5.py

Convert a subset CSV (from prepare_chexpert_subset.py) into HDF5 with key 'cxr'
so it matches your existing run_attack_pgd.py runner.

Example:
  python prepare_chexpert_h5.py ^
    --subset_csv "D:\...\subset_pgd_200.csv" ^
    --out_h5 "D:\...\subset_pgd_200.h5" ^
    --image_size 320
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from PIL import Image
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_csv", type=str, required=True)
    ap.add_argument("--out_h5", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=320, help="Resize to square size. Use 0 to keep original.")
    args = ap.parse_args()

    subset_csv = Path(args.subset_csv)
    out_h5 = Path(args.out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subset_csv)
    if "image_path" not in df.columns:
        raise RuntimeError("subset_csv must contain 'image_path' column.")
    paths = df["image_path"].astype(str).tolist()

    imgs = []
    for p in tqdm(paths, desc="Loading images"):
        img = Image.open(p).convert("L")
        if args.image_size and args.image_size > 0:
            img = img.resize((args.image_size, args.image_size), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32)  # [H,W] in [0,255]
        imgs.append(arr)

    x = np.stack(imgs, axis=0)  # (N,H,W)

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("cxr", data=x, dtype="float32", compression="gzip")
    print(f"[OK] wrote HDF5: {out_h5}  key='cxr'  shape={x.shape}")

if __name__ == "__main__":
    main()
