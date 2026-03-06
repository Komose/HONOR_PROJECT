#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_chexpert_subset.py

Scan your CheXpert train/ folder structure and create a *patient-level* subset CSV
that contains:
  - image_path (absolute path)
  - 14 CheXpert labels (0/1) aligned with your evaluation pipeline

Example (Windows):
  python prepare_chexpert_subset.py ^
    --train_dir "D:\PycharmProjects\HONER_PROJECT\dataset\CheXpert\1\train" ^
    --labels_csv "D:\PycharmProjects\HONER_PROJECT\dataset\CheXpert\1\train.csv" ^
    --out_csv "D:\PycharmProjects\HONER_PROJECT\dataset\CheXpert\subset_pgd_200.csv" ^
    --num_patients 200 ^
    --select random ^
    --seed 42 ^
    --uncertain_policy u_zero

Notes:
- We pick *one* frontal image per patient by default (study1 preferred, then lowest study index).
- labels_csv should contain a "Path" column whose paths include ".../train/patientXXXXX/studyY/view1_frontal.jpg".
- uncertain_policy:
    u_zero: map -1 -> 0   (common "U-Zero" policy)
    u_one : map -1 -> 1   (U-One)
    keep  : keep -1 as -1 (only if you handle it later)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CXR_LABELS_14 = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Effusion","Pleural Other","Pneumonia","Pneumothorax",
    "Support Devices","No Finding",
]

_PATIENT_RE = re.compile(r"patient\d{5}", re.IGNORECASE)
_STUDY_RE = re.compile(r"study(\d+)", re.IGNORECASE)

def key_from_path(p: str) -> str:
    """Extract join-key starting at patientXXXXX/..."""
    p_norm = p.replace("\\", "/")
    m = _PATIENT_RE.search(p_norm)
    if not m:
        return p_norm
    return p_norm[m.start():]

def find_frontal_per_patient(train_dir: Path) -> Dict[str, Path]:
    """patient_id -> chosen frontal image path (1 per patient)."""
    patient_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir() and p.name.lower().startswith("patient")])
    out: Dict[str, Path] = {}

    def study_idx(s: Path) -> int:
        m = _STUDY_RE.match(s.name)
        return int(m.group(1)) if m else 10**9

    for pdir in patient_dirs:
        studies = [s for s in pdir.iterdir() if s.is_dir() and s.name.lower().startswith("study")]
        if not studies:
            continue
        studies_sorted = sorted(studies, key=study_idx)

        chosen_study = None
        for s in studies_sorted:
            if s.name.lower() == "study1":
                chosen_study = s
                break
        if chosen_study is None:
            chosen_study = studies_sorted[0]

        img = chosen_study / "view1_frontal.jpg"
        if img.exists():
            out[pdir.name] = img
        else:
            candidates = sorted(chosen_study.glob("*frontal*.jpg"))
            if candidates:
                out[pdir.name] = candidates[0]
    return out

def apply_uncertain_policy(df: pd.DataFrame, policy: str, label_cols: List[str]) -> pd.DataFrame:
    if policy == "u_zero":
        df[label_cols] = df[label_cols].replace(-1.0, 0.0)
    elif policy == "u_one":
        df[label_cols] = df[label_cols].replace(-1.0, 1.0)
    elif policy == "keep":
        pass
    else:
        raise ValueError(f"Unknown uncertain_policy: {policy}")
    df[label_cols] = df[label_cols].fillna(0.0)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--num_patients", type=int, default=50)
    ap.add_argument("--select", type=str, choices=["first", "random"], default="first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uncertain_policy", type=str, choices=["u_zero", "u_one", "keep"], default="u_zero")
    ap.add_argument("--include_no_finding", type=str, default="true")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    labels_csv = Path(args.labels_csv)
    out_csv = Path(args.out_csv)
    include_no_finding = args.include_no_finding.lower() in ("1","true","t","yes","y")

    patient_to_img = find_frontal_per_patient(train_dir)
    patients = sorted(patient_to_img.keys())
    if not patients:
        raise RuntimeError(f"No patients found under: {train_dir}")

    if args.num_patients > len(patients):
        print(f"[WARN] requested {args.num_patients} but only found {len(patients)}. Using all.")
        chosen = patients
    else:
        if args.select == "first":
            chosen = patients[:args.num_patients]
        else:
            rng = np.random.default_rng(args.seed)
            chosen = list(rng.choice(patients, size=args.num_patients, replace=False))

    chosen_imgs = [patient_to_img[p] for p in chosen]

    df = pd.read_csv(labels_csv)
    if "Path" not in df.columns:
        raise RuntimeError("labels_csv must contain a 'Path' column (CheXpert official format).")

    label_cols = [c for c in CXR_LABELS_14 if c in df.columns]
    if not include_no_finding and "No Finding" in label_cols:
        label_cols = [c for c in label_cols if c != "No Finding"]
    if len(label_cols) < 10:
        raise RuntimeError(f"Not enough label columns found. Found: {label_cols}")

    df_small = df[["Path"] + label_cols].copy()
    df_small["key"] = df_small["Path"].astype(str).apply(key_from_path)
    df_small = apply_uncertain_policy(df_small, args.uncertain_policy, label_cols)
    df_small = df_small.drop(columns=["Path"])

    subset = pd.DataFrame({"image_path": [str(p) for p in chosen_imgs]})
    subset["key"] = subset["image_path"].astype(str).apply(key_from_path)

    merged = subset.merge(df_small, on="key", how="left")
    missing = merged[label_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"[WARN] {missing} images missing labels after merge; filling with 0.")
        merged[label_cols] = merged[label_cols].fillna(0.0)

    merged = merged.drop(columns=["key"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv}")
    print(f"rows={len(merged)}  labels={label_cols}")

if __name__ == "__main__":
    main()
