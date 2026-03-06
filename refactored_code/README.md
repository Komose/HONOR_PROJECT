# Lesion-Aware Adversarial Attacks on Medical Imaging Foundation Models

**Project:** HONER_PROJECT - Dissertation Research
**Objective:** Systematically investigate how lesion-aware adversarial perturbations affect the decision-making and adversarial robustness of medical imaging foundation models (CheXzero on CheXpert dataset).

---

## 📋 Project Overview

This codebase implements **Objectives 4-6** of the dissertation project:

- ✅ **Objective 4:** Unified experimental framework integrating model, data, and attack APIs
- ✅ **Objective 5:** Lesion-aware adversarial attacks using Grad-CAM-based spatial masks
- ✅ **Objective 6:** Comparative evaluation of masked vs. non-masked attacks

### Key Innovation

Traditional adversarial attacks apply **global perturbations** (constrained by ε):
```
δ : ||δ||_∞ ≤ ε  (entire image)
```

Lesion-aware attacks apply **spatially-constrained perturbations** focused on clinically-relevant regions:
```
δ_masked = δ ⊙ M, ||δ||_∞ ≤ ε  (only in lesion mask M)
```

where M is a binary mask generated from Grad-CAM attention heatmaps.

---

## 📁 Project Structure

```
refactored_code/
├── attacks/                     # Adversarial attack implementations
│   ├── __init__.py             # Unified API exports
│   ├── base.py                 # AttackBase protocol
│   ├── pgd_linf.py             # PGD (L∞)
│   ├── cw_l2.py                # Carlini & Wagner (L2)
│   ├── fgsm_linf.py            # FGSM (L∞)
│   ├── deepfool_l2.py          # DeepFool (L2)
│   ├── masked_attacks.py       # Lesion-aware variants (MaskedPGD, MaskedCW, etc.)
│   └── utils.py                # Projection, clamping, metrics
│
├── fm_attack_wrapper.py        # Model wrapper for CheXzero CLIP
├── grad_cam.py                 # Grad-CAM for lesion mask generation
├── evaluation.py               # Robustness metrics (AUROC, ASR, norms)
├── run_attack_pgd.py           # Baseline attack runner (non-masked)
├── run_lesion_aware_attack.py  # Main lesion-aware experiment runner
├── prepare_chexpert_h5.py      # Data preprocessing script
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Installation

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- CUDA (recommended for faster execution)

**Install dependencies:**
```bash
pip install torch torchvision numpy pandas h5py scikit-learn matplotlib tqdm
```

**Project-specific modules:**
- Ensure `model.py` (CheXzero CLIP implementation) is in the same directory
- Ensure `clip.py` (OpenAI CLIP utilities) is available

---

### 2. Data Preparation

Convert CheXpert CSV subset to HDF5 format:

```bash
python prepare_chexpert_h5.py \
    --subset_csv "D:\data\chexpert\subset_test.csv" \
    --out_h5 "D:\data\chexpert\subset_test.h5" \
    --image_size 320
```

**Expected input:**
- `subset_test.csv`: CSV with `image_path` column and 14 CheXpert label columns
- Images should be chest X-rays in any format (PNG, JPG)

**Output:**
- `subset_test.h5`: HDF5 file with dataset key `'cxr'` containing images [N, H, W]

---

### 3. Run Lesion-Aware Attacks

**Main experiment runner** (Objectives 5 & 6):

```bash
python run_lesion_aware_attack.py \
    --model_path "models/chexzero.pt" \
    --h5_path "data/subset_test.h5" \
    --labels_csv "data/subset_test.csv" \
    --attack pgd \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --mask_threshold 0.5 \
    --run_baseline true \
    --out_dir "outputs/lesion_aware_pgd_eps8" \
    --save_vis true \
    --num_vis 20
```

**Key Arguments:**
- `--attack`: Attack type (`pgd`, `cw`, `fgsm`, `deepfool`)
- `--epsilon`: L∞ perturbation budget (pixel domain, e.g., 8/255 for normalized images)
- `--mask_threshold`: Grad-CAM threshold for binarizing lesion masks (0.0-1.0)
- `--run_baseline`: If `true`, also runs non-masked attack for comparison
- `--save_vis`: Save visualizations (clean image, adversarial, CAM, mask)

**Output:**
- `summary.json`: Quantitative results (AUROC, ASR, norms)
- `masked_probs_*.npy`: Raw prediction probabilities
- `visualizations/`: Side-by-side comparisons of clean/adversarial/mask

---

### 4. Run Baseline (Non-Masked) Attacks Only

If you only need standard PGD without lesion-awareness:

```bash
python run_attack_pgd.py \
    --model_path "models/chexzero.pt" \
    --h5_path "data/subset_test.h5" \
    --labels_csv "data/subset_test.csv" \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --out_dir "outputs/baseline_pgd_eps8"
```

---

### 5. Run Multiple Experiments (Epsilon Sweep)

To evaluate robustness across multiple ε values:

```bash
for eps in 2 4 8 16; do
    python run_lesion_aware_attack.py \
        --model_path "models/chexzero.pt" \
        --h5_path "data/subset_test.h5" \
        --labels_csv "data/subset_test.csv" \
        --attack pgd \
        --epsilon $eps \
        --out_dir "outputs/pgd_eps${eps}"
done
```

Aggregate results from multiple `summary.json` files for comparative analysis.

---

## 🧠 Attack Methods Implemented

### Standard Attacks (Global Perturbations)

| Attack | Norm | Description | Key Parameters |
|--------|------|-------------|----------------|
| **PGD** | L∞ | Projected Gradient Descent (Madry et al. 2018) | `epsilon`, `steps`, `step_size` |
| **C&W** | L2 | Carlini & Wagner optimization-based (2017) | `confidence`, `max_iterations`, `learning_rate` |
| **FGSM** | L∞ | Fast Gradient Sign Method (single-step PGD) | `epsilon` |
| **DeepFool** | L2 | Minimal perturbation to decision boundary | `max_iter`, `overshoot` |

### Lesion-Aware Attacks (Spatially-Constrained)

All standard attacks have **masked variants**:
- `MaskedPGDLinf`
- `MaskedCWL2`
- `MaskedFGSMLinf`
- `MaskedDeepFoolL2`

**Masking Strategy:**
1. Generate Grad-CAM heatmap from CheXzero's visual encoder
2. Binarize heatmap at threshold τ (e.g., 0.5) to create mask M
3. Apply perturbations only within M: `δ_final = δ ⊙ M`

---

## 📊 Evaluation Metrics

The `evaluation.py` module computes:

### Classification Performance
- **AUROC (Area Under ROC Curve):** Mean across 14 CheXpert labels
- **AUC Drop:** `AUC_clean - AUC_adv` (higher = attack more effective)

### Attack Effectiveness
- **Attack Success Rate (ASR):** % of samples where label-set prediction changed
- **Mean Labels Flipped:** Average number of labels changed per sample

### Perturbation Magnitude
- **L0 Norm:** Number of perturbed pixels
- **L2 Norm:** Euclidean distance `||δ||_2`
- **L∞ Norm:** Maximum pixel change `||δ||_∞`

### Lesion-Specific Metrics
- **Mask Coverage:** % of image covered by lesion mask
- **Perturbation Concentration:** Ratio of perturbation inside vs. outside mask
- **L∞ Inside/Outside Lesion:** Separate norms for lesion vs. non-lesion regions

---

## 🔬 Experimental Design (Objectives 5 & 6)

### Objective 5: Implement Lesion-Aware Attacks
**Task:** Generate attention masks and apply constrained attacks.

**Validation Criteria:**
- ✅ Grad-CAM successfully generates masks highlighting lesions
- ✅ Perturbations are spatially constrained to masked regions
- ✅ ASR > 0.5 on test set (attacks are effective)

### Objective 6: Compare Masked vs. Non-Masked Attacks
**Task:** Evaluate if lesion-focused perturbations are more effective.

**Hypothesis:** Lesion-aware attacks will achieve:
- Higher ASR (more successful)
- Lower AUC drop per unit L2 norm (more efficient)
- Better transferability across models

**Comparison Metrics:**
- AUC Drop (Baseline) vs. AUC Drop (Masked)
- ASR (Baseline) vs. ASR (Masked)
- L2 Norm (Baseline) vs. L2 Norm (Masked)

---

## 📝 Results Interpretation

### Example Output (`summary.json`)

```json
{
  "attack": "pgd",
  "epsilon": 8,
  "mask_threshold": 0.5,
  "masked_attack": {
    "auc_clean": 0.82,
    "auc_adv": 0.65,
    "auc_drop": 0.17,
    "asr": 0.72,
    "mean_linf": 7.8,
    "mean_l2": 45.3
  },
  "baseline_attack": {
    "auc_clean": 0.82,
    "auc_adv": 0.68,
    "auc_drop": 0.14,
    "asr": 0.65,
    "mean_linf": 7.9,
    "mean_l2": 52.1
  },
  "mean_mask_coverage": 0.23
}
```

**Interpretation:**
- **AUC Drop:** Masked attack (0.17) > Baseline (0.14) → Lesion-focused perturbations more effective
- **ASR:** Masked (0.72) > Baseline (0.65) → Higher success rate for masked attack
- **L2 Norm:** Masked (45.3) < Baseline (52.1) → Masked attack achieves same effect with smaller perturbation
- **Mask Coverage:** 23% → Perturbations concentrated in ~1/4 of image (likely lesion regions)

**Conclusion:** ✅ Lesion-aware attacks are more efficient and effective.

---

## 🧪 Common Use Cases

### 1. Reproduce Paper Results (Li et al., 2025)
```bash
python run_lesion_aware_attack.py \
    --attack pgd \
    --epsilon 8 \
    --mask_threshold 0.5 \
    --run_baseline true \
    --out_dir "outputs/reproduce_li2025"
```

### 2. Ablation Study: Mask Threshold
```bash
for thresh in 0.3 0.4 0.5 0.6 0.7; do
    python run_lesion_aware_attack.py \
        --attack pgd \
        --epsilon 8 \
        --mask_threshold $thresh \
        --out_dir "outputs/ablation_thresh${thresh}"
done
```

### 3. Attack Comparison: PGD vs. C&W
```bash
# PGD
python run_lesion_aware_attack.py --attack pgd --epsilon 8 --out_dir outputs/pgd

# C&W (L2)
python run_lesion_aware_attack.py --attack cw --cw_max_iter 500 --out_dir outputs/cw
```

### 4. Generate Visualizations Only
```bash
python run_lesion_aware_attack.py \
    --attack pgd \
    --epsilon 8 \
    --batch_size 1 \
    --save_vis true \
    --num_vis 50 \
    --out_dir "outputs/visualizations_only"
```

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce `--batch_size` (try 4 or 2)

### Issue: Grad-CAM produces blank masks
**Possible causes:**
- Model not in eval mode → Check `model.eval()`
- Hooks not registered correctly → Verify target layer
- Input preprocessing mismatch → Ensure correct mean/std

**Debug:**
```python
from grad_cam import MultiLabelGradCAM
grad_cam = MultiLabelGradCAM(model)
cam = grad_cam.generate_multilabel_cam(x, text_features, y)
print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")  # Should be [0, 1]
```

### Issue: Attack success rate is 0%
**Possible causes:**
- Epsilon too small → Try increasing `--epsilon`
- PGD steps too few → Increase `--steps` to 20-40
- Model is extremely robust (unlikely for CheXzero)

---

## 📚 References

1. **Madry et al. (2018):** "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
2. **Carlini & Wagner (2017):** "Towards Evaluating the Robustness of Neural Networks" (C&W)
3. **Li et al. (2025):** "LatAtk: A medical image attack method focused on lesion areas" (Lesion-aware attacks)
4. **Selvaraju et al. (2017):** "Grad-CAM: Visual Explanations from Deep Networks" (Grad-CAM)
5. **Tiu et al. (2022):** "CheXzero: Expert-level Zero-shot Chest Radiograph Interpretation" (CheXzero model)
6. **Irvin et al. (2019):** "CheXpert: A Large Chest Radiograph Dataset" (CheXpert dataset)

---

## 🤝 Contributing

This codebase supports **Objectives 4-6** of the dissertation. For Objective 7 (final report), compile results from multiple runs and analyze trends.

**Suggested experiments for dissertation:**
1. ✅ Epsilon sweep: ε ∈ {2, 4, 8, 16} (pixel domain)
2. ✅ Mask threshold sweep: τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
3. ✅ Attack comparison: PGD vs. C&W vs. FGSM vs. DeepFool
4. ✅ Masked vs. non-masked comparison for each attack

**Expected completion:** Week 16 (Objective 5) → Week 1 next semester (Objective 6)

---

## 📧 Contact

For questions or issues, refer to the dissertation supervisor or create an issue in the project repository.

**Author:** HONER_PROJECT Team
**Date:** January 2026
**Version:** 1.0.0
