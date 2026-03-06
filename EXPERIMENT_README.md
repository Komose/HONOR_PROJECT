# RSNA Adversarial Attack Experiments

## 📋 Project Overview

**Title**: Lesion-Targeted vs Full-Image Adversarial Attacks on CheXzero
**Dataset**: RSNA Pneumonia Detection Challenge
**Model**: CheXzero (CLIP-based Medical Imaging Foundation Model)
**Date**: February 2026

---

## 🎯 Research Question

**Does targeting lesion regions specifically make adversarial attacks more efficient than attacking the entire image?**

---

## 📊 Experimental Setup

### Dataset
- **Source**: RSNA Pneumonia Detection Challenge
- **Samples**: 200 pneumonia-positive chest X-rays with lesion bounding boxes
- **Lesion Statistics**:
  - Average lesions per patient: 1.55
  - Average lesion coverage: 12.06% of image area
  - Lesion distribution: 1 lesion (92), 2 lesions (105), 3 lesions (3)

### Model
- **Architecture**: CheXzero (CLIP ViT-B/12, fine-tuned on MIMIC-CXR)
- **Task**: Binary pneumonia classification (Pneumonia vs Normal)
- **Baseline Performance**: 97.5% accuracy, mean prediction 0.516

### Preprocessing
- **Input Size**: 224×224
- **Normalization**: CLIP standard (mean=[0.48, 0.46, 0.41], std=[0.27, 0.26, 0.28])
- **Format**: DICOM → RGB → Normalized tensor

### Attack Methods

#### FGSM (Fast Gradient Sign Method)
- **Epsilon values**: 4/255, 8/255, 16/255
- **Single-step gradient-based attack**

#### PGD (Projected Gradient Descent)
- **Epsilon**: 8/255
- **Step size (α)**: 2/255
- **Iterations**: 10, 20, 40
- **Multi-step iterative attack**

### Attack Modes

**Lesion Attack**:
- Perturbations applied only within lesion bounding boxes
- Background and non-lesion regions remain unchanged
- Mask-based constraint

**Full Image Attack**:
- Perturbations applied across entire image
- Traditional adversarial attack (baseline)

---

## 🔬 Key Results

### Success Rates (PGD-40)
| Attack Mode | Success Rate | Number Successful |
|-------------|--------------|-------------------|
| Lesion Attack | **59.5%** | 119/200 |
| Full Image Attack | **97.5%** | 195/200 |
| **Ratio** | **1.64×** | Full/Lesion |

### Perturbation Magnitude (L2 Norm)
| Attack Mode | Mean L2 Norm | Modified Pixels |
|-------------|--------------|-----------------|
| Lesion Attack | **3.53** | 17,444 (11.6%) |
| Full Image Attack | **9.96** | 142,121 (94.4%) |
| **Ratio** | **2.82×** | **8.15×** (Full/Lesion) |

### Attack Efficiency
| Attack Mode | Success Rate / L2 Norm |
|-------------|------------------------|
| Lesion Attack | **16.87%** per unit L2 |
| Full Image Attack | **9.79%** per unit L2 |
| **Advantage** | **1.72× more efficient** (Lesion) |

---

## 💡 Key Findings

### 1. Full-Image Attacks Are More Effective
- **97.5% vs 59.5%** success rate
- Model relies on **global image features**, not just lesion regions
- Lesion-only perturbations insufficient to fool the model consistently

### 2. Lesion Attacks Are More Efficient
- **2.8× smaller** L2 perturbations
- **8× fewer** pixels modified
- Higher perturbation efficiency but lower overall success

### 3. PGD >> FGSM
- PGD (iterative): 59.5% success (lesion), 97.5% (full)
- FGSM (single-step): 16.0% success (lesion), 39.5% (full)
- **Iterative optimization crucial** for effective attacks

### 4. Trade-Off: Stealth vs Success
| Metric | Lesion Attack | Full Attack |
|--------|---------------|-------------|
| **Stealth** | ✅ Higher (fewer pixels) | ❌ Lower (many pixels) |
| **Success** | ❌ Lower (59.5%) | ✅ Higher (97.5%) |
| **Efficiency** | ✅ Higher (per L2) | ❌ Lower (per L2) |

---

## 📈 Clinical Implications

### For Model Developers
1. **Lesion regions alone are insufficient** for robust pneumonia detection
2. Models rely on global context → need to strengthen local feature importance
3. Consider architectural changes to focus more on pathological regions

### For Security Analysis
1. **High-success attacks are easily detectable** (modify 94% of pixels)
2. **Stealthy attacks are less reliable** (only 60% success)
3. Trade-off between attack effectiveness and detectability

### For Deployment
1. Input validation and anomaly detection recommended
2. Human-in-the-loop for high-stakes decisions
3. Ensemble with lesion-focused models may improve robustness

---

## 📂 File Structure

```
HONER_PROJECT/
├── dataset/
│   └── rsna/
│       ├── rsna_200_samples.h5              # Preprocessed images
│       ├── rsna_200_lesion_info.json        # Lesion metadata
│       └── selected_200_patients.csv        # Patient IDs
├── results/
│   ├── baseline/
│   │   ├── chexzero_rsna_baseline.csv      # Clean predictions
│   │   └── baseline_summary.json           # Baseline metrics
│   └── attacks_full/
│       ├── attack_comparison_plots.png      # Visualizations
│       ├── ANALYSIS_REPORT.txt              # Detailed analysis
│       ├── attack_summaries.csv             # Summary statistics
│       └── [attack]_[mode]_results.csv      # Per-sample results
├── rsna_attack_framework.py                 # Core attack implementation
├── run_rsna_attacks.py                      # Experiment runner
├── analyze_results.py                       # Analysis and visualization
└── prepare_rsna_data.py                     # Data preprocessing
```

---

## 🔧 Reproducing Results

### 1. Prepare Data
```bash
python prepare_rsna_data.py
```

### 2. Verify Baseline
```bash
python test_chexzero_rsna.py
```

### 3. Run Attacks
```bash
python run_rsna_attacks.py --batch_size 8 --output_dir results/attacks_full
```

### 4. Generate Analysis
```bash
python analyze_results.py
```

---

## 📚 Key Parameters

### FGSM
```python
epsilon: [4/255, 8/255, 16/255]  # Perturbation budget
targeted: False                   # Untargeted attack (reduce probability)
```

### PGD
```python
epsilon: 8/255                    # L-infinity constraint
alpha: 2/255                      # Step size
num_steps: [10, 20, 40]          # Iterations
targeted: False                   # Untargeted attack
```

### Text Templates (Improved)
```python
positive: "Pneumonia"
negative: "Normal"  # Changed from "No Pneumonia" for better discrimination
```

---

## 📊 Complete Results Summary

| Attack | Mode | Success Rate | L2 Norm | L∞ Norm | Pixels Modified |
|--------|------|--------------|---------|---------|-----------------|
| FGSM-4 | Lesion | 9.5% | 1.94 | 0.0157 | 18,035 (12%) |
| FGSM-4 | Full | 42.5% | 6.03 | 0.0157 | 147,642 (98%) |
| FGSM-8 | Lesion | 14.5% | 3.88 | 0.0314 | 18,035 (12%) |
| FGSM-8 | Full | 51.0% | 12.05 | 0.0314 | 147,642 (98%) |
| FGSM-16 | Lesion | 16.0% | 7.77 | 0.0627 | 18,035 (12%) |
| FGSM-16 | Full | 39.5% | 24.11 | 0.0627 | 147,642 (98%) |
| PGD-10 | Lesion | 55.5% | 3.33 | 0.0314 | 16,535 (11%) |
| PGD-10 | Full | **97.5%** | 8.98 | 0.0314 | 129,668 (86%) |
| PGD-20 | Lesion | 58.0% | 3.47 | 0.0314 | 17,122 (11%) |
| PGD-20 | Full | **97.5%** | 9.61 | 0.0314 | 137,224 (91%) |
| PGD-40 | Lesion | 59.5% | 3.53 | 0.0314 | 17,444 (12%) |
| PGD-40 | Full | **97.5%** | 9.96 | 0.0314 | 142,121 (94%) |

---

## 🔗 References

- **CheXzero**: Tiu et al., "Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning", Nature Biomedical Engineering, 2022
- **RSNA Dataset**: RSNA Pneumonia Detection Challenge, Kaggle 2018
- **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
- **PGD**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018

---

## ⚙️ Hardware & Software

- **GPU**: NVIDIA GeForce RTX 3060 (6GB VRAM)
- **CUDA**: 12.4
- **PyTorch**: 2.6.0+cu124
- **Python**: 3.12
- **Total Computation Time**: ~3.2 minutes (200 samples, 12 attack configurations)

---

## 📧 Contact

For questions about this experiment, please refer to the main dissertation document or contact the project supervisor.

---

**Last Updated**: February 23, 2026
