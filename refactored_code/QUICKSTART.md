# Quick Start Guide

Get your lesion-aware attack experiments running in 5 minutes.

---

## Installation

```bash
# Clone/navigate to project
cd D:\PycharmProjects\HONER_PROJECT\refactored_code

# Install dependencies
pip install -r requirements.txt
```

---

## Verify Installation

Run the test script to check all components:

```bash
python test_pipeline.py --model_path "path/to/chexzero.pt"
```

**Expected output:** ✅ ALL TESTS PASSED SUCCESSFULLY!

---

## Run Your First Experiment

### Step 1: Prepare Data

```bash
python prepare_chexpert_h5.py \
    --subset_csv "D:\data\chexpert\test.csv" \
    --out_h5 "D:\data\chexpert\test.h5" \
    --image_size 320
```

### Step 2: Run Lesion-Aware PGD Attack

```bash
python run_lesion_aware_attack.py \
    --model_path "models/chexzero.pt" \
    --h5_path "data/test.h5" \
    --labels_csv "data/test.csv" \
    --attack pgd \
    --epsilon 8 \
    --steps 10 \
    --mask_threshold 0.5 \
    --run_baseline true \
    --out_dir "outputs/my_first_experiment"
```

### Step 3: View Results

Check `outputs/my_first_experiment/`:
- `summary.json` - Quantitative results
- `visualizations/` - Visual comparisons

---

## Key Parameters

| Parameter | Typical Values | Description |
|-----------|---------------|-------------|
| `--attack` | `pgd`, `cw` | Attack method |
| `--epsilon` | `2`, `4`, `8`, `16` | L∞ perturbation budget (pixels) |
| `--mask_threshold` | `0.3` - `0.7` | Grad-CAM binarization threshold |
| `--steps` | `10`, `20`, `40` | PGD iterations |

---

## Example Experiments

### Compare Attack Methods
```bash
# PGD
python run_lesion_aware_attack.py --attack pgd --epsilon 8 --out_dir outputs/pgd

# C&W
python run_lesion_aware_attack.py --attack cw --out_dir outputs/cw
```

### Epsilon Sweep
```bash
for eps in 2 4 8 16; do
    python run_lesion_aware_attack.py \
        --attack pgd --epsilon $eps \
        --out_dir "outputs/pgd_eps${eps}"
done
```

### Visualizations Only
```bash
python run_lesion_aware_attack.py \
    --attack pgd --epsilon 8 \
    --batch_size 1 --num_vis 50 \
    --out_dir "outputs/visualizations"
```

---

## Troubleshooting

**CUDA out of memory:** Reduce `--batch_size` to 4 or 2

**Attack ASR = 0%:** Increase `--epsilon` or `--steps`

**Blank masks:** Check model is in eval mode, verify preprocessing

---

## Next Steps

1. ✅ Run test script
2. ✅ Run first experiment
3. 📊 Analyze results in `summary.json`
4. 📈 Run multiple experiments (epsilon sweep, attack comparison)
5. 📝 Compile results for dissertation (Objective 7)

---

## Full Documentation

See `README.md` for comprehensive documentation.

**Questions?** Check the dissertation project plan or contact your supervisor.
