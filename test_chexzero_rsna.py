"""
Test CheXzero Model on RSNA Pneumonia Dataset - Baseline Performance Evaluation

This script evaluates CheXzero's zero-shot performance on RSNA pneumonia detection
before adversarial attacks.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
import h5py
import json
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')
from model import build_model


class RSNADataset(data.Dataset):
    """
    Dataset for RSNA images stored in HDF5 format (already 3-channel).

    Unlike CXRTestDataset which expects single-channel images,
    this class handles images that are already in (C, H, W) format.
    """
    def __init__(self, img_path: str, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]  # np array, already (3, 224, 224)
        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        return {'img': img}


def load_rsna_data(h5_path, lesion_info_path):
    """
    Load RSNA data and lesion information.

    Returns:
        images: numpy array (N, 3, 224, 224)
        patient_ids: list of patient IDs
        lesion_info: dict of lesion metadata
    """
    print(f"Loading RSNA data from {h5_path}...")

    # Load images
    with h5py.File(h5_path, 'r') as f:
        images = f['cxr'][:]

    # Load lesion info
    with open(lesion_info_path, 'r') as f:
        lesion_data = json.load(f)

    patient_ids = lesion_data['patient_ids']
    lesion_info = lesion_data['lesion_data']

    print(f"  - Loaded {len(images)} images")
    print(f"  - Image shape: {images.shape}")
    print(f"  - Patient IDs: {len(patient_ids)}")

    return images, patient_ids, lesion_info


def create_dataloader(h5_path, batch_size=1):
    """
    Create PyTorch DataLoader for RSNA dataset.

    Note: Uses RSNADataset which handles 3-channel images directly.
    """
    dataset = RSNADataset(img_path=h5_path, transform=None)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def evaluate_chexzero_rsna(
    model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
    h5_path='dataset/rsna/rsna_200_samples.h5',
    lesion_info_path='dataset/rsna/rsna_200_lesion_info.json',
    output_dir='results/baseline'
):
    """
    Evaluate CheXzero on RSNA pneumonia detection.

    Since all RSNA samples are Target=1 (Pneumonia), we expect high predictions.
    """
    print("="*70)
    print("CheXzero Baseline Evaluation on RSNA Pneumonia Dataset")
    print("="*70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    images, patient_ids, lesion_info = load_rsna_data(h5_path, lesion_info_path)

    # Create dataloader
    print("\nCreating dataloader...")
    loader = create_dataloader(h5_path, batch_size=1)
    print(f"  - Batch size: 1")
    print(f"  - Total batches: {len(loader)}")

    # Load CheXzero model
    print("\nLoading CheXzero model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Use build_model to automatically infer model parameters from checkpoint
    model = build_model(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  - Model loaded from: {model_path}")

    # Define label and template for Pneumonia detection
    # CheXzero uses contrastive templates
    # IMPROVED: Use "Normal" instead of "No Pneumonia" for better discrimination
    cxr_labels = ['Pneumonia']
    pos_template = '{}'         # Positive: "Pneumonia"
    neg_template = 'Normal'     # Negative: "Normal" (IMPROVED!)

    print("\nGenerating text embeddings...")
    print(f"  - Labels: {cxr_labels}")
    print(f"  - Positive template: '{pos_template}'")
    print(f"  - Negative template: '{neg_template}'")

    import clip

    # Run predictions with positive template
    print("\nRunning predictions with POSITIVE template...")
    pos_pred = []
    with torch.no_grad():
        # Compute positive text embeddings
        pos_texts = [pos_template.format(label) for label in cxr_labels]
        pos_tokens = clip.tokenize(pos_texts, context_length=77).to(device)
        pos_text_features = model.encode_text(pos_tokens)
        pos_text_features /= pos_text_features.norm(dim=-1, keepdim=True)

        # Get image predictions
        for i, data in enumerate(loader):
            images = data['img'].to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ pos_text_features.T).squeeze().cpu().numpy()
            pos_pred.append(logits)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/200 samples...")

    pos_pred = np.array(pos_pred)

    # Run predictions with negative template
    print("\nRunning predictions with NEGATIVE template...")
    neg_pred = []
    with torch.no_grad():
        # Compute negative text embeddings
        neg_texts = [neg_template]  # Just "Normal", no formatting needed
        neg_tokens = clip.tokenize(neg_texts, context_length=77).to(device)
        neg_text_features = model.encode_text(neg_tokens)
        neg_text_features /= neg_text_features.norm(dim=-1, keepdim=True)

        # Get image predictions
        for i, data in enumerate(loader):
            images = data['img'].to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ neg_text_features.T).squeeze().cpu().numpy()
            neg_pred.append(logits)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/200 samples...")

    neg_pred = np.array(neg_pred)

    # Compute softmax probabilities
    print("\nComputing softmax probabilities...")
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    predictions = np.exp(pos_pred) / sum_pred

    print(f"  - Positive predictions range: [{pos_pred.min():.4f}, {pos_pred.max():.4f}]")
    print(f"  - Negative predictions range: [{neg_pred.min():.4f}, {neg_pred.max():.4f}]")

    # predictions shape: (N, 1) for single label
    predictions = predictions.squeeze()  # (N,)

    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  - Predictions mean: {predictions.mean():.4f}")
    print(f"  - Predictions std: {predictions.std():.4f}")

    # Since all samples are Pneumonia (Target=1), we create ground truth labels
    ground_truth = np.ones(len(predictions))  # All positive

    # Compute metrics
    print("\n" + "="*70)
    print("Performance Metrics (All samples are Pneumonia-positive)")
    print("="*70)

    # AUROC (not meaningful here since all labels are 1, but included for completeness)
    try:
        auroc = roc_auc_score(ground_truth, predictions)
        print(f"  AUROC: {auroc:.4f} (Note: Not meaningful with single class)")
    except:
        auroc = None
        print(f"  AUROC: Cannot compute (all labels are identical)")

    # Accuracy at threshold 0.5
    pred_binary = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(ground_truth, pred_binary)
    print(f"  Accuracy @ 0.5: {accuracy:.4f} ({int(accuracy*len(predictions))}/{len(predictions)})")

    # True Positive Rate (sensitivity) at 0.5
    tpr = pred_binary.sum() / len(pred_binary)
    print(f"  True Positive Rate @ 0.5: {tpr:.4f}")

    # Prediction distribution
    print(f"\nPrediction Distribution:")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  25%: {np.percentile(predictions, 25):.4f}")
    print(f"  50%: {np.percentile(predictions, 50):.4f}")
    print(f"  75%: {np.percentile(predictions, 75):.4f}")
    print(f"  Max: {predictions.max():.4f}")

    # Save results
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    # Save predictions
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'prediction': predictions,
        'ground_truth': ground_truth,
        'num_lesions': [lesion_info[pid]['num_lesions'] for pid in patient_ids],
        'mask_percentage': [lesion_info[pid]['mask_percentage'] for pid in patient_ids]
    })

    results_csv = os.path.join(output_dir, 'chexzero_rsna_baseline.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"  - Predictions saved to: {results_csv}")

    # Plot prediction histogram
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold=0.5')
    plt.xlabel('Pneumonia Prediction Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('CheXzero Baseline Predictions on RSNA Pneumonia Dataset (N=200)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = os.path.join(output_dir, 'prediction_histogram.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  - Histogram saved to: {plot_path}")
    plt.close()

    # Save summary statistics
    summary = {
        'model_path': model_path,
        'num_samples': len(predictions),
        'auroc': float(auroc) if auroc is not None else None,
        'accuracy_at_0.5': float(accuracy),
        'tpr_at_0.5': float(tpr),
        'prediction_mean': float(predictions.mean()),
        'prediction_std': float(predictions.std()),
        'prediction_min': float(predictions.min()),
        'prediction_max': float(predictions.max())
    }

    summary_json = os.path.join(output_dir, 'baseline_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  - Summary saved to: {summary_json}")

    print("\n" + "="*70)
    print("Baseline Evaluation Complete!")
    print("="*70)

    return predictions, results_df


if __name__ == '__main__':
    predictions, results = evaluate_chexzero_rsna()
