"""
Test different text templates and prompts to improve prediction confidence.
"""

import torch
import sys
import numpy as np
sys.path.insert(0, 'CheXzero')
from rsna_attack_framework import CheXzeroWrapper, RSNADataset
import clip


def test_template_variants(model, image, device):
    """Test different template formulations."""

    templates_to_test = [
        # Original
        ("Pneumonia", "No Pneumonia"),

        # More specific medical terms
        ("Pneumonia", "Normal"),
        ("Bacterial Pneumonia", "No Bacterial Pneumonia"),
        ("Pneumonia infection", "No infection"),

        # More descriptive
        ("Findings consistent with Pneumonia", "No findings of Pneumonia"),
        ("Opacity suggestive of Pneumonia", "Clear lungs"),

        # Alternative related terms
        ("Lung Opacity", "No Lung Opacity"),
        ("Consolidation", "No Consolidation"),
        ("Infiltrate", "No Infiltrate"),

        # Combined concepts
        ("Pneumonia or Lung Opacity", "Normal chest radiograph"),
    ]

    results = []

    with torch.no_grad():
        for pos_text, neg_text in templates_to_test:
            # Tokenize
            pos_tokens = clip.tokenize([pos_text], context_length=77).to(device)
            neg_tokens = clip.tokenize([neg_text], context_length=77).to(device)

            # Get text features
            pos_features = model.model.encode_text(pos_tokens)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)

            neg_features = model.model.encode_text(neg_tokens)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

            # Get image features
            image_features = model.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            pos_logit = (image_features @ pos_features.T).item()
            neg_logit = (image_features @ neg_features.T).item()

            # Softmax
            prob = np.exp(pos_logit) / (np.exp(pos_logit) + np.exp(neg_logit))

            results.append({
                'positive': pos_text,
                'negative': neg_text,
                'pos_logit': pos_logit,
                'neg_logit': neg_logit,
                'probability': prob,
                'difference': pos_logit - neg_logit
            })

    return results


def test_ensemble_labels(model, image, device):
    """Test using ensemble of related labels."""

    # Labels related to pneumonia
    related_labels = [
        'Pneumonia',
        'Lung Opacity',
        'Consolidation',
        'Infiltrate'
    ]

    all_probs = []

    with torch.no_grad():
        for label in related_labels:
            # Tokenize
            pos_tokens = clip.tokenize([label], context_length=77).to(device)
            neg_tokens = clip.tokenize([f"No {label}"], context_length=77).to(device)

            # Get text features
            pos_features = model.model.encode_text(pos_tokens)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)

            neg_features = model.model.encode_text(neg_tokens)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

            # Get image features
            image_features = model.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            pos_logit = (image_features @ pos_features.T).item()
            neg_logit = (image_features @ neg_features.T).item()

            # Softmax
            prob = np.exp(pos_logit) / (np.exp(pos_logit) + np.exp(neg_logit))
            all_probs.append(prob)

            print(f"  {label:20s}: {prob:.4f}")

    ensemble_prob = np.mean(all_probs)
    print(f"  {'Ensemble Average':20s}: {ensemble_prob:.4f}")

    return ensemble_prob


if __name__ == '__main__':
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=device
    )

    # Load dataset
    dataset = RSNADataset(
        h5_path='dataset/rsna/rsna_200_samples.h5',
        lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
    )

    # Test on first few samples
    print("="*80)
    print("Testing Different Templates and Approaches")
    print("="*80)

    for idx in range(5):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        patient_id = sample['patient_id']

        print(f"\n{'='*80}")
        print(f"Sample {idx+1}: {patient_id}")
        print(f"{'='*80}")

        # Test template variants
        print("\nTemplate Variants:")
        print("-" * 80)
        results = test_template_variants(model, image, device)

        # Sort by probability
        results_sorted = sorted(results, key=lambda x: x['probability'], reverse=True)

        for r in results_sorted:
            print(f"  [{r['probability']:.4f}] {r['positive']:40s} vs {r['negative']:30s} (diff: {r['difference']:+.4f})")

        # Test ensemble
        print("\nEnsemble Approach:")
        print("-" * 80)
        ensemble_prob = test_ensemble_labels(model, image, device)

    dataset.close()
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
