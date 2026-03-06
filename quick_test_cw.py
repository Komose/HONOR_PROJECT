"""
Quick test: Is C&W actually working?
"""

import torch
from rsna_attack_framework import RSNADataset, CheXzeroWrapper
from unified_attack_framework_fixed import cw_attack_unified, compute_metrics
from torch.utils.data import DataLoader

# Load 5 samples for quick test
dataset = RSNADataset(
    h5_path='dataset/rsna/rsna_200_samples.h5',
    lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CheXzeroWrapper(
    model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
    device=device
)

print("=" * 80)
print("Quick C&W Test (5 samples)")
print("=" * 80)

# Test recommended params vs aggressive params
param_sets = [
    {'name': 'Recommended', 'c': 10.0, 'kappa': 0.01, 'steps': 1000, 'lr': 0.01},
    {'name': 'Aggressive', 'c': 100.0, 'kappa': 0.0, 'steps': 1000, 'lr': 0.01},
]

for param_set in param_sets:
    print(f"\n{'=' * 80}")
    print(f"Testing: {param_set['name']}")
    print(f"Params: c={param_set['c']}, kappa={param_set['kappa']}, steps={param_set['steps']}, lr={param_set['lr']}")
    print(f"{'=' * 80}")

    successes = []
    l2_norms = []
    conf_drops = []

    for i, batch in enumerate(dataloader):
        if i >= 5:  # Only 5 samples
            break

        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Get clean prediction
        with torch.no_grad():
            clean_prob = model(images).cpu().item()

        print(f"\nSample {i+1}: Clean prob = {clean_prob:.4f}")

        # Run C&W attack (full mode for simplicity)
        params = {k: v for k, v in param_set.items() if k != 'name'}
        adv_images, perturbations = cw_attack_unified(
            model=model,
            images=images,
            masks=masks,
            attack_mode='full',
            **params
        )

        # Get adv prediction
        with torch.no_grad():
            adv_prob = model(adv_images).cpu().item()

        # Compute metrics
        metrics = compute_metrics(
            clean_images=images,
            adv_images=adv_images,
            clean_probs=torch.tensor([clean_prob]),
            adv_probs=torch.tensor([adv_prob]),
            perturbations=perturbations
        )

        success = metrics['success'][0]
        l2_norm = metrics['l2_norm'][0]
        conf_drop = metrics['confidence_drop'][0]

        print(f"  Adv prob = {adv_prob:.4f}")
        print(f"  Success = {success}, L2 = {l2_norm:.2f}, Conf drop = {conf_drop:.4f}")

        successes.append(success)
        l2_norms.append(l2_norm)
        conf_drops.append(conf_drop)

    print(f"\n{param_set['name']} Summary:")
    print(f"  ASR: {sum(successes)/len(successes):.1%}")
    print(f"  Avg L2: {sum(l2_norms)/len(l2_norms):.2f}")
    print(f"  Avg Conf Drop: {sum(conf_drops)/len(conf_drops):.4f}")

dataset.close()
print("\n" + "=" * 80)
