"""
Sanity Check: Verify mask fix works correctly
Test 5 patients with FGSM epsilon=8/255
"""
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from rsna_attack_framework import RSNADataset, CheXzeroWrapper, fgsm_attack
from multi_metric_attack_framework import extract_lung_region_mask, generate_equivalent_random_mask
import numpy as np

print("="*90)
print("SANITY CHECK: Mask Fix Verification")
print("="*90)
print("\nObjective: Verify that Lesion L0 = Random L0 after bug fix")
print("Test config: 20 patients (expecting ~5-10 successful), FGSM, epsilon=8/255")
print("-"*90)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epsilon = 8/255

# Load dataset (first 5 patients)
dataset = RSNADataset(
    h5_path='dataset/rsna/rsna_200_samples.h5',
    lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
)

# Select first 20 patients (to get ~5 successful after survivor bias filtering)
test_indices = list(range(20))
test_dataset = Subset(dataset, test_indices)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = CheXzeroWrapper(
    model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
    device=device
)

print(f"\nDevice: {device}")
print(f"Epsilon: {epsilon*255:.0f}/255 = {epsilon:.6f}")
print(f"Test samples: {len(test_dataset)}")

results = []

for batch_idx, batch in enumerate(dataloader):
    print(f"\n{'='*90}")
    print(f"Patient {batch_idx+1}/20: {batch['patient_id'][0][:24]}...")
    print(f"{'='*90}")

    image = batch['image'].to(device)
    lesion_mask = batch['mask'].to(device)
    patient_id = batch['patient_id'][0]

    # Calculate lesion L0
    lesion_l0 = (lesion_mask[0, 0] > 0).sum().item()
    print(f"Lesion mask L0: {lesion_l0} pixels")

    # Generate random mask
    try:
        lung_mask = extract_lung_region_mask(image[0])
        lung_mask = lung_mask.to(device)
        random_mask, random_info = generate_equivalent_random_mask(
            lesion_mask=lesion_mask[0],
            lung_mask=lung_mask,
            image=image[0],
            max_attempts=500
        )
        random_mask = random_mask.unsqueeze(0).to(device)
        random_l0 = (random_mask[0, 0] > 0).sum().item()

        print(f"Random mask L0: {random_l0} pixels")
        print(f"L0 difference:  {abs(random_l0 - lesion_l0)} pixels ({abs(random_l0-lesion_l0)/lesion_l0*100:.2f}%)")

        if abs(random_l0 - lesion_l0) < 10:
            print("PASS: L0 alignment verified (difference < 10 pixels)")
        else:
            print("FAIL: L0 mismatch!")

    except ValueError as e:
        print(f"X Random mask generation failed: {e}")
        print("  (This is expected for large lesions - survivor bias)")
        continue

    # Run FGSM attacks
    print(f"\n--- Running FGSM attacks ---")

    # Lesion attack
    adv_lesion, pert_lesion = fgsm_attack(
        model, image, lesion_mask, epsilon,
        targeted=False, attack_mode='lesion'
    )

    # Random attack (FIXED)
    adv_random, pert_random = fgsm_attack(
        model, image, random_mask, epsilon,
        targeted=False, attack_mode='random_patch'
    )

    # Full attack
    adv_full, pert_full = fgsm_attack(
        model, image, torch.ones_like(lesion_mask), epsilon,
        targeted=False, attack_mode='full'
    )

    # Evaluate
    with torch.no_grad():
        clean_prob = model(image).item()
        lesion_prob = model(adv_lesion).item()
        random_prob = model(adv_random).item()
        full_prob = model(adv_full).item()

    # Calculate L0 of perturbations
    lesion_pert_l0 = (torch.abs(pert_lesion[0]) > 1e-6).sum().item() / 3  # Divide by 3 channels
    random_pert_l0 = (torch.abs(pert_random[0]) > 1e-6).sum().item() / 3
    full_pert_l0 = (torch.abs(pert_full[0]) > 1e-6).sum().item() / 3

    # Calculate L2 norms
    lesion_l2 = torch.norm(pert_lesion[0]).item()
    random_l2 = torch.norm(pert_random[0]).item()
    full_l2 = torch.norm(pert_full[0]).item()

    print(f"\nClean prob:       {clean_prob:.4f}")
    print(f"Lesion adv prob:  {lesion_prob:.4f} (ASR: {1 if (clean_prob>=0.5 and lesion_prob<0.5) else 0})")
    print(f"Random adv prob:  {random_prob:.4f} (ASR: {1 if (clean_prob>=0.5 and random_prob<0.5) else 0})")
    print(f"Full adv prob:    {full_prob:.4f} (ASR: {1 if (clean_prob>=0.5 and full_prob<0.5) else 0})")

    print(f"\nPerturbation L0:")
    print(f"  Lesion:  {lesion_pert_l0:.0f} pixels")
    print(f"  Random:  {random_pert_l0:.0f} pixels")
    print(f"  Full:    {full_pert_l0:.0f} pixels")

    print(f"\nPerturbation L2:")
    print(f"  Lesion:  {lesion_l2:.3f}")
    print(f"  Random:  {random_l2:.3f}")
    print(f"  Full:    {full_l2:.3f}")

    # Verification
    if abs(lesion_pert_l0 - random_pert_l0) < 100:
        print("\nPASS PASS: Lesion L0 ≈ Random L0 (perturbation applied correctly)")
    else:
        print(f"\nFAIL FAIL: L0 mismatch! Lesion={lesion_pert_l0:.0f}, Random={random_pert_l0:.0f}")

    # Store results
    results.append({
        'patient_id': patient_id,
        'clean_prob': clean_prob,
        'lesion_prob': lesion_prob,
        'random_prob': random_prob,
        'full_prob': full_prob,
        'lesion_l0': lesion_pert_l0,
        'random_l0': random_pert_l0,
        'full_l0': full_pert_l0,
        'lesion_l2': lesion_l2,
        'random_l2': random_l2,
        'full_l2': full_l2
    })

# Summary
print("\n" + "="*90)
print("SUMMARY STATISTICS")
print("="*90)

df = pd.DataFrame(results)

print(f"\nSample size: {len(df)}")
print(f"\nAverage L0 (perturbation pixels):")
print(f"  Lesion:  {df['lesion_l0'].mean():.0f}")
print(f"  Random:  {df['random_l0'].mean():.0f}")
print(f"  Full:    {df['full_l0'].mean():.0f}")

print(f"\nAverage L2 (perturbation norm):")
print(f"  Lesion:  {df['lesion_l2'].mean():.3f}")
print(f"  Random:  {df['random_l2'].mean():.3f}")
print(f"  Full:    {df['full_l2'].mean():.3f}")

print(f"\nAttack Success Rate:")
print(f"  Lesion:  {((df['clean_prob']>=0.5) & (df['lesion_prob']<0.5)).sum()}/{len(df)}")
print(f"  Random:  {((df['clean_prob']>=0.5) & (df['random_prob']<0.5)).sum()}/{len(df)}")
print(f"  Full:    {((df['clean_prob']>=0.5) & (df['full_prob']<0.5)).sum()}/{len(df)}")

# Final verdict
print("\n" + "="*90)
print("FINAL VERDICT")
print("="*90)

l0_match = abs(df['lesion_l0'].mean() - df['random_l0'].mean()) / df['lesion_l0'].mean() < 0.05

if l0_match:
    print("\nPASS PASS: Bug fix successful!")
    print("  - Lesion and Random have equal L0 (within 5%)")
    print("  - Mask is being applied correctly")
    print("\n  Next step: Run full epsilon sweep experiment")
else:
    print("\nFAIL FAIL: Bug still present!")
    print("  - L0 mismatch detected")
    print("  - Need further investigation")

# Save results
output_dir = 'results/sanity_check'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f'{output_dir}/sanity_check_results.csv', index=False)
print(f"\nResults saved to: {output_dir}/sanity_check_results.csv")

dataset.close()
print("\n" + "="*90)
