"""
RSNA Pneumonia Dataset Preparation for CheXzero Adversarial Attack Experiments

This script:
1. Loads DICOM images from RSNA dataset
2. Extracts lesion bounding boxes
3. Preprocesses images for CheXzero (224x224, normalized)
4. Saves data in HDF5 format compatible with CheXzero
5. Creates lesion mask data structure for targeted attacks
"""

import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import h5py
from tqdm import tqdm
import json


def load_dicom_image(dicom_path):
    """
    Load DICOM image and convert to numpy array.

    Args:
        dicom_path: Path to DICOM file

    Returns:
        img_array: Normalized image array (H, W), values in [0, 1]
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        img_array = dicom.pixel_array.astype(np.float32)

        # Normalize to [0, 1]
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

        return img_array
    except Exception as e:
        print(f"Error loading {dicom_path}: {e}")
        return None


def preprocess_image(img_array, target_size=(224, 224)):
    """
    Preprocess image for CheXzero model with CLIP normalization.

    Args:
        img_array: Input image array (H, W)
        target_size: Target size (default 224x224 for CheXzero)

    Returns:
        processed: Processed image array (C, H, W) with CLIP normalization
    """
    # Convert to PIL Image
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))

    # Resize
    img_pil = img_pil.resize(target_size, Image.BILINEAR)

    # Convert back to numpy and normalize to [0, 1]
    img_array = np.array(img_pil).astype(np.float32) / 255.0

    # Convert grayscale to 3-channel (CheXzero expects RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=0)  # (3, H, W)
    else:
        # If already 3D (H, W, C), transpose to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

    # Apply CLIP normalization (same as CheXzero training)
    # Mean and std for CLIP: https://github.com/openai/CLIP
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)

    img_array = (img_array - mean) / std

    return img_array


def scale_bbox(bbox, original_size, target_size=(224, 224)):
    """
    Scale bounding box coordinates from original image size to target size.

    Args:
        bbox: (x, y, width, height) in original coordinates
        original_size: (H, W) of original image
        target_size: (H, W) of target image

    Returns:
        scaled_bbox: (x, y, width, height) in target coordinates
    """
    x, y, w, h = bbox
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled_x = int(x * scale_x)
    scaled_y = int(y * scale_y)
    scaled_w = int(w * scale_x)
    scaled_h = int(h * scale_y)

    # Ensure within bounds
    scaled_x = max(0, min(scaled_x, target_w - 1))
    scaled_y = max(0, min(scaled_y, target_h - 1))
    scaled_w = max(1, min(scaled_w, target_w - scaled_x))
    scaled_h = max(1, min(scaled_h, target_h - scaled_y))

    return (scaled_x, scaled_y, scaled_w, scaled_h)


def create_lesion_mask(lesion_bboxes, image_shape=(224, 224)):
    """
    Create binary mask from lesion bounding boxes.

    Args:
        lesion_bboxes: List of (x, y, width, height) tuples
        image_shape: (H, W) of target image

    Returns:
        mask: Binary mask (H, W), 1 for lesion regions, 0 otherwise
    """
    mask = np.zeros(image_shape, dtype=np.float32)

    for bbox in lesion_bboxes:
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1.0

    return mask


def prepare_rsna_dataset(
    image_dir='dataset/rsna/stage_2_train_images',
    labels_csv='dataset/rsna/stage_2_train_labels.csv',
    selected_patients_csv='dataset/rsna/selected_200_patients.csv',
    output_h5='dataset/rsna/rsna_200_samples.h5',
    output_lesion_info='dataset/rsna/rsna_200_lesion_info.json',
    target_size=(224, 224)
):
    """
    Prepare RSNA dataset for adversarial attack experiments.

    Args:
        image_dir: Directory containing DICOM images
        labels_csv: Path to labels CSV file
        selected_patients_csv: Path to selected patients CSV
        output_h5: Output HDF5 file path
        output_lesion_info: Output JSON file for lesion metadata
        target_size: Target image size (H, W)
    """
    print("="*60)
    print("RSNA Dataset Preparation for CheXzero Attack Experiments")
    print("="*60)

    # Load data
    print("\n[1/5] Loading labels and patient list...")
    labels_df = pd.read_csv(labels_csv)
    selected_patients = pd.read_csv(selected_patients_csv)
    patient_ids = selected_patients['patientId'].values

    print(f"  - Total patients: {len(patient_ids)}")

    # Filter labels for selected patients
    labels_df = labels_df[labels_df['patientId'].isin(patient_ids)]
    lesion_labels = labels_df[labels_df['Target'] == 1]

    print(f"  - Total lesion annotations: {len(lesion_labels)}")

    # Prepare storage
    print("\n[2/5] Preparing storage...")
    images = []
    lesion_info = {}
    failed = []

    # Process each patient
    print("\n[3/5] Processing DICOM images...")
    for patient_id in tqdm(patient_ids, desc="Processing"):
        dicom_path = os.path.join(image_dir, f"{patient_id}.dcm")

        # Load DICOM
        img_array = load_dicom_image(dicom_path)
        if img_array is None:
            failed.append(patient_id)
            continue

        original_size = img_array.shape  # (H, W)

        # Preprocess image
        processed_img = preprocess_image(img_array, target_size)
        images.append(processed_img)

        # Extract lesion bounding boxes for this patient
        patient_lesions = lesion_labels[lesion_labels['patientId'] == patient_id]
        lesion_bboxes = []

        for _, row in patient_lesions.iterrows():
            x, y, w, h = row['x'], row['y'], row['width'], row['height']

            # Scale bbox to target size
            scaled_bbox = scale_bbox((x, y, w, h), original_size, target_size)
            lesion_bboxes.append(scaled_bbox)

        # Create lesion mask
        lesion_mask = create_lesion_mask(lesion_bboxes, target_size)

        # Store lesion info
        lesion_info[patient_id] = {
            'num_lesions': len(lesion_bboxes),
            'bboxes': lesion_bboxes,  # List of (x, y, w, h) in target coordinates
            'mask_nonzero_pixels': int(lesion_mask.sum()),
            'mask_percentage': float(lesion_mask.sum() / (target_size[0] * target_size[1]) * 100)
        }

    print(f"\n  - Successfully processed: {len(images)}/{len(patient_ids)}")
    if failed:
        print(f"  - Failed: {len(failed)}")
        print(f"    {failed[:5]}...")

    # Convert to numpy array
    print("\n[4/5] Creating HDF5 file...")
    images_array = np.stack(images, axis=0)  # (N, 3, 224, 224)
    print(f"  - Images shape: {images_array.shape}")
    print(f"  - Images dtype: {images_array.dtype}")
    print(f"  - Images range: [{images_array.min():.3f}, {images_array.max():.3f}]")

    # Save to HDF5
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('cxr', data=images_array, compression='gzip')
        f.attrs['num_samples'] = len(images)
        f.attrs['image_size'] = target_size
        f.attrs['description'] = 'RSNA Pneumonia Dataset - 200 patients with lesions'

    print(f"  - Saved to: {output_h5}")

    # Save lesion info
    print("\n[5/5] Saving lesion metadata...")

    # Add patient_id list to maintain order
    lesion_info_export = {
        'patient_ids': patient_ids.tolist(),
        'lesion_data': lesion_info,
        'statistics': {
            'total_patients': len(patient_ids),
            'total_lesions': sum([info['num_lesions'] for info in lesion_info.values()]),
            'avg_lesions_per_patient': np.mean([info['num_lesions'] for info in lesion_info.values()]),
            'avg_mask_percentage': np.mean([info['mask_percentage'] for info in lesion_info.values()])
        }
    }

    with open(output_lesion_info, 'w') as f:
        json.dump(lesion_info_export, f, indent=2)

    print(f"  - Saved to: {output_lesion_info}")

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    stats = lesion_info_export['statistics']
    print(f"  Total patients: {stats['total_patients']}")
    print(f"  Total lesions: {stats['total_lesions']}")
    print(f"  Avg lesions per patient: {stats['avg_lesions_per_patient']:.2f}")
    print(f"  Avg lesion mask coverage: {stats['avg_mask_percentage']:.2f}%")
    print("="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)


if __name__ == '__main__':
    prepare_rsna_dataset()
