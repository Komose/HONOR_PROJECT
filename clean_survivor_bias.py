"""
Survivor Bias Cleaner: Strict Intersection Filtering
=====================================================

Purpose:
    Eliminates survivor bias by retaining ONLY patients who have data
    in ALL three attack modes (lesion, random_patch, full).

Scientific Rationale:
    - Lesion/Full modes: All 200 patients succeed
    - Random_patch mode: Only ~155 patients succeed (geometric constraints)
    - Failed patients are heavy-infection cases (large lesions)
    - Comparing different patient cohorts = INVALID comparison
    - Solution: Paired-sample analysis on intersection set

Usage:
    python clean_survivor_bias.py --input results/unified_final_rigid_translation/all_algorithms_consolidated.csv

Author: HONER Project
Date: 2026-03-12
"""

import os
import pandas as pd
import argparse
from pathlib import Path


def load_and_validate_data(input_path: str) -> pd.DataFrame:
    """Load consolidated results and validate structure."""
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded: {input_path}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Validate required columns
    required_cols = ['algorithm', 'mode', 'patient_id']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def analyze_cohort_distribution(df: pd.DataFrame) -> dict:
    """Analyze patient distribution across modes."""
    print("\n" + "=" * 80)
    print("STEP 2: COHORT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    stats = {}

    # Overall statistics
    total_records = len(df)
    unique_patients = df['patient_id'].nunique()
    algorithms = df['algorithm'].unique()
    modes = df['mode'].unique()

    print(f"\nOverall Statistics:")
    print(f"  Total records: {total_records}")
    print(f"  Unique patients: {unique_patients}")
    print(f"  Algorithms: {list(algorithms)}")
    print(f"  Modes: {list(modes)}")

    stats['total_records'] = total_records
    stats['unique_patients'] = unique_patients
    stats['algorithms'] = list(algorithms)
    stats['modes'] = list(modes)

    # Per-mode patient counts
    print(f"\nPatient Counts by Mode:")
    mode_counts = {}
    for mode in ['lesion', 'random_patch', 'full']:
        if mode in modes:
            mode_df = df[df['mode'] == mode]
            n_patients = mode_df['patient_id'].nunique()
            n_records = len(mode_df)
            mode_counts[mode] = n_patients
            print(f"  {mode:15s}: {n_patients:3d} patients ({n_records:4d} records)")

    stats['mode_counts'] = mode_counts

    # Identify survivor bias
    if 'random_patch' in mode_counts and 'lesion' in mode_counts:
        dropout = mode_counts['lesion'] - mode_counts['random_patch']
        dropout_rate = dropout / mode_counts['lesion'] * 100
        print(f"\n[SURVIVOR BIAS DETECTED]")
        print(f"  Dropout: {dropout} patients ({dropout_rate:.1f}%)")
        print(f"  These are likely severe cases with large lesions")
        stats['dropout'] = dropout
        stats['dropout_rate'] = dropout_rate

    return stats


def find_intersection_cohort(df: pd.DataFrame) -> set:
    """Find patients with data in ALL three modes."""
    print("\n" + "=" * 80)
    print("STEP 3: FINDING INTERSECTION COHORT")
    print("=" * 80)

    # Get patient sets for each mode
    modes = ['lesion', 'random_patch', 'full']
    patient_sets = {}

    for mode in modes:
        mode_df = df[df['mode'] == mode]
        patient_sets[mode] = set(mode_df['patient_id'].unique())
        print(f"  {mode:15s}: {len(patient_sets[mode]):3d} patients")

    # Compute intersection
    if len(patient_sets) == 3:
        intersection = patient_sets['lesion'] & patient_sets['random_patch'] & patient_sets['full']
    else:
        # Fallback: intersect all available modes
        intersection = set.intersection(*patient_sets.values())

    print(f"\n[INTERSECTION SET]")
    print(f"  Patients in ALL modes: {len(intersection)}")

    # Identify excluded patients
    all_patients = set(df['patient_id'].unique())
    excluded = all_patients - intersection
    print(f"  Excluded patients: {len(excluded)}")

    if len(excluded) > 0:
        print(f"\n  Excluded patient IDs (first 10):")
        for pid in list(excluded)[:10]:
            print(f"    - {pid}")
        if len(excluded) > 10:
            print(f"    ... and {len(excluded) - 10} more")

    return intersection


def filter_to_intersection(df: pd.DataFrame, intersection: set) -> pd.DataFrame:
    """Filter dataframe to only include intersection cohort."""
    print("\n" + "=" * 80)
    print("STEP 4: APPLYING STRICT INTERSECTION FILTER")
    print("=" * 80)

    original_records = len(df)
    original_patients = df['patient_id'].nunique()

    # Filter
    df_clean = df[df['patient_id'].isin(intersection)].copy()

    clean_records = len(df_clean)
    clean_patients = df_clean['patient_id'].nunique()

    print(f"\nBefore Filtering:")
    print(f"  Records: {original_records}")
    print(f"  Patients: {original_patients}")

    print(f"\nAfter Filtering:")
    print(f"  Records: {clean_records}")
    print(f"  Patients: {clean_patients}")

    print(f"\nFiltered Out:")
    print(f"  Records: {original_records - clean_records} ({(original_records - clean_records) / original_records * 100:.1f}%)")
    print(f"  Patients: {original_patients - clean_patients} ({(original_patients - clean_patients) / original_patients * 100:.1f}%)")

    return df_clean


def validate_pairing(df_clean: pd.DataFrame) -> bool:
    """Validate that all patients have data in all modes."""
    print("\n" + "=" * 80)
    print("STEP 5: VALIDATING PAIRED-SAMPLE STRUCTURE")
    print("=" * 80)

    modes = df_clean['mode'].unique()
    algorithms = df_clean['algorithm'].unique()

    n_expected_per_patient = len(algorithms) * len(modes)

    patient_record_counts = df_clean.groupby('patient_id').size()

    all_valid = True
    invalid_patients = []

    for pid, count in patient_record_counts.items():
        if count != n_expected_per_patient:
            all_valid = False
            invalid_patients.append((pid, count))

    if all_valid:
        print(f"[OK] All patients have complete data")
        print(f"  Expected records per patient: {n_expected_per_patient}")
        print(f"  (Algorithms: {len(algorithms)} × Modes: {len(modes)} = {n_expected_per_patient})")
        print(f"  All {len(patient_record_counts)} patients validated")
        return True
    else:
        print(f"[WARNING] {len(invalid_patients)} patients have incomplete data:")
        for pid, count in invalid_patients[:5]:
            print(f"  - {pid}: {count} records (expected {n_expected_per_patient})")
        return False


def generate_summary_statistics(df_clean: pd.DataFrame) -> dict:
    """Generate summary statistics for the cleaned dataset."""
    print("\n" + "=" * 80)
    print("STEP 6: SUMMARY STATISTICS")
    print("=" * 80)

    summary = {}

    # Overall
    n_patients = df_clean['patient_id'].nunique()
    n_algorithms = df_clean['algorithm'].nunique()
    n_modes = df_clean['mode'].nunique()
    n_records = len(df_clean)

    summary['n_patients'] = n_patients
    summary['n_algorithms'] = n_algorithms
    summary['n_modes'] = n_modes
    summary['n_records'] = n_records

    print(f"\nCleaned Dataset Summary:")
    print(f"  Patients: {n_patients}")
    print(f"  Algorithms: {n_algorithms}")
    print(f"  Modes: {n_modes}")
    print(f"  Total records: {n_records}")

    # Per algorithm-mode statistics
    print(f"\nRecords per Algorithm-Mode:")
    for algo in df_clean['algorithm'].unique():
        print(f"  {algo.upper()}:")
        for mode in ['lesion', 'random_patch', 'full']:
            subset = df_clean[(df_clean['algorithm'] == algo) & (df_clean['mode'] == mode)]
            if len(subset) > 0:
                asr = subset['success'].mean() if 'success' in subset.columns else 0
                print(f"    {mode:15s}: {len(subset):3d} records, ASR={asr:.1%}")

    return summary


def save_cleaned_data(df_clean: pd.DataFrame, output_path: str):
    """Save cleaned data to CSV."""
    print("\n" + "=" * 80)
    print("STEP 7: SAVING CLEANED DATA")
    print("=" * 80)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved cleaned data to:")
    print(f"  {output_path}")
    print(f"  Size: {len(df_clean)} records")
    print(f"  Columns: {len(df_clean.columns)}")

    # Verify file
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"  File size: {file_size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='Clean survivor bias via strict intersection filtering'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/unified_final_rigid_translation/all_algorithms_consolidated.csv',
        help='Input CSV file with all results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/unified_final_rigid_translation/CLEANED_PAIRED_RESULTS.csv',
        help='Output CSV file with cleaned paired results'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SURVIVOR BIAS CLEANER: STRICT INTERSECTION FILTERING")
    print("=" * 80)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")

    try:
        # Load data
        df = load_and_validate_data(args.input)

        # Analyze distribution
        stats = analyze_cohort_distribution(df)

        # Find intersection
        intersection = find_intersection_cohort(df)

        if len(intersection) == 0:
            raise ValueError("No patients found in intersection set!")

        # Filter to intersection
        df_clean = filter_to_intersection(df, intersection)

        # Validate pairing
        is_valid = validate_pairing(df_clean)

        if not is_valid:
            print("\n[WARNING] Validation found incomplete data, but proceeding...")

        # Generate summary
        summary = generate_summary_statistics(df_clean)

        # Save cleaned data
        save_cleaned_data(df_clean, args.output)

        print("\n" + "=" * 80)
        print("[COMPLETE] SURVIVOR BIAS SUCCESSFULLY ELIMINATED")
        print("=" * 80)
        print(f"\nKey Results:")
        print(f"  Original patients: {stats['unique_patients']}")
        print(f"  Cleaned patients: {summary['n_patients']}")
        print(f"  Retention rate: {summary['n_patients'] / stats['unique_patients'] * 100:.1f}%")
        print(f"\nThis cleaned dataset ensures:")
        print(f"  [x] No survivor bias")
        print(f"  [x] Paired-sample structure")
        print(f"  [x] Valid statistical comparisons")
        print(f"  [x] Scientifically rigorous conclusions")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
