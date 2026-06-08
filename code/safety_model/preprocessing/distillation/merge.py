import os
import argparse
import shutil
from datasets import load_from_disk, concatenate_datasets


def merge_datasets(input_dir: str):
    """
    Merge train and train_fallback datasets, and test and test_fallback datasets.
    
    Process:
    1. Load train and train_fallback datasets
    2. Rename train to train_passed
    3. Merge train_fallback with train_passed and save as train
    4. Repeat for test and test_fallback
    
    Final result: 6 datasets stored
    - train_passed: original train dataset
    - train_fallback: original train_fallback dataset
    - train: merged train + train_fallback
    """
    
    # Validate input directory exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Check for required datasets
    train_path = os.path.join(input_dir, "train")
    train_fallback_path = os.path.join(input_dir, "train_fallback")
    
    required_paths = [train_path, train_fallback_path]
    for path in required_paths:
        if not os.path.isdir(path):
            raise ValueError(f"Required dataset directory not found: {path}")
    
    print(f"Processing datasets in: {input_dir}")
    
    # ===== TRAIN DATASET PROCESSING =====
    print("\n[1/1] Processing TRAIN datasets...")
    
    # Load datasets
    print("  Loading train dataset...")
    train_data = load_from_disk(train_path)
    print(f"    train: {len(train_data)} samples")
    
    print("  Loading train_fallback dataset...")
    train_fallback_data = load_from_disk(train_fallback_path)
    print(f"    train_fallback: {len(train_fallback_data)} samples")
    
    # Rename original train to train_passed
    train_passed_path = os.path.join(input_dir, "train_passed")
    print(f"  Renaming train -> train_passed...")
    shutil.move(train_path, train_passed_path)
    print(f"    Saved to: {train_passed_path}")
    
    # Merge train_fallback with train_passed
    print("  Merging train_fallback + train_passed...")
    train_merged = concatenate_datasets([train_fallback_data, train_data])
    print(f"    Merged dataset: {len(train_merged)} samples")
    
    # Save merged as train (use temp location first to avoid conflicts)
    train_new_path = os.path.join(input_dir, "train")
    train_temp_path = os.path.join(input_dir, "train_temp")
    print(f"  Saving merged dataset as train...")
    train_merged.save_to_disk(train_temp_path)
    shutil.move(train_temp_path, train_new_path)
    print(f"    Saved to: {train_new_path}")
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Final datasets in {input_dir}:")
    print(f"  ✓ train_passed:    {len(train_data)} samples")
    print(f"  ✓ train_fallback:  {len(train_fallback_data)} samples")
    print(f"  ✓ train:           {len(train_merged)} samples (merged)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge train/test datasets with their fallback versions."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the dataset directory containing train, train_fallback, test, test_fallback subdirectories"
    )
    
    args = parser.parse_args()
    merge_datasets(args.input_dir)


if __name__ == "__main__":
    main()
