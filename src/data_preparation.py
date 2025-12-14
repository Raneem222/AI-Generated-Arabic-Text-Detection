"""
Phase 1: Data Loading & Quality Inspection
Project: AI Text Detection
Description:
    - Load dataset from HuggingFace
    - Merge all splits into a single DataFrame
    - Perform basic quality checks
    - Save raw merged dataset into data/raw/
"""

import os
import pandas as pd
from datasets import load_dataset


def load_raw_dataset():
    """Load dataset from HuggingFace repository."""
    print("Loading dataset...")
    dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

    print("Available splits:", list(dataset.keys()))
    return dataset


def merge_splits(dataset):
    """Merge all dataset splits into a single DataFrame and save to data/raw."""
    print("Merging splits into one DataFrame...")

    frames = []

    for split in dataset:
        df_split = pd.DataFrame(dataset[split])
        df_split["split"] = split
        frames.append(df_split)

    df_all = pd.concat(frames, ignore_index=True)
    print("Merged DataFrame shape:", df_all.shape)

    # Save raw dataset
    raw_path = "data/raw"
    os.makedirs(raw_path, exist_ok=True)

    save_path = os.path.join(raw_path, "raw_dataset.csv")
    df_all.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("Raw dataset saved at:", save_path)
    return df_all


def count_texts(df):
    """Calculate number of human and AI-generated texts."""
    print("Counting text samples...")

    human_count = df["original_abstract"].astype(str).str.strip().ne("").sum()

    ai_columns = [
        "allam_generated_abstract",
        "jais_generated_abstract",
        "llama_generated_abstract",
        "openai_generated_abstract",
    ]

    ai_count = sum(df[col].astype(str).str.strip().ne("").sum() for col in ai_columns)

    print("Human text count:", human_count)
    print("AI-generated text count:", ai_count)

    return human_count, ai_count


def inspect_quality(df):
    """Perform quality checks on dataset."""
    print("\nMissing values per column:")
    print(df.isna().sum())

    duplicates = df.duplicated().sum()
    print("Number of duplicate rows:", duplicates)

    # Check extreme text lengths
    text_lengths = df["original_abstract"].dropna().astype(str).str.len()

    print("Very short texts (< 10 characters):", (text_lengths < 10).sum())
    print("Very long texts (> 3× mean length):", (text_lengths > text_lengths.mean() * 3).sum())

    # Check whitespace issues
    clean_text = df["original_abstract"].dropna().astype(str)
    whitespace_issues = clean_text[
        clean_text.str.startswith(" ") | clean_text.str.endswith(" ")
    ]
    print("Texts with leading/trailing whitespace:", whitespace_issues.shape[0])


def run_phase1():
    """Main execution function for Phase 1."""
    dataset = load_raw_dataset()
    df_all = merge_splits(dataset)
    count_texts(df_all)
    inspect_quality(df_all)
    print("\nPhase 1 completed successfully.")


if __name__ == "__main__":
    run_phase1()
