"""
Phase 2 – Feature Engineering
Apply all features on train/val/test sets only
"""

import os
import pandas as pd

from src.feature_avg_word_length import apply_avg_word_length
from src.feature_interjection import apply_interjection_feature
from src.feature_indefinite import apply_indefinite_feature
from src.feature_sentiment import apply_sentiment_feature
from src.feature_exclamation import apply_exclamation_feature


def process_split(df, split_name, output_path):

    print(f"\n=== Processing {split_name} split ===")

    # Apply features one by one
    df = apply_avg_word_length(df)
    df = apply_exclamation_feature(df)
    df = apply_interjection_feature(df)
    df = apply_indefinite_feature(df)
    df = apply_sentiment_feature(df)

    # Save output
    save_path = f"{output_path}/{split_name}_features.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"{split_name} features saved → {save_path}")
    return df


def run_phase2():

    print("Loading Train/Val/Test files...")

    train_df = pd.read_csv("data/raw/train.csv")
    val_df = pd.read_csv("data/raw/val.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    output_path = "data/features"
    os.makedirs(output_path, exist_ok=True)

    # Run features on each split
    train_df = process_split(train_df, "train", output_path)
    val_df = process_split(val_df, "val", output_path)
    test_df = process_split(test_df, "test", output_path)

    print("\nPhase 2 completed successfully!")
    print(f"Files saved under: {output_path}/")


if __name__ == "__main__":
    run_phase2()
