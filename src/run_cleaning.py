import os
import pandas as pd
from data_cleaning import preprocess_text

def clean_split(df, split_name, output_path):
    print(f"\n=== Cleaning {split_name} split ===\n")

    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_text)

    save_path = f"{output_path}/{split_name}_clean.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"{split_name} cleaned file saved → {save_path}")

def run_cleaning():
    print("Loading raw splits...")

    train_df = pd.read_csv("data/raw/train.csv")
    val_df   = pd.read_csv("data/raw/val.csv")
    test_df  = pd.read_csv("data/raw/test.csv")

    output_path = "data/processed"
    os.makedirs(output_path, exist_ok=True)

    clean_split(train_df, "train", output_path)
    clean_split(val_df, "val", output_path)
    clean_split(test_df, "test", output_path)

    print("\nCleaning Phase Completed Successfully!")

if __name__ == "__main__":
    run_cleaning()

