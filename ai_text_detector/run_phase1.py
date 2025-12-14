"""
Phase 1 - Data Preparation:
Load dataset → Convert to long-format → Train/Val/Test Split (70/15/15)
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset


def run_phase1():

    # =============================
    # 1) Load dataset
    # =============================
    print("Loading dataset from HuggingFace ...")
    ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

    rows = []
    ai_columns = [
        "allam_generated_abstract",
        "jais_generated_abstract",
        "llama_generated_abstract",
        "openai_generated_abstract"
    ]

    # =============================
    # 2) Convert dataset to long format
    # =============================
    print("Converting to long-format ...")

    for split in ds.keys():
        df_split = pd.DataFrame(ds[split])
        df_split["source"] = split

        for _, row in df_split.iterrows():

            # Human-written text
            rows.append({
                "text": row["original_abstract"],
                "label": "Human",
                "source": split
            })

            # AI-generated versions
            for col in ai_columns:
                if pd.notna(row[col]):
                    rows.append({
                        "text": row[col],
                        "label": "AI",
                        "source": split
                    })

    df_long = pd.DataFrame(rows)
    print(f"Total long-format rows: {len(df_long)}")

    # =============================
    # 3) Train / Validation / Test split (70 / 15 / 15)
    # =============================
    print("Splitting dataset into Train / Validation / Test ...")

    # Shuffle dataset
    df_long = df_long.sample(frac=1, random_state=42).reset_index(drop=True)

    train_end = int(0.70 * len(df_long))
    val_end = int(0.85 * len(df_long))

    df_train = df_long.iloc[:train_end]
    df_val = df_long.iloc[train_end:val_end]
    df_test = df_long.iloc[val_end:]

    print(f"Train size: {len(df_train)}")
    print(f"Validation size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")

    # =============================
    # 4) Save output files
    # =============================
    output_path = "data/raw"
    os.makedirs(output_path, exist_ok=True)

    df_train.to_csv(f"{output_path}/train.csv", index=False, encoding="utf-8-sig")
    df_val.to_csv(f"{output_path}/val.csv", index=False, encoding="utf-8-sig")
    df_test.to_csv(f"{output_path}/test.csv", index=False, encoding="utf-8-sig")

    print("\nPhase 1 completed successfully.")
    print("Files saved under: data/raw/")


if __name__ == "__main__":
    run_phase1()
