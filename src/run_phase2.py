import os

import pandas as pd
from data_preparation import load_raw_dataset
from feature_avg_word_length import apply_avg_word_length
from feature_interjection import apply_interjection_feature
from feature_sentiment import apply_sentiment_feature


from datasets import load_dataset

def run_phase2():

    print("Loading dataset ...")
    ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

    rows = []
    for split in ds.keys():
        df_split = pd.DataFrame(ds[split])
        df_split["split"] = split
        rows.append(df_split)

    df_all = pd.concat(rows, ignore_index=True)
    print(f"Total rows loaded: {len(df_all)}")

    # Convert to long format
    print("Converting dataset to long-format ...")
    long_rows = []
    ai_columns = [
        "allam_generated_abstract",
        "jais_generated_abstract",
        "llama_generated_abstract",
        "openai_generated_abstract"
    ]

    for _, row in df_all.iterrows():
        long_rows.append({
            "text": row["original_abstract"],
            "label": "Human",
            "source": row["split"]
        })

        for col in ai_columns:
            if pd.notna(row[col]):
                long_rows.append({
                    "text": row[col],
                    "label": "AI",
                    "source": row["split"]
                })

    df_long = pd.DataFrame(long_rows)
    print(f"Long-format total rows: {len(df_long)}")

    # -----------------------------
    # FIX: define output_path early
    # -----------------------------
    output_path = "data/features"
    os.makedirs(output_path, exist_ok=True)

    # Apply features
    print("Applying Feature 1: average_word_length ...")
    df_long = apply_avg_word_length(df_long)

    print("Applying Feature 2: interjection_count ...")
    df_long = apply_interjection_feature(df_long)

    print("Applying Feature 3: sentiment score ...")
    df_long = apply_sentiment_feature(df_long)

    # Save intermediate sentiment file
    sentiment_file = f"{output_path}/sentiment_score.csv"
    df_long.to_csv(sentiment_file, index=False, encoding="utf-8-sig")
    print(f"Sentiment score saved at: {sentiment_file}")

    # Save final features output
    csv_file = f"{output_path}/features_phase2.csv"
    df_long.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"Final features saved at: {csv_file}")

    print("Phase 2 completed successfully.")


if __name__ == "__main__":
    run_phase2()
