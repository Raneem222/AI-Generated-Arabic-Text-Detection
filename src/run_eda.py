"""
Phase 2 - Task 2.2
EDA: Statistical + Lexical + Visualization
Generate plots and CSV reports for Human vs AI text
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
import re

from wordcloud import WordCloud

DATA_PATH = "data/processed"
OUTPUT_DIR = "reports/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================
# Utility functions
# =============================

def sentence_length(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())


def avg_word_length(text):
    if not isinstance(text, str):
        return 0
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(w) for w in words) / len(words)


def type_token_ratio(text):
    if not isinstance(text, str):
        return 0
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)


def punctuation_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"[.,!?:؛،]", text))



# =============================
# N-Gram function (top 20)
# =============================
def extract_ngrams(text_list, n=2, top_k=20):
    # تنظيف القائمة من العناصر غير النصية
    cleaned_list = []
    for t in text_list:
        if isinstance(t, str):
            cleaned_list.append(t)
        elif t is not None and not pd.isna(t):
            cleaned_list.append(str(t))  # تحويل تحويل الأرقام لنص

    # دمج النصوص
    tokens = " ".join(cleaned_list).split()

    # إنشاء n-grams
    if len(tokens) < n:
        return []  # ما يكفي عناصر للـ n-gram

    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams = [" ".join(ng) for ng in ngrams]

    counter = Counter(ngrams)
    return counter.most_common(top_k)


# =============================
# WordCloud generator
# =============================
def generate_wordcloud(text_list, title, save_path):
    # تنظيف القائمة من العناصر غير النصية أو NaN
    cleaned_list = []
    for t in text_list:
        if isinstance(t, str):
            cleaned_list.append(t)
        elif t is not None and not pd.isna(t):
            cleaned_list.append(str(t))  # نحول أي قيمة رقمية إلى نص

    # دمج النصوص في نص واحد
    text = " ".join(cleaned_list)

    # إنشاء WordCloud
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        font_path="/System/Library/Fonts/GeezaPro.ttc"
    )

    img = wc.generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(img, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================
# Main EDA Runner
# =============================
def run_eda():
    print("Loading cleaned datasets...")

    train = pd.read_csv(f"{DATA_PATH}/train_clean.csv")
    val = pd.read_csv(f"{DATA_PATH}/val_clean.csv")
    test = pd.read_csv(f"{DATA_PATH}/test_clean.csv")

    df = pd.concat([train, val, test], ignore_index=True)

    print(f"Total cleaned rows: {len(df)}")

    # Compute EDA statistics
    df["sentence_len"] = df["clean_text"].apply(sentence_length)
    df["avg_word_len"] = df["clean_text"].apply(avg_word_length)
    df["ttr"] = df["clean_text"].apply(type_token_ratio)
    df["punctuation"] = df["clean_text"].apply(punctuation_count)

    # Save numeric stats
    stats_path = f"{OUTPUT_DIR}/eda_stats.csv"
    df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"EDA statistics saved → {stats_path}")

    # Split into classes
    human_texts = df[df["label"] == "Human"]["clean_text"].tolist()
    ai_texts = df[df["label"] == "AI"]["clean_text"].tolist()

    # ======================
    # 1️⃣ Word Clouds
    # ======================
    print("Generating WordClouds ...")

    generate_wordcloud(human_texts, "Human Text WordCloud", f"{OUTPUT_DIR}/human_wordcloud.png")
    generate_wordcloud(ai_texts, "AI Text WordCloud", f"{OUTPUT_DIR}/ai_wordcloud.png")

    print("WordClouds saved!")

    # ======================
    # 2️⃣ N-Grams
    # ======================
    print("Extracting top N-grams ...")

    human_bigrams = extract_ngrams(human_texts, n=2)
    ai_bigrams = extract_ngrams(ai_texts, n=2)

    pd.DataFrame(human_bigrams, columns=["bigram", "count"]).to_csv(f"{OUTPUT_DIR}/human_bigrams.csv", index=False)
    pd.DataFrame(ai_bigrams, columns=["bigram", "count"]).to_csv(f"{OUTPUT_DIR}/ai_bigrams.csv", index=False)

    print("N-gram files saved.")

    # ======================
    # 3️⃣ Boxplots (sentence length, ttr, etc.)
    # ======================
    print("Generating comparison plots...")

    for col in ["sentence_len", "avg_word_len", "ttr", "punctuation"]:
        plt.figure(figsize=(8,5))
        df.boxplot(column=col, by="label")
        plt.title(f"{col} by Class")
        plt.suptitle("")
        plt.savefig(f"{OUTPUT_DIR}/{col}_boxplot.png", dpi=300)
        plt.close()

    print("Boxplot comparison saved!")

    print("\nEDA Completed Successfully!")
    print(f"All results stored under → {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
