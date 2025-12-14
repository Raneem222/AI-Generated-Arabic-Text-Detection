import pandas as pd
import re
import os

# ======================================================
# Load lexicons (positive & negative)
# ======================================================
def load_lexicon(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        words = set([w.strip() for w in f.readlines() if w.strip()])
    return words


# ======================================================
# Feature: Sentiment Score
# ======================================================
def apply_sentiment_feature(df):
    positive_words = load_lexicon("data/external/sentiment_lexicon/positive_words.txt")
    negative_words = load_lexicon("data/external/sentiment_lexicon/negative_words.txt")

    def sentiment_calc(text):
        if not isinstance(text, str):
            return 0, 0, 0
        
        tokens = re.findall(r"\w+", text)

        pos_count = sum(1 for w in tokens if w in positive_words)
        neg_count = sum(1 for w in tokens if w in negative_words)
        score = pos_count - neg_count

        return pos_count, neg_count, score

    df["positive_count"], df["negative_count"], df["sentiment_score"] = zip(
        *df["text"].apply(sentiment_calc)
    )

    return df
