import re
import pandas as pd

def count_interjections(text):
    if not isinstance(text, str):
        return 0

    # أنماط لغوية تمثل interjections في العربية
    patterns = [
        r"\bآه+\b",
        r"\bأوه+\b",
        r"\bإيه+\b",
        r"\bيا\s+\w+\b",       # يا + اسم
        r"\bها+?\b",           # هااا
        r"\bأا+\b",            # أاا
        r"\bأوي+\b",
        r"\bأوه+\b",
        r"\bهه+\b",
        r"\bأف+\b",
        r"\bأوو+\b",
        r"\bنعم+\b",
        r"\bلا+\b"
    ]

    combined = "(" + "|".join(patterns) + ")"
    return len(re.findall(combined, text))


def apply_interjection_feature(df):
    df["interjection_count"] = df["text"].apply(count_interjections)
    return df
