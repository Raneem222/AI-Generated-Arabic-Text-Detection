import re
import pandas as pd

def average_word_length(text: str) -> float:
    """
    حساب متوسط طول الكلمات في النص.
    - يحسب طول كل كلمة بدون علامات ترقيم.
    - يعمل على نصوص عربية/إنجليزية.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # إزالة علامات الترقيم
    cleaned = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)

    words = cleaned.split()
    if not words:
        return 0.0

    total_length = sum(len(w) for w in words)
    return total_length / len(words)


def apply_avg_word_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    تطبيق الفيتشر على داتا فريم كامل.
    يضيف عمود جديد: avg_word_length
    """
    df["avg_word_length"] = df["text"].apply(average_word_length)
    return df
