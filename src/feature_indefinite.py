import re

# كلمات نكرة شائعة في النصوص الأكاديمية العربية
INDEFINITE_COMMON = {
    "شيء", "أحد", "بعض", "كل", "كثير", "قليل", "عدة", "نوع", "شكل", "طريقة",
    "مجال", "مستوى", "جانب", "موضوع", "هدف", "سبب", "ظاهرة", "فكرة", "مكان",
    "وقت", "حالة", "مرحلة", "خيار", "عدد", "مجموعة", "عملية", "تطبيق", "تجربة",
    "مفهوم", "نتيجة", "دراسة", "بحث", "عامل", "عوامل"
}

# أوزان صرفية تقترح النكرة
PATTERN_SUFFIXES = ["ة", "ات", "ان", "ين", "ون", "ي", "ية"]

def is_indefinite_word(word):
    """تحديد ما إذا كانت الكلمة نكرة باستخدام قواعد متعددة."""

    if not isinstance(word, str) or word.strip() == "":
        return False

    w = word.strip()

    # 1) تنوين = نكرة واضحة
    if re.search(r"[ًٌٍ]$", w):
        return True

    # 2) كلمات شائعة نكرة
    if w in INDEFINITE_COMMON:
        return True

    # 3) كلمة بدون "ال" التعريف
    if not w.startswith("ال"):

        # 4) نهايات صرفية توحي بالنكرة
        for suf in PATTERN_SUFFIXES:
            if w.endswith(suf):
                return True

    return False


def apply_indefinite_feature(df):
    """إضافة عمود indefinite_count للداتا."""
    def count_indefinite(text):
        if not isinstance(text, str):
            return 0

        words = re.findall(r"\w+", text)
        return sum(1 for w in words if is_indefinite_word(w))

    df["indefinite_count"] = df["text"].apply(count_indefinite)
    return df
