import os
import re

def remove_diacritics(text):
    arabic_diacritics = re.compile(r"""
        ّ    | 
        َ    |
        ً    |
        ُ    |
        ٌ    |
        ِ    |
        ٍ    |
        ْ    |
        ـ     
    """, re.VERBOSE)
    return re.sub(arabic_diacritics, "", text)

def normalize_text(text):
    text = text.strip()
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = re.sub(r"[^ء-ي\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_stopwords():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # يرجع لمجلد ai_text_detector
    stop_path = os.path.join(BASE_DIR, "data/external/sentiment_lexicon/stopwords.txt")


    with open(stop_path, "r", encoding="utf-8") as f:
        return set([w.strip() for w in f.readlines() if w.strip()])

STOPWORDS = load_stopwords()

def remove_stopwords(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def simple_stem(word):
    suffixes = ["ات", "ون", "ين", "ان", "ه", "ها", "هم", "كم", "نا"]
    prefixes = ["ال", "و", "ف", "ب", "ل"]
    for s in suffixes:
        if word.endswith(s):
            word = word[:-len(s)]
    for p in prefixes:
        if word.startswith(p):
            word = word[len(p):]
    return word

def stem_text(text):
    tokens = text.split()
    tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_diacritics(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text
