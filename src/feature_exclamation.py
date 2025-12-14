import re
import pandas as pd

def apply_exclamation_feature(df):
    """
    Adds a column 'exclamation_count' that counts occurrences of '!' in each text.
    """
    
    def count_exclamation(text):
        if not isinstance(text, str):
            return 0
        return len(re.findall(r"!", text))

    df["exclamation_count"] = df["text"].apply(count_exclamation)
    return df
