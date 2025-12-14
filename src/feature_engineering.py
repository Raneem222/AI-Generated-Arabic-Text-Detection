import re

def exclamation_count(text):
    """Count number of exclamation marks in a text."""
    if not isinstance(text, str):
        return 0
    return text.count("!")
