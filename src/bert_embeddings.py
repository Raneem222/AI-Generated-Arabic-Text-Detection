from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_embedding(text):
    """Convert single text into BERT embedding"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        embedding = hidden.mean(dim=1).squeeze().cpu().numpy()  # (768,)
    
    return embedding


def generate_bert_embeddings(input_path, output_path):
    print(f"\nGenerating BERT embeddings for → {input_path}")

    df = pd.read_csv(input_path)
    embeddings = []

    for text in tqdm(df["clean_text"].fillna("").tolist()):
        emb = get_embedding(text)
        embeddings.append(emb)

    emb_matrix = np.vstack(embeddings)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, emb_matrix)

    print(f"Saved embeddings → {output_path}.npy")
