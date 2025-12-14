from bert_embeddings import generate_bert_embeddings

if __name__ == "__main__":

    generate_bert_embeddings("data/processed/train_clean.csv",
                             "data/bert/train_embeddings")

    generate_bert_embeddings("data/processed/val_clean.csv",
                             "data/bert/val_embeddings")

    generate_bert_embeddings("data/processed/test_clean.csv",
                             "data/bert/test_embeddings")

    print("\nAll BERT embeddings generated successfully!")
