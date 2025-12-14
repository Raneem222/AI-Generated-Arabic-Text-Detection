import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------
# Utility: Load features
# ---------------------------------------------------
def load_features():
    train = pd.read_csv("data/features/train_features.csv")
    val = pd.read_csv("data/features/val_features.csv")
    test = pd.read_csv("data/features/test_features.csv")

    return train, val, test


# ---------------------------------------------------
# Utility: Split X and y
# ---------------------------------------------------
def split_xy(df):

    # Remove label and raw text
    X = df.drop(columns=["label", "text"], errors="ignore")

    # Drop any leftover non-numeric columns
    X = X.select_dtypes(include=["number"])

    # Convert label to numeric
    y = (df["label"] == "AI").astype(int)

    return X, y



# ---------------------------------------------------
# Evaluation function
# ---------------------------------------------------
import csv
import os

def evaluate_model(model_name, y_true, y_pred, y_prob, save_path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)

    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    results = {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": conf_mat,
    }

    print(f"\n=== {model_name} Evaluation ===")
    print(results)

    # -----------------------------
    # Save results into CSV file
    # -----------------------------
    results_dir = "reports/models"
    os.makedirs(results_dir, exist_ok=True)

    csv_path = f"{results_dir}/model_results.csv"

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["model", "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"])

        writer.writerow([
            model_name,
            acc,
            prec,
            rec,
            f1,
            roc,
            conf_mat
        ])

    return results

    # Save evaluation
    os.makedirs("reports/models", exist_ok=True)
    pd.DataFrame([results]).to_csv(f"reports/models/{model_name}_metrics.csv", index=False)

    print(f"\n=== {model_name} Evaluation ===")
    print(results)

    return results


# ---------------------------------------------------
# Baseline Models 
# ---------------------------------------------------
def train_baseline_models():
    print("Loading features ...")
    train, val, test = load_features()

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    # ------------------------
    # Logistic Regression
    # ------------------------
    print("\nTraining Logistic Regression ...")
    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    evaluate_model("logistic_regression", y_test, y_pred, y_prob, save_path="reports/models")

    # ------------------------
    # Naive Bayes
    # ------------------------
    print("\nTraining Naive Bayes ...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]

    evaluate_model("naive_bayes", y_test, y_pred, y_prob, save_path="reports/models")

    print("\nBaseline Models Completed.")
if __name__ == "__main__":
    print("\n=== Loading Features ===")
    train_df, val_df, test_df = load_features()

    print("Splitting X and y ...")
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    # ================================
    # Baseline Model: Naive Bayes
    # ================================
    print("\n=== Training Baseline Model (Naive Bayes) ===")
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]

    evaluate_model(
        model_name="Naive_Bayes",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        save_path="reports/models"
    )

    # ================================
    # Logistic Regression
    # ================================
    print("\n=== Training Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    evaluate_model(
        model_name="Logistic_Regression",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        save_path="reports/models"
    )

    print("\nAll models trained successfully!")
    # ============================================
    # 3) SVM CLASSIFIER (with tuning)
    # ============================================
    print("\n=== Training SVM Model ===")

    from sklearn.svm import LinearSVC

    svm_clf = LinearSVC()
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)
    y_prob = [0] * len(y_pred)   # LinearSVC doesn't output proba

    evaluate_model(
    model_name="SVM",
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
    save_path="reports/models/svm_results.csv"
    )

    # ============================================
    # 4) RANDOM FOREST CLASSIFIER (with tuning)
    # ============================================
    print("\n=== Training Random Forest Model ===")

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)[:, 1]

evaluate_model(
    model_name="Random_Forest",
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
    save_path="reports/models/rf_results.csv"
)
