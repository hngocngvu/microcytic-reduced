import os
import sys

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def load_nhanes_cbc(path):
    df = pd.read_csv(path)

    label = df["anemia_class"]
    df["ACD"] = label.str.contains("ACD", na=False).astype(int)
    df["IDA"] = label.str.contains("IDA", na=False).astype(int)

    cbc_features = [
        "hemoglobin", "rbc", "hematocrit", "mcv", "mch",
        "mchc", "rdw", "platelet", "wbc", "gender"
    ]

    X = df[cbc_features].copy()
    X["gender"] = (X["gender"].str.lower() == "male").astype(int)
    X = X.rename(columns={"gender": "gender_Male"})
    y = df[["ACD", "IDA"]].copy()

    return X, y


def loocv_evaluate(X, y, model_name, pipeline):
    loo = LeaveOneOut()
    n = len(X)

    y_true_all = np.zeros((n, 2), dtype=int)
    y_pred_all = np.zeros((n, 2), dtype=int)
    y_prob_all = np.zeros((n, 2))

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred_all[test_idx] = pipeline.predict(X_test)
        y_true_all[test_idx] = y_test.values

        try:
            probas = pipeline.predict_proba(X_test)
            for j, p in enumerate(probas):
                y_prob_all[test_idx[0], j] = p[0][1]
        except Exception:
            y_prob_all[test_idx[0]] = y_pred_all[test_idx[0]]

    labels = ["ACD", "IDA"]
    print(f"\n{'='*50}")
    print(f"  {model_name} — LOOCV (n={n})")
    print(f"{'='*50}")

    for j, label in enumerate(labels):
        print(f"\n--- {label} ---")
        print(classification_report(
            y_true_all[:, j], y_pred_all[:, j],
            target_names=[f"Not {label}", label],
            digits=4, zero_division=0
        ))
        try:
            auc = roc_auc_score(y_true_all[:, j], y_prob_all[:, j])
            print(f"  AUC-ROC: {auc:.4f}")
        except Exception:
            print("  AUC-ROC: N/A")

    f1_macro = f1_score(y_true_all, y_pred_all, average="macro")
    f1_acd = f1_score(y_true_all[:, 0], y_pred_all[:, 0])
    f1_ida = f1_score(y_true_all[:, 1], y_pred_all[:, 1])

    print(f"\nOverall F1-macro: {f1_macro:.4f}")
    print(f"F1 ACD: {f1_acd:.4f}  |  F1 IDA: {f1_ida:.4f}")

    return {
        "model": model_name,
        "f1_macro": f1_macro,
        "f1_acd": f1_acd,
        "f1_ida": f1_ida,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all,
    }


if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, "data", "concat_for_eda.csv")
    X, y = load_nhanes_cbc(data_path)

    print(f"Dataset: {len(X)} samples")
    print(f"ACD=1: {y['ACD'].sum()}  |  IDA=1: {y['IDA'].sum()}")
    print(f"Features: {X.columns.tolist()}")

    models = {
        "Decision Tree": Pipeline([
            ("clf", MultiOutputClassifier(
                DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
            ))
        ]),
        "Random Forest": Pipeline([
            ("clf", MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            ))
        ]),
    }

    results = {}
    for name, pipe in models.items():
        results[name] = loocv_evaluate(X, y, name, pipe)

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<25} {'F1-macro':>10} {'F1-ACD':>10} {'F1-IDA':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['f1_macro']:>10.4f} {r['f1_acd']:>10.4f} {r['f1_ida']:>10.4f}")
