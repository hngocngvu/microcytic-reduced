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


CBC_FEATURES = [
    "hemoglobin", "rbc", "hematocrit", "mcv", "mch",
    "mchc", "rdw", "platelet", "wbc",
]

IRON_FEATURES = [
    "ferritin", "hscrp", "stfr", "serum_iron", "tibc", "tsat", "stfr_index",
]


def load_nhanes(path):
    df = pd.read_csv(path)

    label = df["anemia_class"]
    df["ACD"] = label.str.contains("ACD", na=False).astype(int)
    df["IDA"] = label.str.contains("IDA", na=False).astype(int)

    all_features = CBC_FEATURES + IRON_FEATURES + ["gender"]

    X = df[all_features].copy()
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

    for train_idx, test_idx in loo.split(X):
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

    f1_acd = f1_score(y_true_all[:, 0], y_pred_all[:, 0], zero_division=0)
    f1_ida = f1_score(y_true_all[:, 1], y_pred_all[:, 1], zero_division=0)
    f1_macro = f1_score(y_true_all, y_pred_all, average="macro")

    auc = {}
    for j, label in enumerate(labels):
        try:
            auc[label] = roc_auc_score(y_true_all[:, j], y_prob_all[:, j])
        except Exception:
            auc[label] = float("nan")

    return {
        "model": model_name,
        "f1_macro": f1_macro,
        "f1_acd": f1_acd,
        "f1_ida": f1_ida,
        "auc": auc,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all,
    }


def build_models():
    return {
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


def run_full_evaluation(X, y):
    print(f"\n{'='*60}")
    print("  CBC + Iron (Oracle Upper Bound)")
    print(f"{'='*60}")
    print(f"  Features ({len(X.columns)}): {X.columns.tolist()}")

    models = build_models()
    results = {}
    for name, pipe in models.items():
        results[name] = loocv_evaluate(X, y, name, pipe)

    print(f"\n{'Model':<25} {'F1-ACD':>10} {'F1-IDA':>10} {'F1-macro':>10} {'AUC-ACD':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<25} {r['f1_acd']:>10.4f} {r['f1_ida']:>10.4f} {r['f1_macro']:>10.4f} {r['auc']['ACD']:>10.4f}")

    return results


def run_ablation(X, y):
    print(f"\n{'='*60}")
    print("  Ablation Study — Remove one iron marker at a time")
    print(f"{'='*60}")
    print("  Baseline: CBC + all iron markers")
    print("  Each row: one iron marker removed\n")

    best_model_name = "Logistic Regression"

    baseline = loocv_evaluate(X, y, "baseline", build_models()[best_model_name])
    baseline_f1 = baseline["f1_acd"]

    print(f"  Baseline F1-ACD ({best_model_name}): {baseline_f1:.4f}\n")

    ablation_results = []

    for feature in IRON_FEATURES:
        if feature not in X.columns:
            continue

        X_ablated = X.drop(columns=[feature])
        pipe = build_models()[best_model_name]
        result = loocv_evaluate(X_ablated, y, f"drop_{feature}", pipe)

        delta = result["f1_acd"] - baseline_f1
        ablation_results.append({
            "removed": feature,
            "f1_acd": result["f1_acd"],
            "f1_ida": result["f1_ida"],
            "delta_f1_acd": delta,
        })

    ablation_df = pd.DataFrame(ablation_results).sort_values("delta_f1_acd")

    print(f"{'Removed Feature':<20} {'F1-ACD':>10} {'Delta':>10} {'Impact':>10}")
    print("-" * 50)
    for _, row in ablation_df.iterrows():
        impact = "HIGH" if abs(row["delta_f1_acd"]) > 0.05 else "low"
        print(f"{row['removed']:<20} {row['f1_acd']:>10.4f} {row['delta_f1_acd']:>+10.4f} {impact:>10}")

    print(f"\n  Most important iron marker for ACD: "
          f"{ablation_df.iloc[0]['removed']} "
          f"(removing it drops F1-ACD by {ablation_df.iloc[0]['delta_f1_acd']:+.4f})")

    return ablation_df


def run_incremental(X, y):
    print(f"\n{'='*60}")
    print("  Incremental Study — Add iron markers one at a time to CBC")
    print(f"{'='*60}")

    best_model_name = "Logistic Regression"

    cbc_cols = [c for c in CBC_FEATURES if c in X.columns] + ["gender_Male"]

    X_cbc = X[cbc_cols]
    cbc_baseline = loocv_evaluate(X_cbc, y, "CBC-only", build_models()[best_model_name])
    print(f"\n  CBC-only F1-ACD: {cbc_baseline['f1_acd']:.4f}\n")

    incremental_results = []

    for feature in IRON_FEATURES:
        if feature not in X.columns:
            continue

        X_added = X[cbc_cols + [feature]]
        pipe = build_models()[best_model_name]
        result = loocv_evaluate(X_added, y, f"add_{feature}", pipe)

        gain = result["f1_acd"] - cbc_baseline["f1_acd"]
        incremental_results.append({
            "added": feature,
            "f1_acd": result["f1_acd"],
            "f1_ida": result["f1_ida"],
            "gain_f1_acd": gain,
        })

    inc_df = pd.DataFrame(incremental_results).sort_values("gain_f1_acd", ascending=False)

    print(f"{'Added Feature':<20} {'F1-ACD':>10} {'Gain':>10}")
    print("-" * 40)
    for _, row in inc_df.iterrows():
        print(f"{row['added']:<20} {row['f1_acd']:>10.4f} {row['gain_f1_acd']:>+10.4f}")

    print(f"\n  Most impactful single iron marker: "
          f"{inc_df.iloc[0]['added']} "
          f"(adding it improves F1-ACD by {inc_df.iloc[0]['gain_f1_acd']:+.4f})")

    return inc_df


if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, "nhanes_data", "data.csv")
    X, y = load_nhanes(data_path)

    print(f"Dataset: {len(X)} samples")
    print(f"ACD=1: {y['ACD'].sum()}  |  IDA=1: {y['IDA'].sum()}")

    full_results = run_full_evaluation(X, y)

    ablation_df = run_ablation(X, y)

    incremental_df = run_incremental(X, y)

    print(f"\n{'='*60}")
    print("  TAKEAWAY")
    print(f"{'='*60}")
    print("  CBC+Iron F1=1.0 is expected (labels derived from iron markers).")
    print("  The ablation/incremental results show WHICH iron markers carry")
    print("  the discriminative signal — useful for test panel optimization.")
