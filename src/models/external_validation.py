import os
import sys
import argparse

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)

from src.dataclass.modelfactory import ModelFactory, models_config
from src.functions.roc_pr import save_curves
from src.preprocess.nhanes import load_nhanes_for_validation

NHANES_PATH = os.path.join(BASE_DIR, "nhanes_data", "data.csv")
CONCAT_PATH = os.path.join(BASE_DIR, "data", "concat.csv")

SHARED_FEATURES = [
    "Hb", "RBC", "MCV", "MCHC", "RDW-CV",
    "Fe", "Ferritin", "TIBC", "TSAT (%)", "CRP",
    "Giới_nữ",
]


def load_concat():
    df = pd.read_csv(CONCAT_PATH)
    X = df.drop(columns=["ACD", "IDA"])
    y = df[["ACD", "IDA"]]
    return X, y


def print_external_result(result, model_name):
    print(f"\n{'='*50}")
    print(f"  {model_name} — External Validation (NHANES)")
    print(f"{'='*50}")
    print(f"Accuracy:      {result['accuracy']:.4f}")
    print(f"F1-macro:      {result['f1_macro']:.4f}")
    print(f"F1-micro:      {result['f1_micro']:.4f}")
    print(f"Hamming Loss:  {result['hamming_loss']:.4f}")
    print(f"Jaccard Score: {result['jaccard_score']:.4f}")
    for label, auc_val in result["auc"].items():
        print(f"AUC-ROC {label}: {auc_val:.4f}")
    print(f"\n{result['report']}")


def evaluate_on_nhanes(model, X_nhanes, y_nhanes):
    y_pred = model.predict(X_nhanes)
    y_true = y_nhanes.values

    y_prob = np.zeros((len(y_nhanes), 2))
    try:
        probas = model.predict_proba(X_nhanes)
        for j, p in enumerate(probas):
            y_prob[:, j] = p[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)

    labels = ["ACD", "IDA"]
    auc = {}
    for j, label in enumerate(labels):
        try:
            auc[label] = roc_auc_score(y_true[:, j], y_prob[:, j])
        except Exception:
            auc[label] = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "f1_micro": f1_score(y_true, y_pred, average='micro'),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard_score": jaccard_score(y_true, y_pred, average='samples', zero_division=0),
        "auc": auc,
        "report": classification_report(y_true, y_pred, target_names=labels, digits=4),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def run_single(model_name, X_train, y_train, X_nhanes, y_nhanes, mode):
    print(f"\n>>> Training {model_name} on concat ({len(X_train)} samples)...")
    factory = ModelFactory(model_name)

    if mode == "intersection":
        available = [c for c in SHARED_FEATURES if c in X_train.columns]
        X_train = X_train[available]
        X_nhanes = X_nhanes[available]
        print(f"  Intersection mode: {len(available)} features")
    else:
        print(f"  Full mode: {len(X_train.columns)} features (NaN-padded for NHANES)")

    final_model, best_params = factory.train_final_model(X_train, y_train)
    print(f"  Best params: {best_params}")

    print(f"\n>>> Evaluating on NHANES ({len(X_nhanes)} samples)...")
    result = evaluate_on_nhanes(final_model, X_nhanes, y_nhanes)
    print_external_result(result, model_name)

    return result


def run_all(X_train, y_train, X_nhanes, y_nhanes, mode):
    all_results = {}

    for name in models_config:
        all_results[name] = run_single(name, X_train.copy(), y_train, X_nhanes.copy(), y_nhanes, mode)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — External Validation (mode={mode})")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Acc':>8} {'F1-mac':>8} {'F1-mic':>8} {'Hamming':>8} {'AUC-ACD':>8} {'AUC-IDA':>8}")
    print("-" * 67)
    for name, r in all_results.items():
        print(
            f"{name:<15} "
            f"{r['accuracy']:>8.4f} "
            f"{r['f1_macro']:>8.4f} "
            f"{r['f1_micro']:>8.4f} "
            f"{r['hamming_loss']:>8.4f} "
            f"{r['auc'].get('ACD', float('nan')):>8.4f} "
            f"{r['auc'].get('IDA', float('nan')):>8.4f}"
        )

    save_dir = os.path.join("notebooks", "png", f"external_{mode}")
    save_curves(all_results, ["ACD", "IDA"], save_dir=save_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, choices=models_config.keys())
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "intersection"],
        default="full",
        help="full = NaN-pad missing NHANES features; intersection = use only shared features",
    )

    args = parser.parse_args()

    X_train, y_train = load_concat()
    X_nhanes, y_nhanes = load_nhanes_for_validation(NHANES_PATH, X_train.columns.tolist())

    print(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"NHANES data:   {len(X_nhanes)} samples")
    print(f"Mode:          {args.mode}")
    print(f"ACD=1 (train): {y_train['ACD'].sum()}  |  IDA=1 (train): {y_train['IDA'].sum()}")
    print(f"ACD=1 (test):  {y_nhanes['ACD'].sum()}  |  IDA=1 (test):  {y_nhanes['IDA'].sum()}")

    if args.all:
        run_all(X_train, y_train, X_nhanes, y_nhanes, args.mode)
    else:
        run_single(args.model_name, X_train, y_train, X_nhanes, y_nhanes, args.mode)
