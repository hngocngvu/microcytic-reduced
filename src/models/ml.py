import os
import sys
import argparse

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd

from src.dataclass.modelfactory import ModelFactory, models_config
from src.functions.function import print_result, friedman_compare
from src.functions.roc_pr import save_curves


def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "concat.csv"))
    X = df.drop(columns=["ACD", "IDA"])
    y = df[["ACD", "IDA"]]
    return X, y


def print_loocv_result(result):
    print(f"\n{'='*50}")
    print(f"  {result['model_name']} — LOOCV (n={len(result['y_true'])})")
    print(f"{'='*50}")
    print(f"Accuracy:      {result['accuracy']:.4f}")
    print(f"F1-macro:      {result['f1_macro']:.4f}")
    print(f"F1-micro:      {result['f1_micro']:.4f}")
    print(f"F1-weighted:   {result['f1_weighted']:.4f}")
    print(f"Hamming Loss:  {result['hamming_loss']:.4f}")
    print(f"Jaccard Score: {result['jaccard_score']:.4f}")
    for label, auc_val in result["auc"].items():
        print(f"AUC-ROC {label}: {auc_val:.4f}")
    print(f"Time:          {result['training_time_sec']:.1f}s")
    print(f"\n{result['report']}")


def run_single(model_name, X, y):
    print(f"\n>>> Training {model_name}...")
    factory = ModelFactory(model_name)

    result = factory.loocv_evaluate(X, y)
    print_loocv_result(result)

    print(f"\n>>> Training final {model_name} model on all data...")
    final_model, best_params = factory.train_final_model(X, y)
    print(f"  Best params: {best_params}")

    factory.explain_with_shap(X, feature_names=X.columns.tolist(), trained_model=final_model)

    return result


def run_all(X, y):
    all_results = {}

    for name in models_config:
        all_results[name] = run_single(name, X, y)

    print(f"\n{'='*60}")
    print("  SUMMARY — LOOCV")
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

    print(f"\n{'='*60}")
    print("  Friedman Test")
    print(f"{'='*60}")
    cv_scores = {name: r["per_sample_exact_match"].tolist() for name, r in all_results.items()}
    friedman_compare(cv_scores)

    labels = ["ACD", "IDA"]
    save_curves(all_results, labels)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-name",
        type=str,
        choices=models_config.keys(),
        help="Train a single model",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all models and compare",
    )

    args = parser.parse_args()

    X, y = load_data()
    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"ACD=1: {y['ACD'].sum()}  |  IDA=1: {y['IDA'].sum()}")
    print(f"Features: {X.columns.tolist()}")

    if args.all:
        run_all(X, y)
    else:
        run_single(args.model_name, X, y)
