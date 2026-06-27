import os
import time

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from catboost import CatBoostClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from xgboost import XGBClassifier

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.functions.function import make_pipeline

models_config = {
    "XGBoost": {
        "model": make_pipeline(XGBClassifier(eval_metric='logloss', random_state=42)),
        "params": {
            "clf__estimator__n_estimators": [100, 200],
            "clf__estimator__learning_rate": [0.05, 0.1],
            "clf__estimator__max_depth": [3, 5],
            "clf__estimator__scale_pos_weight": [1, 3, 6],
        },
    },
    "LightGBM": {
        "model": make_pipeline(LGBMClassifier(random_state=42, verbose=-1, is_unbalanced=True)),
        "params": {
            "clf__estimator__n_estimators": [100, 200],
            "clf__estimator__learning_rate": [0.05, 0.1],
            "clf__estimator__max_depth": [3, 5],
        },
    },
    "CatBoost": {
        "model": make_pipeline(CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')),
        "params": {
            "clf__estimator__iterations": [100, 200],
            "clf__estimator__learning_rate": [0.05, 0.1],
            "clf__estimator__depth": [3, 5],
        },
    },
}


class ModelFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        if self.model_name in models_config:
            return (
                models_config[self.model_name]["model"],
                models_config[self.model_name]["params"],
            )
        raise ValueError(f"Model {self.model_name} not found.")

    def _fresh_model(self):
        cfg = models_config[self.model_name]
        from sklearn.base import clone
        return clone(cfg["model"]), cfg["params"]

    def loocv_evaluate(self, X, y):
        loo = LeaveOneOut()
        n = len(X)

        y_true_all = np.zeros((n, 2), dtype=int)
        y_pred_all = np.zeros((n, 2), dtype=int)
        y_prob_all = np.zeros((n, 2))

        scorer = make_scorer(f1_score, average='macro')
        start_time = time.time()

        for i, (train_idx, test_idx) in enumerate(loo.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model, param_grid = self._fresh_model()

            inner_cv = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=5,
                cv=inner_cv,
                scoring=scorer,
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_

            y_pred_all[test_idx] = best.predict(X_test)
            y_true_all[test_idx] = y_test.values

            try:
                probas = best.predict_proba(X_test)
                for j, p in enumerate(probas):
                    y_prob_all[test_idx[0], j] = p[0][1]
            except Exception:
                y_prob_all[test_idx[0]] = y_pred_all[test_idx[0]]

            if (i + 1) % 50 == 0:
                print(f"  LOOCV fold {i + 1}/{n}")

        training_time = time.time() - start_time

        acc = accuracy_score(y_true_all, y_pred_all)
        f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
        f1_micro = f1_score(y_true_all, y_pred_all, average='micro')
        f1_weighted = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        jaccard = jaccard_score(y_true_all, y_pred_all, average='samples', zero_division=0)
        hamming = hamming_loss(y_true_all, y_pred_all)

        auc = {}
        labels = ["ACD", "IDA"]
        for j, label in enumerate(labels):
            try:
                auc[label] = roc_auc_score(y_true_all[:, j], y_prob_all[:, j])
            except Exception:
                auc[label] = float("nan")

        exact_match = (y_pred_all == y_true_all).all(axis=1).astype(int)

        report = classification_report(
            y_true_all, y_pred_all,
            target_names=labels, digits=4
        )

        return {
            "model_name": self.model_name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "jaccard_score": jaccard,
            "hamming_loss": hamming,
            "auc": auc,
            "report": report,
            "training_time_sec": training_time,
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "y_prob": y_prob_all,
            "per_sample_exact_match": exact_match,
        }

    def train_final_model(self, X, y, save=True):
        model, param_grid = self._fresh_model()

        cv_strategy = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')

        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=5,
            cv=cv_strategy,
            scoring=scorer,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X, y)
        best_model = search.best_estimator_

        if save:
            model_filepath = os.path.join("artifacts", f"{self.model_name}_best_model.pkl")
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(best_model, model_filepath)
            print(f"  Saved: {model_filepath}")

        return best_model, search.best_params_

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        model, param_grid = self.get_model()
        start_time = time.time()

        cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')

        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=5,
            cv=cv_strategy,
            scoring=scorer,
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
            verbose=0,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

        cv_mean_f1 = search.best_score_
        cv_std_f1 = search.cv_results_['std_test_score'][search.best_index_]
        cv_fold_scores = [
            search.cv_results_[f'split{i}_test_score'][search.best_index_]
            for i in range(cv_strategy.get_n_splits())
        ]

        end_time = time.time()
        training_time = end_time - start_time

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        jaccard = jaccard_score(y_test, y_pred, average="samples", zero_division=0)
        hamming = hamming_loss(y_test, y_pred)

        model_filename = f"{self.model_name}_best_model.pkl"
        model_filepath = f"artifacts/{model_filename}"
        joblib.dump(best_model, model_filepath)

        return {
            "model_name": self.model_name,
            "best_model": best_model,
            "best_params": best_params,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "jaccard_score": jaccard,
            "hamming_loss": hamming,
            "cv_mean_f1": cv_mean_f1,
            "cv_std_f1": cv_std_f1,
            "cv_fold_scores": cv_fold_scores,
            "training_time_sec": training_time,
            "report": classification_report(y_test, y_pred, target_names=["ACD", "IDA"], digits=4),
            "saved_model": model_filename,
        }

    def explain_with_shap(self, X_test, feature_names, trained_model, save_dir="notebooks/png/shap"):
        if self.model_name not in ["XGBoost", "LightGBM", "CatBoost"]:
            print(f"SHAP explanation skipped: {self.model_name} is not a tree-based model.")
            return

        os.makedirs(save_dir, exist_ok=True)

        model = trained_model
        output_names = [f"{self.model_name}_ACD", f"{self.model_name}_IDA"]

        multi_output_model = model.named_steps["clf"]

        if "scaler" in model.named_steps:
            X_input = model.named_steps["scaler"].transform(X_test)
        else:
            X_input = X_test.values if hasattr(X_test, 'values') else X_test

        for _, (estimator, output_name) in enumerate(zip(multi_output_model.estimators_, output_names)):
            print(f"\nGenerating SHAP explanations for output: {output_name}")

            try:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_input)

                if isinstance(shap_values, list):
                    shap_vals_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    shap_vals_to_plot = shap_values

                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals_to_plot,
                    X_input,
                    feature_names=feature_names,
                    show=False,
                )
                plt.title(f"SHAP Summary - {output_name}", fontsize=14)
                output_file = f"{save_dir}/summary_plot_{output_name}.png"
                plt.savefig(output_file, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"  Saved: {output_file}")

                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals_to_plot,
                    X_input,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False,
                )
                plt.title(f"SHAP Feature Importance - {output_name}", fontsize=14)
                output_file = f"{save_dir}/feature_importance_{output_name}.png"
                plt.savefig(output_file, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"  Saved: {output_file}")

            except Exception as e:
                print(f"  Error generating SHAP plots for {output_name}: {str(e)}")

        print(f"\nSHAP analysis complete. Plots saved to {save_dir}/")
