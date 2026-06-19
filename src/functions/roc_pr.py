import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


save_dir= os.path.join("notebooks", "png")
pr_paths= [os.path.join(save_dir, "pr_curve_cb.png"), os.path.join(save_dir, "pr_curve_xgb.png"), os.path.join(save_dir, "pr_curve_lgbm.png")]
roc_paths= [os.path.join(save_dir, "roc_auc_cb.png"), os.path.join(save_dir, "roc_auc_xgb.png"), os.path.join(save_dir, "roc_auc_lgbm.png")]

def plot_multi_roc(models_dict, y_true, X_test):
    labels = y_true.columns.tolist()  # ["ACD", "IDA", "Thal"]
    
    figs = {}

    for i, label in enumerate(labels):
        fig, ax = plt.subplots()

        for name, path in models_dict.items():
            model = joblib.load(path)
            y_probs = model.predict_proba(X_test)

            y_prob = y_probs[i][:, 1]   # lấy class 1 của label i
            y_true_label = y_true.iloc[:, i]

            fpr, tpr, _ = roc_curve(y_true_label, y_prob)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curve - {label}")
        ax.legend()

        figs[label] = fig

    return figs


def plot_multi_pr(models_dict, y_true, X_test):
    labels = y_true.columns.tolist()
    
    figs = {}

    for i, label in enumerate(labels):
        fig, ax = plt.subplots()

        for name, path in models_dict.items():
            model = joblib.load(path)
            y_probs = model.predict_proba(X_test)

            y_prob = y_probs[i][:, 1]
            y_true_label = y_true.iloc[:, i]

            precision, recall, _ = precision_recall_curve(y_true_label, y_prob)
            ap = average_precision_score(y_true_label, y_prob)

            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve - {label}")
        ax.legend()

        figs[label] = fig

    return figs

if __name__ == "__main__":
    data_shap= os.path.join("data", "test_set_reduced_features.csv")

    df_shap= pd.read_csv(data_shap)

    X = df_shap.drop(columns=["ACD", "IDA"])
    y = df_shap[["ACD", "IDA"]]


    models = {
    "CatBoost": os.path.join("artifacts", "CatBoost_best_model.pkl"),
    "XGBoost": os.path.join("artifacts", "XGBoost_best_model.pkl"),
    "LightGBM": os.path.join("artifacts", "LightGBM_best_model.pkl")}


    fig_roc = plot_multi_roc(models, y, X)
    fig_pr = plot_multi_pr(models, y, X)

    os.makedirs(save_dir, exist_ok=True)

    # Save ROC
    for label, fig in fig_roc.items():
        filename = f"roc_curve_{label.replace(' ', '_').lower()}.png"
        fig.savefig(os.path.join(save_dir, filename),
                    dpi=300,
                    bbox_inches="tight")
        plt.show()

    # Save PR
    for label, fig in fig_pr.items():
        filename = f"pr_curve_{label.replace(' ', '_').lower()}.png"
        fig.savefig(os.path.join(save_dir, filename),
                    dpi=300,
                    bbox_inches="tight")
        plt.show()


