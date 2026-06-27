import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

SAVE_DIR = os.path.join("notebooks", "png")


def plot_multi_roc(results_dict, labels):
    figs = {}

    for i, label in enumerate(labels):
        fig, ax = plt.subplots()

        for name, result in results_dict.items():
            y_true = result["y_true"][:, i]
            y_prob = result["y_prob"][:, i]

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curve - {label}")
        ax.legend()

        figs[label] = fig

    return figs


def plot_multi_pr(results_dict, labels):
    figs = {}

    for i, label in enumerate(labels):
        fig, ax = plt.subplots()

        for name, result in results_dict.items():
            y_true = result["y_true"][:, i]
            y_prob = result["y_prob"][:, i]

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve - {label}")
        ax.legend()

        figs[label] = fig

    return figs


def save_curves(results_dict, labels, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    fig_roc = plot_multi_roc(results_dict, labels)
    fig_pr = plot_multi_pr(results_dict, labels)

    for label, fig in fig_roc.items():
        filename = f"roc_curve_{label.replace(' ', '_').lower()}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    for label, fig in fig_pr.items():
        filename = f"pr_curve_{label.replace(' ', '_').lower()}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"ROC/PR curves saved to {save_dir}/")
