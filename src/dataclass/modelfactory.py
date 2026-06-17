import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.functions.function import make_pipeline 

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, jaccard_score
import time
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier

models_config = {

    "XGBoost": {
        "model": make_pipeline(XGBClassifier(eval_metric='logloss', random_state=42)),
        
        "params": {
            "clf__estimator__n_estimators": [100, 200],
            "clf__estimator__learning_rate": [0.05, 0.1],
            "clf__estimator__max_depth": [3, 5]
        }
    },

    "LightGBM": {
        "model": make_pipeline(LGBMClassifier(random_state=42, verbose=-1)),

        "params": {
            "clf__estimator__n_estimators": [100, 200],
            "clf__estimator__learning_rate": [0.05, 0.1],
            "clf__estimator__max_depth": [3, 5]
        }
    },

    "CatBoost": {
        "model": make_pipeline(CatBoostClassifier(random_state=42, verbose=0)),

        "params": {
            "clf__estimator__iterations": [100, 200],
            "clf__estimator__learning_rate": [0.05, 0.1],
            "clf__estimator__depth": [3, 5]
        }
    }
}


class ModelFactory():
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        if self.model_name in models_config:
            return (
                models_config[self.model_name]["model"],
                models_config[self.model_name]["params"]
            )
        else:
            raise ValueError(f"Model {self.model_name} not found.")

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        model, param_grid = self.get_model()
        start_time = time.time()

        cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')
        
        if param_grid:
            if self.model_name in ["LightGBM", "XGBoost", "CatBoost"]:
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=5,
                    cv=cv_strategy,
                    scoring=scorer,
                    n_jobs=-1,
                    random_state=42,
                    return_train_score=True,
                    verbose= 0
                )
            else:
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_strategy,
                    scoring=scorer,
                    n_jobs=-1,
                    return_train_score=True,
                    verbose= 0
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
            
        else:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=scorer,
                n_jobs=-1
            )
            cv_mean_f1 = cv_scores.mean()
            cv_std_f1 = cv_scores.std()
            cv_fold_scores = cv_scores.tolist()
            
            model.fit(X_train, y_train)
            best_model = model
            best_params = "default"
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Predict
        y_pred = best_model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        jaccard = jaccard_score(y_test, y_pred, average="samples", zero_division=0)
        hamming= hamming_loss(y_test, y_pred)
        
        # Save model
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
            "saved_model": model_filename
        }
    
    def explain_with_shap(self, X_test, feature_names, trained_model, save_dir="notebooks/png/shap"):
            """
            Generate SHAP explanations for tree-based models.
            For MultiOutputClassifier, we explain each output separately.
            """
            
            if self.model_name not in ["XGBoost", "LightGBM", "CatBoost"]:
                print(f"SHAP explanation skipped: {self.model_name} is not a tree-based model.")
                return
    
            os.makedirs(save_dir, exist_ok=True)

            model= trained_model 
            # Load the trained model
            
            # For MultiOutputClassifier, we need to explain each estimator separately
            output_names = [f"{self.model_name}_ACD", f"{self.model_name}_IDA"]  # Update based on your targets

            multi_output_model = model.named_steps["clf"]

            if "scaler" in model.named_steps:
                X_input= model.named_steps["scaler"].transform(X_test)
            else:
                X_input= X_test.values
            
            for _, (estimator, output_name) in enumerate(zip(multi_output_model.estimators_, output_names)):
                print(f"\nGenerating SHAP explanations for output: {output_name}")
                
                try:
                    # Create TreeExplainer for individual estimator
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_input)

                    # For binary classification, shap_values might be a 2D array
                    # For multiclass, it's a list of arrays
                    if isinstance(shap_values, list):
                        # Multiclass case - use class 1 (positive class)
                        shap_vals_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_vals_to_plot = shap_values

                    
                    # Summary plot
                    
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_vals_to_plot,
                        X_input,
                        feature_names=feature_names,
                        show=False
                    )
                    plt.title(f"SHAP Summary - {output_name}", fontsize= 14)
                    output_file = f"{save_dir}/summary_plot_{output_name}.png"
                    plt.savefig(output_file, bbox_inches="tight", dpi=300)
                    print(f"  Saved: {output_file}")
                    
                    # Feature importance plot
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_vals_to_plot,
                        X_input,
                        feature_names=feature_names,
                        plot_type="bar",
                        show=False
                    )
                    plt.title(f"SHAP Feature Importance - {output_name}", fontsize= 14)
                    output_file = f"{save_dir}/feature_importance_{output_name}.png"
                    plt.savefig(output_file, bbox_inches="tight", dpi=300)
                    print(f"  Saved: {output_file}")
                    
                except Exception as e:
                    print(f"  Error generating SHAP plots for {output_name}: {str(e)}")
            
            print(f"\nSHAP analysis complete. Plots saved to {save_dir}/")
            
            """
            shap.initjs()

            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values.values[0],
                X_test.iloc[0]
            )

            shap.save_html(f"{save_dir}/force_plot_sample0.html", force_plot)

            print(f"SHAP plots saved to {save_dir}")
            """
    