from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    KFold
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import joblib
import pandas as pd 
import argparse
from sklearn.model_selection import train_test_split
import shap
import os
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder


def print_result(result):
    print(f"Best params: {result['best_params']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"F1-macro: {result['f1_macro']:.4f}")
    print(f"CV F1: {float(result['cv_mean_f1']):.4f} ± {float(result['cv_std_f1']):.4f}")
    print(f"Training time: {result['training_time_sec']:.2f} sec")

    print("\nClassification Report:\n")
    print(result["report"])

    print(f"Saved model    : {result['saved_model']}")


models_config = {
    "RF": {
        "model": MultiOutputClassifier(
            RandomForestClassifier(random_state=42)
        ),
        "params": {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [5, 10, None]
        }
    },

    "XGBoost": {
        "model": MultiOutputClassifier(
            XGBClassifier(eval_metric='logloss', random_state=42)
        ),
        "params": {
            "estimator__n_estimators": [100, 200],
            "estimator__learning_rate": [0.05, 0.1],
            "estimator__max_depth": [3, 5]
        }
    },

    "LightGBM": {
        "model": MultiOutputClassifier(
            LGBMClassifier(random_state=42)
        ),
        "params": {
            "estimator__n_estimators": [100],
            "estimator__learning_rate": [0.05],
            "estimator__max_depth": [-1, 5]
        }
    },

    "LR": {
        "model": MultiOutputClassifier(
            LogisticRegression(max_iter=1000)
        ),
        "params": {
            "estimator__C": [0.1, 1, 10]
        }
    },

    "SVM": {
        "model": MultiOutputClassifier(
            SVC(probability=True)
        ),
        "params": {
            "estimator__C": [0.1, 1, 10],
            "estimator__kernel": ["rbf"]
        }
    },

    "KNN": {
        "model": MultiOutputClassifier(
            KNeighborsClassifier()
        ),
        "params": {
            "estimator__n_neighbors": [3, 5, 7]
        }
    },

    "NaiveBayes": {
        "model": MultiOutputClassifier(
            GaussianNB()
        ),
        "params": {}
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

        cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')
        
        if param_grid:
            if self.model_name in ["LightGBM", "XGBoost", "RF"]:
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=5,
                    cv=cv_strategy,
                    scoring=scorer,
                    n_jobs=-1,
                    random_state=42,
                    return_train_score=True
                )
            else:
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_strategy,
                    scoring=scorer,
                    n_jobs=-1,
                    return_train_score=True
                )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            cv_mean_f1 = search.best_score_
            cv_std_f1 = search.cv_results_['std_test_score'][search.best_index_]
            
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
            "cv_mean_f1": cv_mean_f1,
            "cv_std_f1": cv_std_f1,
            "training_time_sec": training_time,
            "report": classification_report(y_test, y_pred, target_names=["ACD", "IDA", "Thal"], digits=4),
            "saved_model": model_filename
        }
    
    def explain_with_shap(self, X_test, feature_names, save_dir="notebooks/eda/shap"):
            """
            Generate SHAP explanations for tree-based models.
            For MultiOutputClassifier, we explain each output separately.
            """
            
            if self.model_name not in ["RF", "XGBoost", "LightGBM"]:
                print(f"SHAP explanation skipped: {self.model_name} is not a tree-based model.")
                return
    
            os.makedirs(save_dir, exist_ok=True)
    
            # Load the trained model
            model = joblib.load(f"artifacts/{self.model_name}_best_model.pkl")
            
            # For MultiOutputClassifier, we need to explain each estimator separately
            output_names = [f"{self.model_name}_ACD", f"{self.model_name}_IDA", f"{self.model_name}_Thal"]  # Update based on your targets
            
            for i, (estimator, output_name) in enumerate(zip(model.estimators_, output_names)):
                print(f"\nGenerating SHAP explanations for output: {output_name}")
                
                try:
                    # Create TreeExplainer for individual estimator
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_test)
                    
                    # Summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values,
                        X_test,
                        feature_names=feature_names,
                        show=False
                    )
                    plt.title(f"SHAP Summary - {output_name}")
                    output_file = f"{save_dir}/summary_plot_{output_name}.png"
                    plt.savefig(output_file, bbox_inches="tight", dpi=300)
                    plt.close()
                    print(f"  Saved: {output_file}")
                    
                    # Feature importance plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values,
                        X_test,
                        feature_names=feature_names,
                        plot_type="bar",
                        show=False
                    )
                    plt.title(f"SHAP Feature Importance - {output_name}")
                    output_file = f"{save_dir}/feature_importance_{output_name}.png"
                    plt.savefig(output_file, bbox_inches="tight", dpi=300)
                    plt.close()
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
    

if __name__ == "__main__":
    # RF, XGBoost, LightGBM, LR, SVM, KNN, NaiveBayes

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=models_config.keys(),
        help="Choose model name"
    )

    args = parser.parse_args()
    model_name = args.model_name

    # df = pd.read_excel("data/anemia-data.xlsx")
    df= pd.read_csv("data/concat.csv")
    df = pd.get_dummies(df, columns=["Giới"], drop_first=True)
    
    print(df.head())

    #df = df.apply(pd.to_numeric, errors='coerce')

    X = df.drop(columns=["ACD", "IDA", "Thal"])
    y = df[["ACD", "IDA", "Thal"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    df_test = df.loc[X_test.index]
    df_test.to_csv("data/test_data.csv", index=False)

    factory = ModelFactory(model_name)
    results = factory.train_and_evaluate(X_train, y_train, X_test, y_test)

    print_result(results)
    factory.explain_with_shap(X_test, feature_names=X.columns.tolist())