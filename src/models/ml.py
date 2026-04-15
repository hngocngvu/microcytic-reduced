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
    StratifiedKFold
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

# Fix LightGBM cho no chay nhanh hon

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
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        }
    },

    "XGBoost": {
        "model": XGBClassifier(eval_metric='mlogloss', random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    },

    "LightGBM": {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "n_estimators": [100],
            "learning_rate": [0.05],
            "max_depth": [-1, 5]
        }
    },

    "LR": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1, 10]
        }
    },

    "SVM": {
        "model": SVC(probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"]
        }
    },

    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7]
        }
    },

    "NaiveBayes": {
        "model": GaussianNB(),
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
        
        # Sử dụng StratifiedKFold để đảm bảo phân phối lớp đồng đều trong mỗi fold
        # hyper-params tuning: RandomizedSearchCV cho các model có nhiều hyper-params (RF, XGBoost, LightGBM) để tiết kiệm thời gian
        # còn lại sẽ dùng GridSearchCV hoặc fit trực tiếp nếu không có hyper-params nào cần tuning

        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        if param_grid:
            if self.model_name in ["LightGBM", "XGBoost", "RF"]:
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=5,
                    cv=cv_strategy,
                    scoring='f1_macro',
                    n_jobs=-1,
                    random_state=42,
                    return_train_score=True  # Để lấy thêm thông tin
                )
            else:
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_strategy,
                    scoring='f1_macro',
                    n_jobs=-1,
                    return_train_score=True
                )
            
            # search.fit(): CV + fit trên toàn bộ training set với best hyper-params để deploy
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            cv_mean_f1 = search.best_score_  
            cv_std_f1 = search.cv_results_['std_test_score'][search.best_index_]
            
        else:
            # k-cross validation để đánh giá
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring='f1_macro',
                n_jobs=-1
            )
            cv_mean_f1 = cv_scores.mean()
            cv_std_f1 = cv_scores.std()
            
            # fit trên toàn bộ training set để deploy
            model.fit(X_train, y_train)
            best_model = model
            best_params = "default"
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Đánh giá trên test set
        y_pred = best_model.predict(X_test)
        
        # Lưu model
        model_filename = f"{self.model_name}_best_model.pkl"
        model_filepath = f"artifacts/{model_filename}"
        joblib.dump(best_model, model_filepath)
        
        return {
            "model_name": self.model_name,
            "best_model": best_model,
            "best_params": best_params,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "cv_mean_f1": cv_mean_f1,
            "cv_std_f1": cv_std_f1,  
            "training_time_sec": training_time,
            "report": classification_report(y_test, y_pred, digits=4),
            "saved_model": model_filename
        }
    
    def explain_with_shap(self, X_test, feature_names, save_dir="notebooks/eda/shap"):
        
        if self.model_name not in ["RF", "XGBoost", "LightGBM"]:
            print("SHAP only supports tree-based models.")
            return

        os.makedirs(save_dir, exist_ok=True)

        model = joblib.load(f"artifacts/{self.model_name}_best_model.pkl")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)


        plt.figure()
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.savefig(f"{save_dir}/summary_plot.png", bbox_inches="tight", dpi=300)
        plt.close()

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

    df = pd.read_excel("data/anemia-data.xlsx")
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    
    print(df.head())

    df = df.apply(pd.to_numeric, errors='coerce')

    X = df.drop(columns=["Decision_Class"])
    y = df["Decision_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y 
    )

    factory = ModelFactory(model_name)
    results = factory.train_and_evaluate(X_train, y_train, X_test, y_test)

    print_result(results)
    factory.explain_with_shap(X_test, feature_names=X.columns.tolist())