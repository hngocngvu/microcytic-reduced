import os
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.functions.function import print_result
from src.dataclass.modelfactory import ModelFactory, models_config

import pandas as pd 
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

if __name__ == "__main__":
    # RF, XGBoost, LightGBM, LR, SVM, KNN, NaiveBayes, DT

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
    
    print(df.head())

    #df = df.apply(pd.to_numeric, errors='coerce')

    X = df.drop(columns=["ACD", "IDA", "Alpha thalassemia", "Beta thalassemia"])
    y = df[["ACD", "IDA", "Alpha thalassemia" ,"Beta thalassemia"]]

    msss= MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(msss.split(X, y))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    

    df_test = df.loc[test_idx]
    df_test.to_csv("data/test_data.csv", index=False)

    factory = ModelFactory(model_name)
    results = factory.train_and_evaluate(X_train, y_train, X_test, y_test)

    print_result(results)
    factory.explain_with_shap(X_test, feature_names=X.columns.tolist(), trained_model= results["best_model"])