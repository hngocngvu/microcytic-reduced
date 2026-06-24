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
    # XGBoost, CatBoost, LightGBM 
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

    df_train = pd.read_csv("data/train_set.csv")
    df_test = pd.read_csv("data/test_set.csv")

    print(df_train.head())

    #df = df.apply(pd.to_numeric, errors='coerce')

    X_train = df_train.drop(columns=["ACD", "IDA"])
    y_train = df_train[["ACD", "IDA"]]

    X_test = df_test.drop(columns=["ACD", "IDA"])
    y_test = df_test[["ACD", "IDA"]]

    factory = ModelFactory(model_name)
    results = factory.train_and_evaluate(X_train, y_train, X_test, y_test)

    print_result(results)
    factory.explain_with_shap(X_test, feature_names=X_train.columns.tolist(), trained_model= results["best_model"])