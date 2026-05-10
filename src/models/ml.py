import os
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.functions.function import print_result
from src.dataclass.modelfactory import ModelFactory, models_config
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import argparse

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
    
    print(df.head())

    #df = df.apply(pd.to_numeric, errors='coerce')

    X = df.drop(columns=["ACD", "IDA", "Thal"])
    y = df[["ACD", "IDA", "Thal"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    df_test = df.loc[X_test.index]
    df_test.to_csv("data/test_data.csv", index=False)

    factory = ModelFactory(model_name)
    results = factory.train_and_evaluate(X_train, y_train, X_test, y_test)

    print_result(results)
    factory.explain_with_shap(X_test, feature_names=X.columns.tolist(), trained_model= results["best_model"])