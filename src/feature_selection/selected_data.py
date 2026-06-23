import pandas as pd
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.feature_selection.feature_selection import FeatureSelection

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join("data", "concat.csv"))

    X= df[df.columns.difference(["ACD", "IDA"])]
    y = df[["ACD", "IDA"]]


    rf= XGBClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    fs= FeatureSelection(X, y, model= rf)

    corr_filter= fs.correlation_filter()

    print("PERMUTATION IMPORTANCE")


    perm_df= fs.get_features_permutation()
    print(perm_df)

    perm_results, perm_best_k= fs.evaluate_k(perm_df)
    print(perm_results)
    print(f"Best k (Permutation): {perm_best_k}")

    fs.plot_permutation_importance(perm_df)
    fs.plot_elbow_method(perm_df)
    fs.plot_k_performance(perm_results)

    print(f"Permutation Importance -> best k = {perm_best_k}, top features: {perm_df['feature'].head(perm_best_k).tolist()}")

    df_reduced= pd.concat([df[perm_df['feature'].head(perm_best_k).tolist()[:perm_best_k]], y], axis=1)
    df_reduced.to_csv(os.path.join("data", "reduced_features.csv"), index=False)


    

