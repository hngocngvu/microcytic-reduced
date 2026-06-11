import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from feature_selection import FeatureSelection

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join("data", "concat.csv"))

    X= df[df.columns.difference(["ACD", "IDA", "Alpha thalassemia" , "Beta thalassemia"])]
    y = df[["ACD", "IDA", "Alpha thalassemia" ,"Beta thalassemia"]]


    rf= RandomForestClassifier(
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

    print("MUTUAL INFORMATION")

    mi_df= fs.get_features_mutual_info()
    print(mi_df)

    mi_results, mi_best_k= fs.evaluate_k(mi_df)
    print(mi_results)
    print(f"Best k (Mutual Info): {mi_best_k}")

    fs.plot_permutation_importance(mi_df)
    fs.plot_elbow_method(mi_df)
    fs.plot_k_performance(mi_results)

    print(f"Permutation Importance -> best k = {perm_best_k}, top features: {perm_df['feature'].head(perm_best_k).tolist()}")
    print(f"Mutual Information     -> best k = {mi_best_k}, top features: {mi_df['feature'].head(mi_best_k).tolist()}")
