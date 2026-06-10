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
    importance_df= fs.get_features_permutation()

    results_df, best_k= fs.evaluate_k(importance_df) 
    ranked_features= fs.features.iloc[:best_k]

    print(importance_df)
    print(results_df)
    print(f"Best k: {best_k}")

    fs.plot_elbow_method(importance_df)
    fs.plot_elbow_method(importance_df)
    fs.plot_k_performance(results_df)


    # final_df = df.drop(columns=["NEUTp", "LYMn", "LYMp"])
    # final_df.to_csv(os.path.join(BASE_DIR, "data", "final.csv"), index=False)




