import pandas as pd
import os

from feature_selection import FeatureSelection

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join("data", "concat.csv"))

    X= df[df.columns.difference(["ACD", "IDA", "Thal"])]
    y = df[["ACD", "IDA", "Thal"]]

    fs= FeatureSelection(X, y)
    corr_filter= fs.correlation_filter()

    rf= fs.get_features_rf()
    print(rf)

    lr= fs.get_features_lr()
    print(lr)

    cb= fs.combine_features()
    print(cb)

    e= fs.evaluate_k(cb)
    print(e)

    # final_df = df.drop(columns=["NEUTp", "LYMn", "LYMp"])
    # final_df.to_csv(os.path.join(BASE_DIR, "data", "final.csv"), index=False)




