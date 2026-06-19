import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


if __name__ == "__main__":
    df= pd.read_csv("data/reduced_features.csv")
    print(df.head())

    X = df.drop(columns=["ACD", "IDA"])
    y = df[["ACD", "IDA"]]


    msss= MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(msss.split(X, y))


    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    df_train= df.loc[train_idx]
    df_train.to_csv("data/train_set_reduced_features.csv", index=False)

    #required_features = ["MCV", "Hb", "TSAT (%)", "Ferritin", "CRP"]

    df_test = df.loc[test_idx]

    # Remove records with missing values in any required feature
    #df_test_complete = df_test.dropna(subset=required_features)

    #print(f"Original test set size: {len(df_test)}")
    #print(f"Complete-case test set size: {len(df_test_complete)}")
    #print(f"Excluded records: {len(df_test) - len(df_test_complete)}")

    df_test.to_csv(
        "data/test_set_reduced_features.csv",
        index=False
    )
    
