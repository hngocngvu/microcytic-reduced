import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


if __name__ == "__main__":
    df= pd.read_csv("data/concat.csv")
    print(df.head())

    X = df.drop(columns=["ACD", "IDA"])
    y = df[["ACD", "IDA"]]


    msss= MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(msss.split(X, y))


    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    df_train= df.loc[train_idx]
    df_train.to_csv("data/train_set.csv", index=False)

    
    df_test = df.loc[test_idx]
    df_test.to_csv("data/test_set.csv", index=False)
