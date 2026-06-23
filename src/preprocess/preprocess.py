import pandas as pd
import os 

cwd= os.getcwd()
BASE_DIR= cwd


def clean_concat(df):
    df["ACD"] = df["anemia_class"].str.contains("ACD", na=False).astype(int)
    df["IDA"] = df["anemia_class"].str.contains("IDA", na=False).astype(int)

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    df = df.drop_duplicates()

    drop_cols = [c for c in ["SEQN", "age", "cycle"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["stfr_index"] = df["stfr_index"].round(1)

    return df


if __name__ == "__main__":
    
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    df= pd.read_csv(os.path.join(BASE_DIR, "nhanes_data", "data.csv"))

    df = clean_concat(df)

    df.to_csv(os.path.join(BASE_DIR, "data", "concat_for_eda.csv"), index=False)

    if "gender" in df.columns:
        df = pd.get_dummies(df, columns=["gender"], drop_first=True, dtype=int)

    df_final = df.drop(columns=["anemia_class"])
    df_final.to_csv(os.path.join(BASE_DIR, "data", "concat.csv"), index=False)
