import pandas as pd
import numpy as np

NHANES_TO_CONCAT = {
    "hemoglobin": "Hb",
    "rbc": "RBC",
    "mcv": "MCV",
    "mchc": "MCHC",
    "rdw": "RDW-CV",
    "ferritin": "Ferritin",
    "serum_iron": "Fe",
    "tibc": "TIBC",
    "tsat": "TSAT (%)",
    "hscrp": "CRP",
}


def load_nhanes_for_validation(nhanes_path, concat_feature_columns):
    df = pd.read_csv(nhanes_path)

    label = df["anemia_class"]
    y = pd.DataFrame({
        "ACD": label.str.contains("ACD", na=False).astype(int),
        "IDA": label.str.contains("IDA", na=False).astype(int),
    })

    df = df.rename(columns=NHANES_TO_CONCAT)
    df["Hb"] = df["Hb"] * 10
    df["Giới_nữ"] = (df["gender"].str.lower() == "female")

    X = pd.DataFrame(index=df.index)
    for col in concat_feature_columns:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = np.nan

    return X, y
