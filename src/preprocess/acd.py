import pandas as pd
import os

cwd = os.getcwd()
BASE_DIR = cwd


def _label_diagnoses(df):
    original = df["Chẩn đoán"]
    df["Chẩn đoán"] = "ACD"

    df.loc[original.str.contains("thiếu máu thiếu sắt", case=False, na=False),
           "Chẩn đoán"] += ", IDA"

    df.loc[original.str.contains("beta thalassemia", case=False, na=False),
           "Chẩn đoán"] += ", Beta thalassemia"

    df.loc[original.str.contains("alpha thalassemia", case=False, na=False),
           "Chẩn đoán"] += ", Alpha thalassemia"

    return df


def clean_acd(df_acd, drop_col, rename_map, gender=None):
    a = df_acd.drop([drop_col], axis=1)
    a["Tiền sử hoặc bệnh kèm theo"] = True
    b = a.rename(columns=rename_map)

    if gender is not None:
        b["Giới"] = gender

    b = _label_diagnoses(b)
    return b


if __name__ == "__main__":
    configs = [
        {
            "path": os.path.join(BASE_DIR, "data", "acd.xlsx"),
            "drop_col": "Mã bệnh nhân",
            "rename_map": {"Họ và tên": "Giới", "Hb (g/l)": "Hb"},
            "gender": None,
        },
        {
            "path": os.path.join(BASE_DIR, "data", "acd_2.xlsx"),
            "drop_col": "Mã điều trị",
            "rename_map": {"Hb (g/l)": "Hb"},
            "gender": "Nam",
        },
        {
            "path": os.path.join(BASE_DIR, "data", "acd_3.xlsx"),
            "drop_col": "Mã điều trị",
            "rename_map": {"Hb (g/l)": "Hb"},
            "gender": "Nữ",
        },
    ]

    frames = []
    for cfg in configs:
        df = pd.read_excel(cfg["path"])
        cleaned = clean_acd(df, cfg["drop_col"], cfg["rename_map"], cfg["gender"])
        frames.append(cleaned)

    final_acd = pd.concat(frames, ignore_index=True)
    final_acd.to_csv(os.path.join(BASE_DIR, "data", "final_acd.csv"), index=False)

    print(final_acd.head())
    print(final_acd.shape)
    final_acd.info()
