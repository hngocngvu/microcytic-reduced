import pandas as pd
import os 

cwd= os.getcwd()
BASE_DIR= cwd

df_acd= pd.read_csv(os.path.join(BASE_DIR, "data", "final_acd.csv"))
df_thalass= pd.read_csv(os.path.join(BASE_DIR, "data", "final_thalass.csv"))
df_ida= pd.read_csv(os.path.join(BASE_DIR, "data", "final_ida.csv"))
df_mix= pd.read_csv(os.path.join(BASE_DIR, "data", "final_mix.csv"))

def convert_label(x):
    if "Thalassemia" in str(x):
        return "Thalassemia"
    
    x = str(x).lower()

    labels = []

    if "acd" in x:
        labels.append("ACD")

    if "ida" in x:
        labels.append("IDA")

    elif "beta thalassemia" in x:
        labels.append("Beta thalassemia")

    elif "alpha thalassemia" in x:
        labels.append("Alpha thalassemia")

    elif "crnn" in x:
        labels.append("CRNN")
    
    if len(labels) == 0:
        return "None"

    return ", ".join(sorted(labels))


def normalize_missing_values(df):

    # strip khoảng trắng
    df = df.map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    # replace về pd.NA
    df = df.replace(
        ["", "NaN", "nan", "None", "null", "NULL"],
        pd.NA
    )

    return df

def clean_concat(df):
    df["Chẩn đoán"] = df["Chẩn đoán"].apply(convert_label)

    df["ACD"] = df["Chẩn đoán"].str.contains("ACD", na=False).astype(int)
    df["IDA"] = df["Chẩn đoán"].str.contains("IDA", na=False).astype(int)
    df["Alpha thalassemia"] = df["Chẩn đoán"].str.contains("alpha thalassemia", case=False, na=False).astype(int)
    df["Beta thalassemia"] = df["Chẩn đoán"].str.contains("beta thalassemia", case=False, na=False).astype(int)


    for i, row in df.iterrows():

        tsat = row["TSAT (%)"]
        fe = row["Fe"]
        transferin = row["Transferin"]

        # chỉ tính khi TSAT đang thiếu
        if pd.isna(tsat):

            # phải có Fe và Transferin hợp lệ
            if pd.notna(fe) and pd.notna(transferin) and transferin != 0:

                df.loc[i, "TSAT (%)"] = (
                    fe * 100 / (transferin * 0.179)
                )

    df = df.apply(normalize_missing_values)

    df["Giới"] = df["Giới"].str.strip().str.lower()
    df = pd.get_dummies(df, columns=["Giới"], drop_first=True)

    df["TSAT (%)"] = df["TSAT (%)"].round(1)

    return df


if __name__ == "__main__":
    
    df= pd.concat([df_acd, df_thalass, df_ida], axis=0)


    df_new = clean_concat(df)

    print(df_new.info())
    df_new.to_csv(os.path.join(BASE_DIR, "data", "concat_for_eda.csv"), index=False)

    df_final= df_new.drop(columns=["Chẩn đoán"])
    df_final.to_csv(os.path.join(BASE_DIR, "data", "concat.csv"), index=False)

