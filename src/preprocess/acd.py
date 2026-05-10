import pandas as pd
import os

cwd= os.getcwd()
BASE_DIR= cwd


def clean_acd(df_acd):
    a= df_acd.drop(["Mã bệnh nhân"], axis=1)
    a["Tiền sử hoặc bệnh kèm theo"]= True 
    b= a.rename(columns={
    "Họ và tên": "Giới",
    "Hb (g/l)": "Hb"
    })

    original = b["Chẩn đoán"]
    b["Chẩn đoán"] = "ACD"

    b.loc[original.str.contains("thiếu máu thiếu sắt", case=False, na=False),
    "Chẩn đoán"] += ", IDA"


    b.loc[original.str.contains("beta thalassemia", case=False, na=False),
    "Chẩn đoán"] += ", Beta thalassemia"

    b.loc[original.str.contains("alpha thalassemia", case=False, na=False),
    "Chẩn đoán"] += ", Alpha thalassemia"

    return b


if __name__ == "__main__":
    data_acd= os.path.join(BASE_DIR, "data", "acd.xlsx")

    df_acd= pd.read_excel(data_acd)

    final_acd= clean_acd(df_acd)

    final_acd.to_csv(os.path.join(BASE_DIR, "data", "final_acd.csv"), index=False)
    
    print(final_acd.head())
    print(final_acd.shape)
    final_acd.info()

