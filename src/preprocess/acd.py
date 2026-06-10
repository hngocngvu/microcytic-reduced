import pandas as pd
import os

cwd= os.getcwd()
BASE_DIR= cwd


def clean_acd_1(df_acd):
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

def clean_acd_2(df_acd):
    a= df_acd.drop(["Mã điều trị"], axis=1)
    a["Tiền sử hoặc bệnh kèm theo"]= True 
    b= a.rename(columns={
    "Hb (g/l)": "Hb"
    })

    b["Giới"]= "Nam"


    
    original = b["Chẩn đoán"]
    b["Chẩn đoán"] = "ACD"

    b.loc[original.str.contains("thiếu máu thiếu sắt", case=False, na=False),
    "Chẩn đoán"] += ", IDA"


    b.loc[original.str.contains("beta thalassemia", case=False, na=False),
    "Chẩn đoán"] += ", Beta thalassemia"

    b.loc[original.str.contains("alpha thalassemia", case=False, na=False),
    "Chẩn đoán"] += ", Alpha thalassemia"

    return b

def clean_acd_3(df_acd):
    a= df_acd.drop(["Mã điều trị"], axis=1)
    a["Tiền sử hoặc bệnh kèm theo"]= True 
    b= a.rename(columns={
    "Hb (g/l)": "Hb"
    })

    b["Giới"]= "Nữ"
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
    data_acd_2= os.path.join(BASE_DIR, "data", "acd_2.xlsx")
    data_acd_3= os.path.join(BASE_DIR, "data", "acd_3.xlsx")

    df_acd_1= pd.read_excel(data_acd)
    df_acd_2= pd.read_excel(data_acd_2)
    df_acd_3= pd.read_excel(data_acd_3)

    final_acd_1= clean_acd_1(df_acd_1)
    final_acd_2= clean_acd_2(df_acd_2)
    final_acd_3= clean_acd_3(df_acd_3)

    final_acd= pd.concat([final_acd_1, final_acd_2, final_acd_3], ignore_index=True)

    final_acd.to_csv(os.path.join(BASE_DIR, "data", "final_acd.csv"), index=False)
    
    print(final_acd.head())
    print(final_acd.shape)
    final_acd.info()

