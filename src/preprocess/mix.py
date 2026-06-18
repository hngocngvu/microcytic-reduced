import pandas as pd
import os

cwd= os.getcwd()
BASE_DIR= cwd

mix= os.path.join(BASE_DIR, "data", "data.xlsx")

def clean_df(df, text_cols):
    a= df.drop(index= range(3))
    b= a.reset_index(drop= True)
    b.columns= b.iloc[0]
    c= b.drop(index= 0)
    d= c.drop(columns= "STT")


    e = d.iloc[:, :-4] # bỏ cột tên NaN

    

    for i in range(len(e.columns)):
        if i not in text_cols:
            col_name= e.columns[i]
            e[col_name] = pd.to_numeric(e[col_name], errors="coerce")

        elif i==22:
            col_name = e.columns[i]
            e[col_name] = e[col_name].astype("bool")

        else:
            col_name= e.columns[i]
            e[col_name] = e[col_name].astype("string")

    

    e= e.rename(columns={
    "RBC (T/l)\nSố lượng hồng cầu": "RBC",
    "Hb (g/L)\nNồng độ Hemoglobin": "Hb",
    "MCV\nKích thước Hồng cầu": "MCV",
    "MCHC\nNồng độ hemoglogin trung bình hồng cầu": "MCHC",
    "RDW-CV\nĐộ phân bố kích thước hồng cầu": "RDW-CV",
    "Fe \nSắt huyết thanh": "Fe",
    "Ferritin\nDự trữ sắt": "Ferritin",
    "Transferin (mg/dL)\nProtein vận chuyển sắt": "Transferrin",
    "TIBC\nkhả năng gắn sắt toàn bộ": "TIBC",
    "CRP\nChỉ số viêm": "CRP",
    "Ret-He\nChỉ số hemoglobin của hồng cầu lưới": "Ret-He",
    "HbA1\nhemboglobin A1": "HbA1",
    "HbA2\nHemoglobin A2": "HbA2",
    "HbF\nHemoglobin F": "HbF",
    "HbH\nHemoblobin H": "HbH",
    "HbBart\nHemoglobin Bart": "HbBart",
    "HbS\nHemoglobin S": "HbS",
    "HbE\nhemoglobin E": "HbE",
    "Đột biến gen thalassemia (nếu có)": "Đột biến gen thalassemia",
    "Chẩn đoán (chỉ liên quan hồng cầu nhỏ: IDA, ACD, Thalassemia, CRNN)": "Chẩn đoán"
    })
    
    e= e[~e["Chẩn đoán"]
    .fillna("")
    .str.contains(r"Thalassemia|CRNN", case=False, na=False)]


    e.drop(
    columns=["Đột biến gen thalassemia"],
    errors="ignore",
    inplace=True)

    e["tiensu_ida"]= e["Tiền sử hoặc bệnh kèm theo"].str.contains(r"ung thư|rong kinh|rong huyết|thiếu sắt|loét dạ dày|kí sinh trùng|trĩ chảy máu", case=False, na=False)
    e["tiensu_acd"]= e["Tiền sử hoặc bệnh kèm theo"].str.contains(r"viêm khớp|gout|suy thận|áp xe gan|ung thư|mạn tính", case=False, na=False)

    e= e.drop(columns=["Tiền sử hoặc bệnh kèm theo"], errors="ignore")

    return e


if __name__ == "__main__":

    mix= os.path.join(BASE_DIR, "data", "data.xlsx")

    df_mix= pd.read_excel(mix)

    text_cols= [1, 2, 22,23]

    cleaned_mix_df= clean_df(df_mix, text_cols)

    cleaned_mix_df.to_csv(os.path.join(BASE_DIR, "data", "final_mix.csv"), index=False)


    print(cleaned_mix_df.head())
    print(cleaned_mix_df.shape)
    cleaned_mix_df.info()