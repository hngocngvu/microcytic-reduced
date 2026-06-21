import pandas as pd
import os
import sys

cwd= os.getcwd()
BASE_DIR= cwd
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.functions.function import encode_dotbiengen

data_ida= os.path.join(BASE_DIR, "data", "ida.xlsx")


def clean_ida(df, text_cols):
    a= df.drop(index= range(3))
    b= a.reset_index(drop= True)
    b.columns= b.iloc[0]
    c= b.drop(index= 0)
    d = c.iloc[:, 2:-4] # bỏ cột tên NaN
    d["Giới"] = d["Giới"].str.capitalize()


    for i in range(len(d.columns)):
        if i not in text_cols:
            col_name = d.columns[i]
            print(col_name)
            # Gán lại bằng TÊN cột, không dùng iloc
            d[col_name] = pd.to_numeric(d[col_name], errors="coerce")
        else:
            col_name = d.columns[i]
            d[col_name] = d[col_name].astype("string")

    d= d.rename(columns={
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
    "Chẩn đoán (chỉ liên quan hồng cầu nhỏ: IDA, ACD, Thalassemia, CRNN)": "Chẩn đoán",
    "XHTH, giun sán, giảm cung cấp ( ăn kiêng) , có thai ": "nguyên nhân thiếu sắt"
    })



    #d = encode_dotbiengen(d)

    d["Chẩn đoán"]= "IDA"


    print(d.columns.tolist())

    original_ida= d["nguyên nhân thiếu sắt"]

    d.loc[original_ida.str.contains(r"viêm khớp|ung thư|mạn tính|gout|suy thận|áp xe gan", case=False, na=False),
    "Chẩn đoán"] += "/ACD"


    d["tiensu_ida"]= d["nguyên nhân thiếu sắt"].str.contains(r"ung thư|rong kinh|rong huyết|thiếu sắt|loét dạ dày|kí sinh trùng|trĩ chảy máu", case=False, na=False)
    d["tiensu_acd"]= d["nguyên nhân thiếu sắt"].str.contains(r"viêm khớp|gout|suy thận|áp xe gan|ung thư|mạn tính", case=False, na=False)


    d = d.drop(columns=["nguyên nhân thiếu sắt", "Tiền sử hoặc bệnh kèm theo"])


    return d



if __name__ == "__main__":
    text_cols = [3, 24, 25, 26]

    data_ida= os.path.join(BASE_DIR, "data", "ida.xlsx")

    sheets= pd.read_excel(data_ida, sheet_name= None)


    cleaned_sheets = [
    clean_ida(df, text_cols)
    for df in sheets.values()]

    final_ida = pd.concat(cleaned_sheets, ignore_index=True)

    final_ida = final_ida.drop(final_ida.columns[25], axis=1)
    final_ida.to_csv(
        os.path.join(BASE_DIR, "data", "final_ida.csv"),
        index=False
    )

    print(final_ida.head())
    print(final_ida.shape)
    final_ida.info()