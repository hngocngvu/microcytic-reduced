import pandas as pd
import os

cwd= os.getcwd()
BASE_DIR= cwd

data_thalass= os.path.join(BASE_DIR, "data", "thalass.xlsx")


def clean_thalass(df_thalass, cols):
    df_thalass.columns = df_thalass.iloc[1]
    a = df_thalass.drop([0, 1])
    b = a.iloc[:, 1:-1]
    c = b.drop(["PID"], axis=1)
    d = c.reset_index(drop=True)
    d= d.rename(columns={
    "RBC (T/l)": "RBC",
    "Hb (g/L)": "Hb",
    "Transferin (mg/dL)": "Transferin",
    "Đột biến gen thalassemia (nếu có)": "Đột biến gen thalassemia"
    })

    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")


    d["Tiền sử hoặc bệnh kèm theo"] = (
        d["Tiền sử hoặc bệnh kèm theo"].notna() &
        d["Tiền sử hoặc bệnh kèm theo"].astype(str).str.strip().ne("")
    )

    alpha_pattern = r"sea|3\.7|4\.2|cd142"
    beta_pattern = r"cd41[,;/ ]*42|cd17|cd26"

    d["alpha_gen"] = (
        d["Đột biến gen thalassemia"].str.contains(alpha_pattern, regex=True, na=False)
    ).astype(int)

    d["beta_gen"] = (
        d["Đột biến gen thalassemia"].str.contains(beta_pattern, regex=True, na=False)
    ).astype(int)

    d = d.drop(columns=["Đột biến gen thalassemia"])


    df_part = d.head(28)

    return df_part



if __name__ == "__main__":
    cols = ["RBC", "Hb", "MCV", "MCHC", "RDW-CV", "Fe", "Ferritin", "Transferin", "TIBC", "CRP", 
        "Ret-He", "HbA1", "HbA2", "HbF", "HbH", "HbBart", "HbS", "HbE", "Hb khác", "Tuổi"]
    

    data_thalass= os.path.join(BASE_DIR, "data", "thalass.xlsx")

    df_thalass= pd.read_excel(data_thalass)

    final_thalass= clean_thalass(df_thalass, cols)

    final_thalass.to_csv(os.path.join(BASE_DIR, "data", "final_thalass.csv"), index=False)

    print(final_thalass.head())
    print(final_thalass.shape)
    final_thalass.info()

