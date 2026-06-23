import numpy as np 
import streamlit as st 
"""
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd


def make_pipeline(model, use_scaler= False):
    return Pipeline([
            ("clf", MultiOutputClassifier(model))
        ])
    

def join_text(texts):
    if texts is not None: 
        return ", ".join(texts)
    return texts



def input_field(label, key):
    val = st.text_input(label, value=str(st.session_state.patient.get(key, "")))
    try:
        val = float(val)
    except:
        val = pd.NA
    st.session_state.patient[key] = val
    return val


def clean_text(t):
    if t is not None:
        return t.lower()
    return t

def clean_cham(text):
    if text is not None:
        return text.rstrip('.').strip()
    return ""

def clean_phay(text):
    if text is not None:
        return text.lstrip(',').strip()
    return ""

def remove_duplicates_comma(text):
    items = text.split(", ")
    unique_items = list(dict.fromkeys(items))
    return ", ".join(unique_items)



# def cal_mentzer(mcv, rbc):
#     if rbc is np.nan or rbc == 0:
#         return np.nan
#     return mcv/rbc

# def stfr_ferritin_index(stfr, ferritin, thres, labels):
#     diagnosis: str
#     reason: str
#     idx= 0

#     if ferritin is not np.nan and stfr is not np.nan and ferritin >0 and ferritin !=1: 
#         idx= stfr / math.log10(ferritin)

#     if idx > thres['stfr_fer_idx'][1]:
#         diagnosis= labels[1]
#         reason= ""
                        
#     elif idx < thres['stfr_fer_idx'][0]:
#         diagnosis= labels[3]
#         reason= ""


#     else:
#         reason= "Theo dõi thêm (thử sắt)"
#         diagnosis= labels[2]
#         reason= ""


#     return diagnosis, reason


# def diendihst(data, thres, labels):
#     diagnosis = ""
#     reason = ""
    

#     hb_list= ["hbbart", "hbh", "hbe", "hbs", "hb_other", "hbf"]
#     alpha = ["sea dị hợp tử", "cd142", "3.7", "4.2"]
#     beta = ["cd41,42", "cd26", "ivs1.1.", "cd17"]



#     if ((not (pd.isna(data.hba) and pd.isna(data.hba2))) or not pd.isna(data.hbbart) or not pd.isna(data.hbh) or not pd.isna(data.hbf)):

        
#         if pd.notna(data.dotbiengen) and any(x in data.dotbiengen.lower() for x in alpha):
#             diagnosis = labels[2]

#         elif pd.notna(data.dotbiengen) and any(x in data.dotbiengen.lower() for x in beta):
#             diagnosis = labels[3]


#         elif ((data.hba < thres['hba'][1] and data.hba > thres['hba'][0]) and (data.hba2 < thres['hba2'][1] and data.hba2 > thres['hba2'][0]) and all(pd.isna(getattr(data, f)) or getattr(data, f) == 0 for f in hb_list)):
#                 diagnosis= labels[0]

#         elif (data.hbbart >= thres["hbbart"]) or (data.hbh > thres["hbh"]): 
#                 diagnosis = labels[2]
        
#         elif (data.hba2 > thres["hba2"][1]) or (data.hbf > thres["hbf"]):
#                 diagnosis = labels[3]


#         else:
#             if not pd.isna(data.dotbiengen) or not pd.isna(data.man_tinh):
#                 if not pd.isna(data.dotbiengen) and data.man_tinh:
#                     diagnosis= labels[1]
#                 else:
#                     diagnosis= labels[0]
#                     reason= "IDA hoặc Hội chẩn chuyên gia"

#     return diagnosis, reason


def cal_tsat(fe, transferrin):
    try:
        if pd.isna(fe) or pd.isna(transferrin):
            return pd.NA
        if transferrin == 0:
            return pd.NA
        return fe*100 / (transferrin * 0.179)
    except:
        return pd.NA


ALPHA_PATTERN = r"sea|3\.7|4\.2|cd142"
BETA_PATTERN = r"cd41[,;/ ]*42|cd17|cd26"


def encode_dotbiengen(df, col="Đột biến gen thalassemia"):
    df["alpha_gen"] = df[col].str.contains(ALPHA_PATTERN, regex=True, na=False).astype(int)
    df["beta_gen"] = df[col].str.contains(BETA_PATTERN, regex=True, na=False).astype(int)

    return df.drop(columns=[col])


def parse_dotbiengen(text):
    import re
    if not text or pd.isna(text):
        return pd.NA, pd.NA
    text_lower = str(text).lower()
    gen_alpha = bool(re.search(ALPHA_PATTERN, text_lower))
    gen_beta = bool(re.search(BETA_PATTERN, text_lower))
    return gen_alpha, gen_beta


def print_result(result):
    print(f"Best params: {result['best_params']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"F1-macro: {result['f1_macro']:.4f}")
    print(f"F1-micro: {result['f1_micro']:.4f}")
    print(f"Hamming Loss: {result['hamming_loss']:.4f}")
    print(f"Jaccard Score: {result['jaccard_score']:.4f}")

    print(f"CV F1: {float(result['cv_mean_f1']):.4f} ± {float(result['cv_std_f1']):.4f}")
    print(f"Training time: {result['training_time_sec']:.2f} sec")

    print("\nClassification Report:\n")
    print(result["report"])

    print(f"Saved model    : {result['saved_model']}")


def friedman_compare(results_dict):
    from scipy.stats import friedmanchisquare
    import scikit_posthocs as sp

    names = list(results_dict.keys())
    scores = list(results_dict.values())

    min_len = min(len(s) for s in scores)
    scores = [s[:min_len] for s in scores]

    stat, p = friedmanchisquare(*scores)

    print(f"Friedman chi2={stat:.4f}, p={p:.6f}")

    if p < 0.05:
        print("Significant difference found → Nemenyi post-hoc:\n")
        df_scores = pd.DataFrame(dict(zip(names, scores)))
        nemenyi = sp.posthoc_nemenyi_friedman(df_scores)
        print(nemenyi.round(4))

        sig_pairs = []
        for i in range(len(nemenyi.index)):
            for j in range(i + 1, len(nemenyi.columns)):
                g1, g2 = nemenyi.index[i], nemenyi.columns[j]
                pv = nemenyi.iloc[i, j]
                if pv < 0.05:
                    star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*"
                    sig_pairs.append(f"  {g1} vs {g2}: p={pv:.6f} {star}")

        if sig_pairs:
            print("\nSignificant pairs:")
            for pair in sig_pairs:
                print(pair)
    else:
        print("No significant difference between models.")

    return stat, p



                    
                


                    

        


    
            

