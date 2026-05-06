import os
import numpy as np 

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
    if use_scaler:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(model))
        ])
    else:
        return Pipeline([
            ("clf", MultiOutputClassifier(model))
        ])

def join_text(texts):
    if texts is not None: 
        return ", ".join(texts)
    return texts


def parse_number(t):
    if t == "":
        return pd.NA
    else:
        return float(t)
    

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


"""
def cal_mentzer(mcv, rbc):
    if rbc is pd.NA or rbc == 0:
        return pd.NA
    return mcv/rbc

def stfr_ferritin_index(stfr, ferritin, thres, labels):
    diagnosis: str
    reason: str
    idx= 0

    if ferritin is not pd.NA and stfr is not pd.NA and ferritin >0 and ferritin !=1: 
        idx= stfr / math.log10(ferritin)

    if idx > thres['stfr_fer_idx'][1]:
        diagnosis= labels[1]
        reason= ""
                        
    elif idx < thres['stfr_fer_idx'][0]:
        diagnosis= labels[3]
        reason= ""


    else:
        reason= "Theo dõi thêm (thử sắt)"
        diagnosis= labels[2]
        reason= ""


    return diagnosis, reason
"""

def diendihst(data, thres, labels):
    diagnosis = ""
    reason = ""

    hb_list= ["hbbart", "hbh", "hbe", "hbs", "hb_other", "hbf"]

    if ((not (pd.isna(data.hba) and pd.isna(data.hba2))) or not pd.isna(data.hbbart) or not pd.isna(data.hbh) or not pd.isna(data.hbf)):

        if (data.hba < thres['hba'][1] and data.hba > thres['hba'][0]) and (data.hba2 < thres['hba2'][1] and data.hba2 > thres['hba2'][0]) and (all(pd.isna(getattr(data, f)) or getattr(data, f) ==0 for f in hb_list)):
            diagnosis= labels[0]

        elif (data.hbbart >= thres["hbbart"]) or (data.hbh > thres["hbh"]): 
            diagnosis = labels[2]

    
        elif (data.hba2 > thres["hba2"][1]) or (data.hbf > thres["hbf"]):
            diagnosis = labels[3]

        
        else:
            if not pd.isna(data.dotbiengen) and not pd.isna(data.man_tinh):
                if data.dotbiengen==1:
                    diagnosis= labels[4]
                elif data.dotbiengen==1 and data.man_tinh==1:
                    diagnosis= labels[1]
                else:
                    diagnosis= labels[0]
                    reason= "IDA hoặc Hội chẩn chuyên gia"

    return diagnosis, reason


def cal_tsat(fe, transferrin):
    try:
        if pd.isna(fe) or pd.isna(transferrin):
            return pd.NA
        if transferrin == 0:
            return pd.NA
        return fe*100 / (transferrin * 0.179)
    except:
        return pd.NA


def print_result(result):
    print(f"Best params: {result['best_params']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"F1-macro: {result['f1_macro']:.4f}")
    print(f"CV F1: {float(result['cv_mean_f1']):.4f} ± {float(result['cv_std_f1']):.4f}")
    print(f"Training time: {result['training_time_sec']:.2f} sec")

    print("\nClassification Report:\n")
    print(result["report"])

    print(f"Saved model    : {result['saved_model']}")



                    
                


                    

        


    
            

