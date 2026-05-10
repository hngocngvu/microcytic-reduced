import os 

# tsat%= fe*70.9/transferin

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.dataclass.schema import Output
# from src.functions.function import join_text, clean_text, stfr_ferritin_index, cal_mentzer, diendihst, cal_tsat, clean_phay, clean_cham
from src.functions.function import join_text, cal_tsat, clean_phay, clean_cham, diendihst

import joblib
import pandas as pd
import numpy as np 

class Classifier():
    def __init__(self, data, config):
        self.data= data
        self.config= config

    def classify(self) -> Output:
        data= self.data 
        thres= self.config.thres
        labels= self.config.labels

        reasons= []
        diagnoses= []
        tsat= pd.NA

        #data.gender= clean_text(data.gender)
        #if data.gender == "nữ" and data.pregnant is True:
            #data.gender= "pregnant"

        FIELDS = ["mcv", "crp", "ferritin", "transferrin", "dotbiengen", "hbbart", "hba2", "hbh", "hbf", "fe"]

        if not pd.isna(data.fe) and not pd.isna(data.transferrin) and data.transferrin != 0:
            tsat= cal_tsat(data.fe, data.transferrin)
            # mentzer= cal_mentzer(data.mcv, data.rbc)

        if any(not pd.isna(getattr(data, f)) for f in FIELDS):
            if data.mcv < thres['mcv']:
                if data.ferritin < thres['ferritin'][0]:
                    #d1= labels[0]
                    #diagnoses.append(d1)
                    d2, r1= diendihst(data, thres, labels)
                    diagnoses.append(d2)
                    reasons.append(r1)

                elif data.ferritin > thres['ferritin'][1]:
                    d3= ""
                    if (data.crp > thres['crp'] and data.fe < thres['fe']) and (not pd.isna(tsat) and tsat < thres['tsat']):
                        d3= labels[1]
                    else:
                        d3, r2= diendihst(data, thres, labels)
                        reasons.append(r2)
        
                    diagnoses.append(d3)
                
                else:
                    d4= ""
                    if not pd.isna(tsat) and tsat < thres['tsat']: 
                        d4= labels[0]
                    else:
                        d4, r3= diendihst(data, thres, labels)
                        reasons.append(r3)
                    diagnoses.append(d4)

        diagnoses= [clean_cham(d) for d in diagnoses]
        reasons = [clean_cham(r) for r in reasons]

        concat_diagnoses= join_text(diagnoses)
        concat_reasons= join_text(reasons)
        
        concat_diagnoses= clean_phay(concat_diagnoses)
        concat_reasons= clean_phay(concat_reasons)

        
        return Output(diagnoses= concat_diagnoses, reasons= concat_reasons)
    
    def ml_classify(self):

        try:
            model = joblib.load(os.path.join(BASE_DIR, "artifacts", "XGBoost_best_model.pkl"))
        except Exception as e:
            print("LOAD FAIL:", e)
            return pd.NA

        # encode gender
        gender = 0 if self.data.gender == "Nữ" else 1

        data = np.array([[
            self.data.age,
            self.data.man_tinh,
            self.data.rbc,
            self.data.hb,
            self.data.mcv,
            self.data.mchc,
            self.data.rdw,
            self.data.fe,
            self.data.ferritin,
            self.data.transferrin,
            self.data.tibc,
            self.data.crp,
            self.data.ret_he,
            self.data.hba,
            self.data.hba2,
            self.data.hbf,
            self.data.hbh,
            self.data.hbbart,
            self.data.hbs,
            self.data.hbe,
            self.data.hb_other,
            self.data.dotbiengen,
            self.data.tsat,
            self.data.mch,
            gender
        ]])

        # predict
        probs = model.predict_proba(data)

        labels = ["ACD", "IDA", "Thal"]

        result = {}

        for i, label in enumerate(labels):
            prob_1 = probs[i][0][1]  # xác suất class = 1
            result[label] = f"{float(prob_1) * 100:.2f}%"

        return Output(
            diagnoses=result,
            reasons="Dựa trên mô hình XGBoost multi-label."
        )






