import os 

# tsat%= fe*70.9/transferin

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.dataclass.schema import Output
# from src.functions.function import join_text, clean_text, stfr_ferritin_index, cal_mentzer, diendihst, cal_tsat, clean_phay, clean_cham
from src.functions.function import join_text, cal_tsat, clean_phay, clean_cham

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

        #reasons= []
        diagnoses= str
        data.tsat= pd.NA

        #data.gender= clean_text(data.gender)
        #if data.gender == "nữ" and data.pregnant is True:
            #data.gender= "pregnant"

        #FIELDS = ["hb", "mcv", "ferritin", "transferrin", "fe", "crp"]

        if not pd.isna(data.fe) and not pd.isna(data.transferrin) and data.transferrin != 0:
            data.tsat= cal_tsat(data.fe, data.transferrin)
            # mentzer= cal_mentzer(data.mcv, data.rbc)

        if data.gender is not None and data.hb < thres["hb"][data.gender]:
            if not pd.isna(data.mcv) and data.mcv < thres['mcv']:
                if not (pd.isna(data.tsat) and pd.isna(data.crp) and pd.isna(data.ferritin)):
                    if data.tsat < thres['tsat'] and data.ferritin < thres['ferritin'][0] and data.crp < thres['crp']:
                        diagnoses= labels[0]
                    elif data.tsat >= thres['tsat'] and data.ferritin >= thres['ferritin'][1] and data.crp >= thres['crp']:
                        diagnoses= labels[1]

                    elif data.tsat < thres['tsat'] and data.ferritin < thres['ferritin'][1] and data.crp >= thres['crp']:
                        diagnoses= labels[2]

        else: diagnoses= "None"

        #diagnoses= [clean_cham(d) for d in diagnoses]
        #reasons = [clean_cham(r) for r in reasons]

        #concat_diagnoses= join_text(diagnoses)
        #concat_reasons= join_text(reasons)
        
        #concat_diagnoses= clean_phay(concat_diagnoses)
        #concat_reasons= clean_phay(concat_reasons)

        
        return Output(diagnoses= diagnoses) #reasons= concat_reasons)
    
    def ml_classify(self):

        try:
            model = joblib.load(os.path.join(BASE_DIR, "artifacts", "XGBoost_best_model.pkl"))
        except Exception as e:
            print("LOAD FAIL:", e)
            return pd.NA

        # encode gender
        gender = 0 if self.data.gender == "Nữ" else 1

        data = np.array([[
            #self.data.age,
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
            #self.data.dotbiengen,
            self.data.tsat,
            #self.data.mch,
            gender
        ]])

        # predict
        probs = model.predict_proba(data)

        labels = ["ACD", "IDA"]

        result = {}

        for i, label in enumerate(labels):
            prob_1 = probs[i][0][1]  # xác suất class = 1
            result[label] = f"{float(prob_1) * 100:.2f}%"

        return Output(
            diagnoses=result
                            )






