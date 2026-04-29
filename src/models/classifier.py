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
        tsat= 0

        #data.gender= clean_text(data.gender)
        #if data.gender == "nữ" and data.pregnant is True:
            #data.gender= "pregnant"

        FIELDS = ["mcv", "crp", "ferritin", "transferrin", "stfr", "rdw", "rbc", "dotbiengen", "hbbart", "hba2", "hb_other", "hbs", "hbe"]

        if data.fe is not None and data.transferrin is not None and data.transferrin != 0:
            tsat= cal_tsat(data.fe, data.transferrin)
            # mentzer= cal_mentzer(data.mcv, data.rbc)

        if any(getattr(data, f) is not None for f in FIELDS):

            """
            d, r= diendihst(data, thres, labels)
            diagnoses.append(d)
            reasons.append(r)

            if data.dotbiengen:
                diagnoses.append("")
                reasons.append("")
            
    
            if data.mcv is not None and data.mcv < 80: 
                reasons.append("Thể tích trung bình hồng cầu nhỏ hơn bình thường.")


            if data.hb < thres["hb"][data.gender]:
                if data.ferritin is None: pass

                elif data.ferritin < thres["ferritin"][0]:
                    # trả về IDA
                    diagnoses.append(labels[1])
                
                else: 
                    if data.crp is None: pass
                    elif data.crp > thres['crp']:
                        if data.ferritin is None: pass

                        elif data.ferritin < thres["ferritin"][1]:
                            if tsat is not None or data.stfr is not None or data.ferritin is not None:
                                if tsat is not None:
                                    if tsat < thres['tsat'][0]:
                                        # xét tiền sử, scoring dùng cho đoạn này, chưa biết vì sơ đồ không đề cập
                                        reasons.append("Cần dựa vào tiền sử để đưa ra kết luận chính xác.")
                                        diagnoses.append(labels[4])
                                        
                                    elif tsat > thres["tsat"][1]:
                                        # trả về ACD
                                        diagnoses.append(labels[3])
                                        reasons.append(r)
                                    else: 
                                        d,r= stfr_ferritin_index(data.stfr, data.ferritin, thres, labels)
                                        diagnoses.append(d)
                                        reasons.append(r)
                        else: 
                            d, r= stfr_ferritin_index(data.stfr, data.ferritin, thres, labels)
                            diagnoses.append(d)
                            reasons.append(r)



                    else:
                        if tsat is None: pass

                        elif tsat < thres['tsat'][0]:
                            # trả về IDA
                            diagnoses.append(labels[1])

                        elif tsat > thres['tsat'][1]:
                            if not data.dotbiengen:
                                reasons.append("Cần điện di huyết sắc tố để kiểm tra đột biến gen.")
                            else: 
                                d, r= diendihst(data, thres, labels)
                                diagnoses.append(d)
                                reasons.append(r)

                        else: 
                            reasons.append("Gợi ý IDA + tiền sử")
                            diagnoses.append(labels[1])


            else: 
                if data.rdw is not None or data.mcv is not None or data.rbc is not None or data.ferritin is not None:


                    if data.ferritin < thres["ferritin"][0] and (mentzer is not None and (data.rdw < thres["rdw"] or mentzer > thres['mentzer'])):
                        # trả về IDA
                        diagnoses.append(labels[1])

                    else:
                        if not data.dotbiengen:
                            reasons.append("Cần điện di huyết sắc tố để kiểm tra đột biến gen.")
                        else: 
                            d, r= diendihst(data, thres, labels)
                            diagnoses.append(d)
                            reasons.append(r)

            """

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
            return None

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






