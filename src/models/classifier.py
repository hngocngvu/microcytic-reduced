import os 

# mentzer= mcv/rbc
# tsat%= fe*100%/tibc or fe*70.9/transferin
# stfr/fer_idx= stfr/log(ferritin)

cwd= os.getcwd()
BASE_DIR= os.path.join(cwd,"..","..")
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

from src.dataclass.schema import Input, Output, Config
from src.functions.function import clean_text, stfr_ferritin_index, cal_mentzer, diendihst, cal_tsat

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

        data.gender= clean_text(data.gender)
        if data.gender == "nữ" and data.pregnant is True:
            data.gender= "pregnant"

        FIELDS = ["mcv", "hb", "crp", "ferritin", "transferrin", "stfr", "rdw", "rbc", "dotbiengen", "hbbart", "hba2", "hb_other", "hbs", "hbe"]

        if data.fe is not None and data.transferrin is not None and data.transferrin != 0:
            tsat= cal_tsat(data.fe, data.transferrin)
            mentzer= cal_mentzer(data.mcv, data.rbc)

        if any(getattr(data, f) is not None for f in FIELDS):

            d, r= diendihst(data, thres, labels)
            diagnoses.append(d)
            reasons.append(r)

            if data.dotbiengen:
                diagnoses.append(labels[0])
                reasons.append("Phát hiện đột biến gen Thalassemia")
            
    
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

                    if data.ferritin < thres["ferritin"][0] and (data.rdw < thres["rdw"] or mentzer > thres['mentzer']):
                        # trả về IDA
                        diagnoses.append(labels[1])

                    else:
                        if not data.dotbiengen:
                            reasons.append("Cần điện di huyết sắc tố để kiểm tra đột biến gen.")
                        else: 
                            d, r= diendihst(data, thres, labels)
                            diagnoses.append(d)
                            reasons.append(r)

        return Output(diagnoses= diagnoses, reasons=reasons)



