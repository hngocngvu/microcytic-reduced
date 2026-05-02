import os

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
    return ", ".join(texts)


def parse_number(t):
    t = str(t).strip()

    if t == "":
        return None
    else:
        return float(t)
    

def clean_text(t):
    return t.lower()

def clean_cham(text):
    return text.rstrip('.').strip()

def clean_phay(text):
    return text.lstrip(',').strip()

"""
def cal_mentzer(mcv, rbc):
    if rbc is None or rbc == 0:
        return None
    return mcv/rbc

def stfr_ferritin_index(stfr, ferritin, thres, labels):
    diagnosis: str
    reason: str
    idx= 0

    if ferritin is not None and stfr is not None and ferritin >0 and ferritin !=1: 
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

    if data.hbbart is not None and data.hbbart > thres["hbbart"]: 
        diagnosis = labels[3]
        reason= ""

    if data.hbh is not None and data.hbh > thres["hbh"]:
        diagnosis = labels[3]
        reason= ""


    if data.hba2 is not None and data.hba2 > thres["hba2"]:
        diagnosis = labels[4]
        reason= ""


    if data.hbf is not None and data.hbf > thres["hbf"]:
        diagnosis = labels[4]
        reason= ""


    #else: 
        #reason = "Hội chẩn chuyên gia kèm tiền sử gia đình."

    return diagnosis, reason


def cal_tsat(fe, transferrin):
    try:
        if fe is None or transferrin is None:
            return None
        if transferrin == 0:
            return None
        return fe * 70.9 / transferrin
    except:
        return None


def print_result(result):
    print(f"Best params: {result['best_params']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"F1-macro: {result['f1_macro']:.4f}")
    print(f"CV F1: {float(result['cv_mean_f1']):.4f} ± {float(result['cv_std_f1']):.4f}")
    print(f"Training time: {result['training_time_sec']:.2f} sec")

    print("\nClassification Report:\n")
    print(result["report"])

    print(f"Saved model    : {result['saved_model']}")



                    
                


                    

        


    
            

