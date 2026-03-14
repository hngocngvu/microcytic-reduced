import os
from typing import Dict, List, Optional, Tuple

cwd= os.getcwd()
BASE_DIR= os.path.join(cwd,"..","..")
if BASE_DIR not in os.sys.path:
    os.sys.path.append(BASE_DIR)

import math

def clean_text(t):
    return t.lower()

def cal_mentzer(mcv, rbc):
    return mcv/rbc

def stfr_ferritin_index(stfr, ferritin, thres, labels):
    diagnosis: str
    reason: str
    idx= 0

    if ferritin !=0 and stfr is not None and ferritin >0 and ferritin !=1: 
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

def cal_mentzer(mcv, rbc):
    return mcv/rbc


def diendihst(data, thres, labels):
    diagnosis = ""
    reason = ""

    if data.hbbart !=0 and data.hbbart > thres["hbbart"]: 
        diagnosis = labels[5]
        reason= ""


    if data.hba2 !=0 and data.hba2 > thres["hba2"]:
        diagnosis = labels[6]
        reason= ""


    if (data.hb_other !=0 and data.hb_other > thres["hb_other"]) or \
        (data.hbs !=0 and data.hbs > thres["hbs"]) or \
        (data.hbe !=0 and data.hbe > thres["hbe"]):
        diagnosis = labels[0]
        reason= ""


    else: 
        reason = "Hội chẩn chuyên gia kèm tiền sử gia đình."

    return diagnosis, reason


def cal_tsat(fe, transferrin):
    # or fe*100/tibc
    return fe*70.9/transferrin




                    
                


                    

        


    
            

