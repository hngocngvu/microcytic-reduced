from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import date
import pandas as pd

@dataclass
class Input():
    gender: Optional[str]= None
    kinh_nguyet: Optional[bool]= False
    da_day:  Optional[bool]= False
    tri:  Optional[bool]= False
    pregnant:  Optional[bool]= False
    diet:  Optional[bool]= False
    age: Optional[int]= None

    man_tinh:  Optional[bool]= False
    cancer:  Optional[bool]= False
    phau_thuat:  Optional[bool]= False

    rbc: Optional[float]= None
    hb: Optional[float]=  None
    mcv: Optional[float]=  None 
    mchc: Optional[float]=  None 
    rdw: Optional[float]=  None
    ret_he: Optional[float]=  None


    fe: Optional[float]=  None 
    ferritin: Optional[float]=  None 
    transferrin: Optional[float]=  None 
    tibc: Optional[float]=  None 
    # stfr: Optional[float]=  None 
    tsat: Optional[float]=  None
    crp: Optional[float]=  None 

    #dotbiengen: Optional[str]= None
    #gen_alpha: Optional[bool]= False
    #gen_beta: Optional[bool]= False
    hba: Optional[float]=  None 
    hba2: Optional[float]=  None 
    hbf: Optional[float]=  None 
    hbh: Optional[float]=  None 
    hbe: Optional[float]=  None 
    # hbc: Optional[float]=  None 
    hbs: Optional[float]=  None
    hbbart: Optional[float]=  None 
    hb_other: Optional[float]=  None 


@dataclass
class Output():
    diagnoses: Optional[str]
    #reasons: Optional[str]
    # confidence: Optional[float]
    # scores: Optional[Dict[str, float]]

@dataclass
class Config():
    # weights: dict= field(default_factory= lambda: {})
    thres: dict= field(default_factory= lambda: {
        'hb': {
                "Nữ": 120, 
                "Nam": 130},

        'ferritin': [30, 100],

        'tsat': 16,
        'crp': 10,
        'mcv': 80
        # 'stfr_fer_idx' : [1, 2]
    })

    labels= [
            "IDA",
            "ACD",
            "IDA, ACD"
            ]

"""
@dataclass
class Patient:
    id: str
    full_name: str
    dob: str
    gender: str
    phone_number: str
    address: str


@dataclass
class DiagnosisRecord:
    id: str
    patient_id: str
    kinh_nguyet: Optional[bool]= False
    da_day:  Optional[bool]= False
    tri:  Optional[bool]= False
    pregnant:  Optional[bool]= False
    diet:  Optional[bool]= False

    man_tinh:  Optional[bool]= False
    cancer:  Optional[bool]= False
    phau_thuat:  Optional[bool]= False

    rbc: Optional[float]= None
    hb: Optional[float]=  None 
    mcv: Optional[float]=  None 
    mchc: Optional[float]=  None 
    rdw: Optional[float]=  None 
    ret_he: Optional[float]=  None 

    fe: Optional[float]=  None 
    ferritin: Optional[float]=  None 
    transferrin: Optional[float]=  None 
    tibc: Optional[float]=  None 
    #stfr: Optional[float]=  None 
    crp: Optional[float]=  None 

    diagnoses: Optional[str]= None
    #reasons: Optional[str|None]= None
"""
