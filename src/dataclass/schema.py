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
    # pcv: Optional[float]= None
    mch: Optional[float]= None

    fe: Optional[float]=  None 
    ferritin: Optional[float]=  None 
    transferrin: Optional[float]=  None 
    tibc: Optional[float]=  None 
    # stfr: Optional[float]=  None 
    tsat: Optional[float]=  None
    crp: Optional[float]=  None 

    dotbiengen: Optional[str]= False
    gen_alpha: Optional[bool]= False
    gen_beta: Optional[bool]= False
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
    reasons: Optional[str]
    # confidence: Optional[float]
    # scores: Optional[Dict[str, float]]

@dataclass
class Config():
    # weights: dict= field(default_factory= lambda: {})
    thres: dict= field(default_factory= lambda: {
        # 'hb': {"pregnant": 110, 
                #"nữ": 120, 
                #"nam": 130}, #thresholds cho PN có thai, nữ và nam

        'ferritin': [30, 100],
        #'rdw': 0.15,
        #'mentzer': 13,
        'hbbart':1,
        'hbh': 1,
        'hba': [96.5, 98],
        'hba2': [2, 3.5],
        'hbf': 1,
        #'hbe': 0,
        #'hbs': 0,
        #'hbc': 0,
        #'hb_other': 0,
        'tsat': 20,
        'crp': 5,
        'fe': 5.8,
        'mcv': 80
        # 'stfr_fer_idx' : [1, 2]
    })

    labels= [
            "IDA",
            "ACD",
            'Alpha thalassemia',
            'Beta thalassemia'
            ]

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

    dotbiengen: Optional[str]= None
    hba: Optional[float]=  None 
    hba2: Optional[float]=  None 
    hbf: Optional[float]=  None 
    hbh: Optional[float]=  None 
    hbe: Optional[float]=  None 
    #hbc: Optional[float]=  None 
    hbs: Optional[float]=  None
    hbbart: Optional[float]=  None 
    hb_other: Optional[float]=  None 


    diagnoses: Optional[str]= None
    reasons: Optional[str|None]= None

