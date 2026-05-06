from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import date
import numpy as np 

@dataclass
class Input():
    gender: Optional[str]= None
    kinh_nguyet: Optional[bool]= False
    da_day:  Optional[bool]= False
    tri:  Optional[bool]= False
    pregnant:  Optional[bool]= False
    diet:  Optional[bool]= False
    age: Optional[int]= np.nan

    man_tinh:  Optional[bool]= False
    cancer:  Optional[bool]= False
    phau_thuat:  Optional[bool]= False

    rbc: Optional[float]= np.nan
    hb: Optional[float]=  np.nan
    mcv: Optional[float]=  np.nan 
    mchc: Optional[float]=  np.nan 
    rdw: Optional[float]=  np.nan
    ret_he: Optional[float]=  np.nan
    # pcv: Optional[float]= np.nan
    mch: Optional[float]= np.nan

    fe: Optional[float]=  np.nan 
    ferritin: Optional[float]=  np.nan 
    transferrin: Optional[float]=  np.nan 
    tibc: Optional[float]=  np.nan 
    # stfr: Optional[float]=  np.nan 
    tsat: Optional[float]=  np.nan
    crp: Optional[float]=  np.nan 

    dotbiengen: Optional[bool]= False
    hba: Optional[float]=  np.nan 
    hba2: Optional[float]=  np.nan 
    hbf: Optional[float]=  np.nan 
    hbh: Optional[float]=  np.nan 
    hbe: Optional[float]=  np.nan 
    # hbc: Optional[float]=  np.nan 
    hbs: Optional[float]=  np.nan
    hbbart: Optional[float]=  np.nan 
    hb_other: Optional[float]=  np.nan 


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
            'alpha Thalassemia',
            'beta Thalassemia',
            'Thalassemia'
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

    rbc: Optional[float]= np.nan
    hb: Optional[float]=  np.nan 
    mcv: Optional[float]=  np.nan 
    mchc: Optional[float]=  np.nan 
    rdw: Optional[float]=  np.nan 
    ret_he: Optional[float]=  np.nan 

    fe: Optional[float]=  np.nan 
    ferritin: Optional[float]=  np.nan 
    transferrin: Optional[float]=  np.nan 
    tibc: Optional[float]=  np.nan 
    #stfr: Optional[float]=  np.nan 
    crp: Optional[float]=  np.nan 

    dotbiengen: Optional[bool]= False
    hba: Optional[float]=  np.nan 
    hba2: Optional[float]=  np.nan 
    hbf: Optional[float]=  np.nan 
    hbh: Optional[float]=  np.nan 
    hbe: Optional[float]=  np.nan 
    #hbc: Optional[float]=  np.nan 
    hbs: Optional[float]=  np.nan
    hbbart: Optional[float]=  np.nan 
    hb_other: Optional[float]=  np.nan 


    diagnoses: Optional[str]= None
    reasons: Optional[str|None]= None

