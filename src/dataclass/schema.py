from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class Input():
    gender: str 
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
    stfr: Optional[float]=  None 
    crp: Optional[float]=  None 

    dotbiengen: Optional[bool]= False
    hba: Optional[float]=  None 
    hba2: Optional[float]=  None 
    hbf: Optional[float]=  None 
    hbh: Optional[float]=  None 
    hbe: Optional[float]=  None 
    hbc: Optional[float]=  None 
    hbs: Optional[float]=  None
    hbbart: Optional[float]=  None 
    hb_other: Optional[float]=  None 


@dataclass
class Output():
    diagnoses: Optional[List[str]]
    reasons: Optional[List[str]]
    # confidence: Optional[float]
    # scores: Optional[Dict[str, float]]

@dataclass
class Config():
    # weights: dict= field(default_factory= lambda: {})
    thres: dict= field(default_factory= lambda: {
        'hb': {"pregnant": 110, 
                "nữ": 120, 
                "nam": 130}, #thresholds cho PN có thai, nữ và nam

        'ferritin': [30, 100],
        'rdw': 0.15,
        'mentzer': 13,
        'hbbart': 0,
        'hbh': 0,
        'hba2': 0.035,
        'hbf': 0.01,
        'hbe': 0,
        'hbs': 0,
        'hbc': 0,
        'hb_other': 0,
        'tsat': [0.2, 0.3],
        'crp': 5, 
        'stfr_fer_idx' : [1, 2]
    })

    labels= [
            'Thalassemia',
            "IDA",
            "CRNN",
            "ACD",
            "IDA/ACD",
            'alpha-Thalassemia',
            'beta-Thalassemia'
        ]


