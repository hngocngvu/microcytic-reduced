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

    rbc: Optional[float]= 0
    hb: Optional[float]= 0
    mcv: Optional[float]= 0
    mchc: Optional[float]= 0
    rdw: Optional[float]= 0
    ret_he: Optional[float]= 0

    fe: Optional[float]= 0
    ferritin: Optional[float]= 0
    transferrin: Optional[float]= 0
    tibc: Optional[float]= 0
    stfr: Optional[float]= 0
    crp: Optional[float]= 0

    dotbiengen: Optional[bool]= False
    hba: Optional[float]= 0
    hba2: Optional[float]= 0
    hbf: Optional[float]= 0
    hbh: Optional[float]= 0
    hbe: Optional[float]= 0
    hbc: Optional[float]= 0
    hbs: Optional[float]= 0
    hbbart: Optional[float]= 0
    hb_other: Optional[float]= 0


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


