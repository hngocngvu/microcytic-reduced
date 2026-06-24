"""
Download and merge NHANES datasets for anemia research.
Files: CBC, Ferritin, Iron/TIBC/TSAT, hs-CRP, Demographics, sTfR.

NHANES public-use data: free to download, no registration required.
Data Use Agreement (implicit): use for statistical analysis only,
do not attempt to identify individuals.

Usage:
    pip install pandas requests
    python nhanes_download.py
"""

import os
import sys
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nhanes_data")
OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "data.csv"
)

CYCLES = {
    "F": {
        "year": "2009",
        "files": {
            "demo": "DEMO_F.xpt",
            "cbc": "CBC_F.xpt",
            "ferritin": "FERTIN_F.xpt",
            "iron": "FETIB_F.xpt",
            "crp": "HSCRP_F.xpt",
            "stfr": "TFR_F.xpt",
        }
    },

    "G": {
        "year": "2011",
        "files": {
            "demo": "DEMO_G.xpt",
            "cbc": "CBC_G.xpt",
            "ferritin": "FERTIN_G.xpt",
            "iron": "FETIB_G.xpt",
            "crp": "HSCRP_G.xpt",
            "stfr": "TFR_G.xpt",
        }
    },

    "H": {
        "year": "2013",
        "files": {
            "demo": "DEMO_H.xpt",
            "cbc": "CBC_H.xpt",
            "ferritin": "FERTIN_H.xpt",
            "iron": "FETIB_H.xpt",
            "crp": "HSCRP_H.xpt",
            "stfr": "TFR_H.xpt",
        }
    },

    "I": {
        "year": "2015",
        "files": {
            "demo": "DEMO_I.xpt",
            "cbc": "CBC_I.xpt",
            "ferritin": "FERTIN_I.xpt",
            "iron": "FETIB_I.xpt",
            "crp": "HSCRP_I.xpt",
            "stfr": "TFR_I.xpt",
        }
    },

    "J": {
        "year": "2017",
        "files": {
            "demo": "DEMO_J.xpt",
            "cbc": "CBC_J.xpt",
            "ferritin": "FERTIN_J.xpt",
            "iron": "FETIB_J.xpt",
            "crp": "HSCRP_J.xpt",
            "stfr": "TFR_J.xpt",
        }
    }
}

DATASETS = {
    "demo": {
        "file": "DEMO_{}.xpt",
        "desc": "Demographics",
        "cols": ["SEQN", "RIDAGEYR", "RIAGENDR"],
    },

    "cbc": {
        "file": "CBC_{}.xpt",
        "desc": "Complete Blood Count",
        "cols": [
            "SEQN",
            "LBXHGB",
            "LBXRBCSI",
            "LBXHCT",
            "LBXMCVSI",
            "LBXMCHSI",
            "LBXMC",
            "LBXRDW",
            "LBXPLTSI",
            "LBXWBCSI",
        ],
    },

    "ferritin": {
        "file": "FERTIN_{}.xpt",
        "desc": "Ferritin",
        "cols": ["SEQN", "LBXFER"],
    },


    "iron": {
        "file": "FETIB_{}.xpt",
        "desc": "Iron/TIBC/TSAT",
        "cols": ["SEQN", "LBXIRN", "LBDTIB", "LBDPCT"],
    },

    "crp": {
        "file": "HSCRP_{}.xpt",
        "desc": "hsCRP",
        "cols": ["SEQN", "LBXHSCRP"],
    },

    "stfr": {
        "file": "TFR_{}.xpt",
        "desc": "sTfR",
        "cols": [
            "SEQN",
            "LBXTFR",
        ],
    },
}

RENAME = {
    "RIDAGEYR": "age",
    "RIAGENDR": "gender",
    "LBXHGB": "hemoglobin",
    "LBXRBCSI": "rbc",
    "LBXHCT": "hematocrit",
    "LBXMCVSI": "mcv",
    "LBXMCHSI": "mch",
    "LBXMC": "mchc",
    "LBXRDW": "rdw",
    "LBXPLTSI": "platelet",
    "LBXWBCSI": "wbc",
    "LBXFER": "ferritin",
    "LBXIRN": "serum_iron",
    "LBDTIB": "tibc",
    "LBDPCT": "tsat",
    "LBXHSCRP": "hscrp",
    "LBXTFR":   "stfr",        # sTfR in mg/L
}

GENDER_MAP = {1: "Male", 2: "Female"}


def download_xpt(name, cycle):

    if name not in CYCLES[cycle]["files"]:
        print(f"  [skip] {name}: not available in cycle {cycle}")
        return None

    year = CYCLES[cycle]["year"]
    filename = CYCLES[cycle]["files"][name]

    base_url = (
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
        f"{year}/DataFiles"
    )

    url = f"{base_url}/{filename}"

    xpt_path = os.path.join(
        DATA_DIR,
        f"{name}_{cycle}.xpt"
    )

    if os.path.exists(xpt_path):
        print(f"  [cached] {filename}")

    else:
        print(f"  [downloading] {filename}")

        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [error] {filename}: {e}")
            return None

        with open(xpt_path, "wb") as f:
            f.write(r.content)

    df = pd.read_sas(xpt_path, format="xport")

    available = [
        c
        for c in DATASETS[name]["cols"]
        if c in df.columns
    ]

    missing = set(DATASETS[name]["cols"]) - set(available)

    if missing:
        print(
            f"    [warning] {filename}: "
            f"missing columns {missing}"
        )

    if "SEQN" not in available:
        print(f"  [error] {filename}: missing SEQN column")
        return None

    return df[available]


def add_stfr_index(df: pd.DataFrame) -> pd.DataFrame:
    if "stfr" not in df.columns or "ferritin" not in df.columns:
        print("  [skip] sTfR Index: missing stfr or ferritin column")
        return df

    valid = (df["stfr"] > 0) & (df["ferritin"] > 0)
    df["stfr_index"] = np.where(
        valid,
        df["stfr"] / np.log10(df["ferritin"].where(valid, np.nan)),
        np.nan,
    )
    n_valid = valid.sum()
    n_total = len(df)
    print(f"  sTfR Index computed for {n_valid}/{n_total} rows "
          f"({n_valid/n_total*100:.1f}%); NaN for zero/missing values.")
    return df


def add_anemia_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    anemic = (
        ((df["gender"] == "Male") & (df["hemoglobin"] < 13))
        | ((df["gender"] == "Female") & (df["hemoglobin"] < 12))
    )

    microcytic = df["mcv"] < 80
    base = anemic & microcytic

    muc_d_ferritin = df["ferritin"] < 30
    muc_u_ferritin = df["ferritin"] >= 100
    muc_l_ferritin = df["ferritin"] < 100
    muc_crp_high = df["hscrp"] > 10
    #muc_stfr_l_index = df["stfr_index"] >= 2
    #muc_stfr_i_index = df["stfr_index"] >= 3.2

    if "tsat" in df.columns and "ferritin" in df.columns:
        has_iron = df["tsat"].notna() & df["ferritin"].notna()
    else:
        has_iron = pd.Series(False, index=df.index)

    muc_tsat = df["tsat"] < 16 if "tsat" in df.columns else pd.Series(False, index=df.index)

    conditions = [
        # Co du chi so sat (serum_iron, tibc, tsat, ferritin)
        base & has_iron & (~ muc_crp_high) & (muc_tsat & muc_d_ferritin), #| muc_stfr_i_index),
        base & has_iron & muc_crp_high & (muc_tsat & muc_u_ferritin), #| (~ muc_stfr_l_index & ~muc_d_ferritin)) ,
        base & has_iron & muc_crp_high & (muc_tsat & muc_l_ferritin) #| muc_stfr_l_index),

        # Khong co chi so sat — fallback dung stfr_index
        #base & ~has_iron & muc_stfr_i_index & muc_crp_low,
        #base & ~has_iron & (~muc_stfr_l_index & ~muc_d_ferritin) & muc_crp_high,
        #base & ~has_iron & muc_stfr_l_index & muc_crp_high,
    ]
    labels = ["IDA", "ACD", "IDA/ACD"] #"IDA", "ACD", "IDA/ACD"]

    df["anemia_class"] = np.select(conditions, labels, default="Unclassified")

    classified = df["anemia_class"] != "Unclassified"
    print(f"  Classified: {classified.sum()}/{len(df)}")
    with_iron = (classified & has_iron).sum()
    without_iron = (classified & ~has_iron).sum()
    print(f"    With iron panel: {with_iron}, Without (sTfR fallback): {without_iron}")

    df = df[classified]
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("NHANES 2013-2018 Anemia Dataset Downloader")
    print(f"Output directory: {DATA_DIR}\n")

    print("[1/4] Downloading NHANES cycles...")

    all_cycles = []

    for cycle, cycle_info in CYCLES.items():

        year = cycle_info["year"]

        print(f"\n===== {year}-{int(year)+1} =====")

        frames = {}

        for name in DATASETS:
            result = download_xpt(name, cycle)
            if result is not None:
                frames[name] = result

        required = ["demo", "cbc", "ferritin", "crp"]
        missing_req = [r for r in required if r not in frames]
        if missing_req:
            print(f"  [skip cycle {cycle}] missing required datasets: {missing_req}")
            continue

        print(f"[2/4] Merging {cycle}...")

        df_cycle = frames["demo"]

        for name in ["cbc", "ferritin", "crp"]:
            df_cycle = df_cycle.merge(
                frames[name],
                on="SEQN",
                how="inner"
            )

        for name in ["iron", "stfr"]:
            if name in frames:
                df_cycle = df_cycle.merge(
                    frames[name],
                    on="SEQN",
                    how="left"
                )

        df_cycle["cycle"] = cycle

        print(f"  {cycle}: {len(df_cycle)} rows")

        all_cycles.append(df_cycle)

    # merge all cycles
    df = pd.concat(
        all_cycles,
        ignore_index=True
    )

    print(f"\nMerged records across all cycles: {len(df)}")

    # rename columns
    df.rename(columns=RENAME, inplace=True)
    df["gender"] = df["gender"].map(GENDER_MAP)

    # drop rows with missing core labs
    complete_before = len(df)

    core_labs = [
        "hemoglobin",
        "mcv",
        "ferritin",
        "hscrp",
        "stfr",
    ]

    df = df.dropna(subset=core_labs)

    print(
        f"  After dropping rows with missing core lab values: "
        f"{len(df)} (dropped {complete_before-len(df)})"
    )

    for col in ["stfr", "serum_iron", "tibc", "tsat"]:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  Rows with {col}: {n}/{len(df)} ({n/len(df)*100:.1f}%)")

    print("\n[3/4] Computing derived features...")
    df = add_stfr_index(df)

    print("\n[4/4] Adding anemia classification labels...")
    df = add_anemia_labels(df)

    class_counts = df["anemia_class"].value_counts()

    print("\nLabel distribution:")
    for label, count in class_counts.items():
        print(
            f"  {label:20s}"
            f"{count:>6d}"
            f" ({count/len(df)*100:.1f}%)"
        )

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns in output:")
    for col in df.columns:
        print(f"  - {col}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()