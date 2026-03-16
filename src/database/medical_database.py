import os
import sys
import sqlite3
from contextlib import contextmanager

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dataclass.schema import Patient, DiagnosisRecord

import sqlite3
import os


class MedicalDatabase:
    def __init__(self, db_path=None):

        if db_path is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(BASE_DIR, "data", "medical.db")

        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path


    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


    def create_tables(self):

        with self.get_connection() as conn:

            conn.execute("""
            CREATE TABLE IF NOT EXISTS patients(
                id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                dob TEXT NOT NULL,
                gender TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                address TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS records(
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,

                kinh_nguyet BOOLEAN,
                da_day BOOLEAN,
                tri BOOLEAN,
                pregnant BOOLEAN,
                diet BOOLEAN,

                man_tinh BOOLEAN,
                cancer BOOLEAN,
                phau_thuat BOOLEAN,

                rbc REAL,
                hb REAL,
                mcv REAL,
                mchc REAL,
                rdw REAL,
                ret_he REAL,

                fe REAL,
                ferritin REAL,
                transferrin REAL,
                tibc REAL,
                stfr REAL,
                crp REAL,

                dotbiengen BOOLEAN,
                hba REAL,
                hba2 REAL,
                hbf REAL,
                hbh REAL,
                hbe REAL,
                hbc REAL,
                hbs REAL,
                hbbart REAL,
                hb_other REAL,

                diagnoses TEXT,
                reasons TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
            """)


    def add_patient(self, patient: Patient):

        with self.get_connection() as conn:

            conn.execute("""
                INSERT INTO patients
                (id, full_name, dob, gender, phone_number, address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                patient.id,
                patient.full_name,
                patient.dob,
                patient.gender,
                patient.phone_number,
                patient.address
            ))

        return patient.id


    def add_record(self, record: DiagnosisRecord):

        with self.get_connection() as conn:

            conn.execute("""
                INSERT INTO records(
                    id, patient_id, kinh_nguyet, da_day,
                    tri, pregnant, diet, man_tinh, cancer, phau_thuat,
                    rbc, hb, mcv, mchc, rdw, ret_he,
                    fe, ferritin, transferrin, tibc, stfr, crp,
                    dotbiengen, hba, hba2, hbf, hbh, hbe, hbc,
                    hbs, hbbart, hb_other, diagnoses, reasons
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record.id, record.patient_id,
                record.kinh_nguyet, record.da_day,
                record.tri, record.pregnant, record.diet,
                record.man_tinh, record.cancer, record.phau_thuat,
                record.rbc, record.hb, record.mcv, record.mchc,
                record.rdw, record.ret_he,
                record.fe, record.ferritin, record.transferrin,
                record.tibc, record.stfr, record.crp,
                record.dotbiengen, record.hba, record.hba2,
                record.hbf, record.hbh, record.hbe, record.hbc,
                record.hbs, record.hbbart, record.hb_other,
                record.diagnoses, record.reasons
            ))


    def search_by_field(self, value):

        search_fields = [
            "p.full_name",
            "p.phone_number",
            "p.address",

            "r.diagnoses",
            "r.reasons"
        ]

        conditions = " OR ".join([f"{field} LIKE ? COLLATE NOCASE" for field in search_fields])

        query = f"""
            SELECT
                p.id AS patient_id,
                p.full_name,
                p.phone_number,
                p.address,
                p.dob,
                p.gender,

                r.id AS record_id,
                r.kinh_nguyet,
                r.da_day,
                r.tri,
                r.pregnant,
                r.diet,
                r.man_tinh,
                r.cancer,
                r.phau_thuat,

                r.dotbiengen,

                r.diagnoses,
                r.reasons

            FROM patients p
            LEFT JOIN records r
            ON p.id = r.patient_id

            WHERE {conditions}
            LIMIT 20
        """

        search_val = "%" + value + "%"

        with self.get_connection() as conn:
            cursor = conn.execute(query, (search_val,) * len(search_fields))
            rows = cursor.fetchall()

        return [dict(row) for row in rows]
    
    def get_record_by_id(self, record_id):

        query = """
            SELECT
                r.id AS record_id,
                r.patient_id,

                p.full_name,
                p.dob,
                p.gender,
                p.phone_number,
                p.address,

                r.kinh_nguyet,
                r.da_day,
                r.tri,
                r.pregnant,
                r.diet,

                r.man_tinh,
                r.cancer,
                r.phau_thuat,

                r.rbc,
                r.hb,
                r.mcv,
                r.mchc,
                r.rdw,
                r.ret_he,

                r.fe,
                r.ferritin,
                r.transferrin,
                r.tibc,
                r.stfr,
                r.crp,

                r.dotbiengen,

                r.hba,
                r.hba2,
                r.hbf,
                r.hbh,
                r.hbe,
                r.hbc,
                r.hbs,
                r.hbbart,
                r.hb_other,

                r.diagnoses,
                r.reasons,

                r.created_at
            FROM records r
            JOIN patients p
            ON r.patient_id = p.id
            WHERE r.id = ?
        """

        with self.get_connection() as conn:
            cursor = conn.execute(query, (record_id,))
            row = cursor.fetchone()

        if row:
            return dict(row)

        return None
            


        
