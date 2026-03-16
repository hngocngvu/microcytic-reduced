
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import sys 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dataclass.schema import Patient, DiagnosisRecord

import streamlit as st

class MedicalDatabase:
    def __init__(self, db_url=None):
        """
        PostgreSQL connection
        db_url: postgresql://user:pass@host:port/dbname
        """
        if db_url is None:
            # Lấy từ secrets
            db_url = (
                f"postgresql://{st.secrets['db']['user']}:{st.secrets['db']['password']}"
                f"@{st.secrets['db']['host']}:{st.secrets['db'].get('port', 5432)}"
                f"/{st.secrets['db']['name']}?sslmode=require"
            )
        
        self.db_url = db_url

    @contextmanager
    def get_connection(self):
        """Context manager: tự động mở-đóng"""
        conn = None
        try:
            conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
            yield conn
            conn.commit()  
        except Exception as e:
            if conn:
                conn.rollback() 
            raise e
        finally:
            if conn:
                conn.close()

    def create_tables(self):
        """Tạo bảng PostgreSQL"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Bảng patients
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients(
                    id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    dob DATE NOT NULL,          
                    gender TEXT NOT NULL,
                    phone_number TEXT NOT NULL,
                    address TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Bảng records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS records(
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE, 

                    kinh_nguyet BOOLEAN DEFAULT FALSE,
                    da_day BOOLEAN DEFAULT FALSE,
                    tri BOOLEAN DEFAULT FALSE,
                    pregnant BOOLEAN DEFAULT FALSE,
                    diet BOOLEAN DEFAULT FALSE,

                    man_tinh BOOLEAN DEFAULT FALSE,
                    cancer BOOLEAN DEFAULT FALSE,
                    phau_thuat BOOLEAN DEFAULT FALSE,

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

                    dotbiengen BOOLEAN DEFAULT FALSE,
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

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_records_patient 
                ON records(patient_id)
            """)
            
            # Trigger tự động cập nhật updated_at (PostgreSQL)
            cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            cursor.execute("""
                DROP TRIGGER IF EXISTS update_patients_updated_at ON patients;
                CREATE TRIGGER update_patients_updated_at
                    BEFORE UPDATE ON patients
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """)
            
            cursor.execute("""
                DROP TRIGGER IF EXISTS update_records_updated_at ON records;
                CREATE TRIGGER update_records_updated_at
                    BEFORE UPDATE ON records
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """)

    def add_patient(self, patient: Patient):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patients
                (id, full_name, dob, gender, phone_number, address)
                VALUES (%s, %s, %s, %s, %s, %s) 
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
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO records(
                    id, patient_id, kinh_nguyet, da_day,
                    tri, pregnant, diet, man_tinh, cancer, phau_thuat,
                    rbc, hb, mcv, mchc, rdw, ret_he,
                    fe, ferritin, transferrin, tibc, stfr, crp,
                    dotbiengen, hba, hba2, hbf, hbh, hbe, hbc,
                    hbs, hbbart, hb_other, diagnoses, reasons
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s) 
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
        
        return record.id

    def search_by_field(self, value):
        search_fields = [
            "p.full_name",
            "p.phone_number",
            "p.address",
            "r.diagnoses",
            "r.reasons"
        ]

        conditions = " OR ".join([f"{field} ILIKE %s" for field in search_fields])

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
            cursor = conn.cursor()
            cursor.execute(query, (search_val,) * len(search_fields))
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_record_by_id(self, record_id):
        """Lấy record theo ID - %s thay vì ?"""
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
            WHERE r.id = %s  
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (record_id,))  
            row = cursor.fetchone()

        if row:
            return dict(row)

        return None

        
