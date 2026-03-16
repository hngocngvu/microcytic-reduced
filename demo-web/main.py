import streamlit as st 
import os 
import sys 
import pandas as pd 

# mentzer= mcv/rbc
# tsat%= fe*100%/tibc or fe*70.9/transferin
# stfr/fer_idx= stfr/log(ferritin)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from src.dataclass.schema import Input, Output, Config
from src.models.classifier import Classifier
from src.functions.function import parse_number 
from src.functions.utils import export_excel, export_word, export_pdf 
from src.dataclass.schema import Patient, DiagnosisRecord
from src.database.medical_database import MedicalDatabase
import datetime

today = datetime.date.today()

page = st.sidebar.selectbox(
    "Menu",
    ["Chẩn đoán", "Tìm kiếm bệnh nhân", "Xuất báo cáo"]
)

st.set_page_config(layout="wide")

if page == "Chẩn đoán":
    st.markdown(
        "<h1 style='text-align: center;'>Hệ thống chẩn đoán bệnh về hồng cầu nhỏ</h1>",
        unsafe_allow_html=True
    )



    c1, c2, c3 = st.columns(3)

    with c1:
        full_name= st.text_input("Họ và Tên")
        dob= st.date_input("Ngày sinh",    
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date.today())
        gender = st.radio(
        "Giới tính",
        ["Nam", "Nữ","Khác"]
            )

    with c2:
        phone_number= st.text_input("Số điện thoại liên hệ")
        address= st.text_input("Địa chỉ")

    with c3:
        patient_id= st.text_input("Mã số bệnh nhân")
        record_id= st.text_input("Mã số hồ sơ")

    dob_str = dob.isoformat()

    if gender == "Nữ":
        kinh_nguyet= st.checkbox("Kinh nguyệt nhiều, kéo dài")
        pregnant= st.checkbox("Có thai")
    else:
        kinh_nguyet= False
        pregnant= False



    with st.form("form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Tiền sử")
            
            da_day= st.checkbox("Viêm loét dạ dày, tá tràng hoặc tiền sử cẳt dạ dày, tá tràng")
            # st.write(da_day)

            tri= st.checkbox("Trĩ chảy máu")
            # st.write(tri)

            diet= st.checkbox("Chế độ ăn kiêng, ăn chay")
            # st.write(diet)

            # Danh sách bệnh

            danh_sach_benh = ["Gout", "Viêm khớp", "Lupus ban đỏ hệ thống", "Bệnh thận mạn", "Viêm gan mạn", "Loãng xương", 
                              "Rối loạn lipid", "Rối loạn tiền đình", "Viêm gân gấp", "Áp xe gan", "Sốt kéo dài",
                              "Quá tải sắt", "Đau ngực", "U lành", "HIV", "Giãn động mạch", "Polyp đại tràng", "Polyp dạ dày",
                              "Nhồi máu não", "Stent mạch vành"]

            # Khởi tạo (chỉ chạy 1 lần đầu)
            if "tien_su_text" not in st.session_state:
                st.session_state.tien_su_text = ""  # Lưu nội dung text area
            if "da_chon" not in st.session_state:
                st.session_state.da_chon = ""  # Lưu bệnh vừa chọn

            # Dropdown chọn bệnh
            selected = st.selectbox(
                "Chọn tiền sử hoặc bệnh kèm theo",
                options=[""] + danh_sach_benh
            )

            # Nếu chọn bệnh MỚI (khác lần trước) → thêm vào text
            if selected and selected != st.session_state.da_chon:
                if st.session_state.tien_su_text:
                    st.session_state.tien_su_text += f"\n- {selected}"
                else:
                    st.session_state.tien_su_text = f"- {selected}"
                st.session_state.da_chon = selected  # Ghi nhớ đã chọn rồi

            # Text area hiển thị + cho sửa
            tien_su = st.text_area(
                "Bổ sung tiền sử (có thể chỉnh sửa)",
                value=st.session_state.tien_su_text,
                height=150
            )

            # Lưu lại nếu user tự sửa trong text area
            st.session_state.tien_su_text = tien_su

            man_tinh = bool(selected) or bool(tien_su.strip())
            # st.write(man_tinh)

            cancer= st.checkbox("Ung thư đang điều trị")
            # st.write(cancer)

            phau_thuat= st.checkbox("Sau phẫu thuật lớn, bỏng, chấn thương")
            # st.write(phau_thuat)

        with col2:
            st.subheader("Các đặc điểm hồng cầu")
            hb= parse_number(st.text_input("Hemoglobin (Hb)"))
            # st.write(hb)

            mcv= parse_number(st.text_input("MCV"))
            # st.write(mcv)

            mchc= parse_number(st.text_input("MCHC"))
            # st.write(mchc)

            rbc= parse_number(st.text_input("RBC"))
            # st.write(rbc)

            rdw= parse_number(st.text_input("RDW-CV"))
            # st.write(rdw)

            ret_he= parse_number(st.text_input("Ret-He"))
            # st.write(ret_he)

        with col3: 
            st.subheader("Chỉ số hóa sinh máu")
            fe= parse_number(st.text_input("Định lượng Sắt huyết thanh (Fe)"))
            # st.write(fe)

            ferritin= parse_number(st.text_input("Định lượng Ferritin"))
            # st.write(ferritin)

            transferrin= parse_number(st.text_input("Định lượng Transferrin"))
            # st.write(transferrin)

            tibc= parse_number(st.text_input("TIBC (khả năng gắn sắt toàn phần: máy đo)"))
            # st.write(tibc)

            stfr= parse_number(st.text_input("Nồng độ thụ thể transferrin hòa tan"))    
            # st.write(stfr)

            crp= parse_number(st.text_input("CRP"))
            # st.write(crp)

        with col4: 
            st.subheader("Xét nghiệm di huyết sắt tố và đột biến gen")
            hba= parse_number(st.text_input("HbA"))
            # st.write(hba)

            hba2= parse_number(st.text_input("HbA2"))
            # st.write(hba2)

            hbf= parse_number(st.text_input("HbF"))
            # st.write(hbf)

            hbh= parse_number(st.text_input("HbH"))
            # st.write(hbh)

            hbe= parse_number(st.text_input("HbE"))
            # st.write(hbe)

            hbc= parse_number(st.text_input("HbC"))
            # st.write(hbc)

            hbs= parse_number(st.text_input("HbS"))
            # st.write(hbs)

            hbbart= parse_number(st.text_input("Hb Bart"))
            # st.write(hbbart)

            hb_other= parse_number(st.text_input("Chỉ số Hb khác nếu có"))
            # st.write(hb_other)

            dotbiengen= st.checkbox("Đột biến gen thalassemia")
            # st.write(dotbiengen)

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")
                                        
        if submitted:
            config= Config()

            patient= Input(gender,
            kinh_nguyet= kinh_nguyet,
            da_day= da_day,
            tri= tri,
            pregnant= pregnant,
            diet= diet,

            man_tinh= man_tinh,
            cancer= cancer,
            phau_thuat= phau_thuat,

            rbc= rbc,
            hb= hb,
            mcv= mcv,
            mchc= mchc,
            rdw= rdw,
            ret_he= ret_he,


            fe=fe,
            ferritin= ferritin,
            transferrin= transferrin,
            tibc= tibc,
            stfr= stfr,
            crp= crp,

            dotbiengen= dotbiengen,
            hba= hba,
            hba2= hba2,
            hbf= hbf,
            hbh= hbh,
            hbe= hbe, 
            hbc= hbc,
            hbs= hbs,
            hbbart= hbbart,
            hb_other= hb_other,

            )

            c= Classifier(patient, config)
            result= c.classify()

            st.subheader("Kết quả chẩn đoán")

            st.success(result.diagnoses)

            st.subheader("Nguyên nhân và gợi ý")

            st.info(result.reasons)

            info= Patient(id= patient_id, full_name= full_name,
                        dob= dob_str, gender= gender, 
                        phone_number= phone_number, address= address
                        )
            record= DiagnosisRecord(
                id= record_id, patient_id= patient_id,
                kinh_nguyet= kinh_nguyet, da_day= da_day,
                tri= tri, pregnant= pregnant, diet= diet,
                man_tinh= man_tinh, cancer= cancer, phau_thuat= phau_thuat,
                rbc= rbc, hb=hb, mcv= mcv, mchc= mchc, rdw= rdw, ret_he= ret_he,
                fe= fe, ferritin= ferritin, transferrin= transferrin,
                tibc= tibc, stfr= stfr, crp= crp, dotbiengen= dotbiengen,
                hba= hba, hba2= hba2, hbf= hbf, hbh= hbh, hbe= hbe, hbc= hbc,
                hbs= hbs, hbbart= hbbart, hb_other= hb_other, 
                diagnoses= result.diagnoses, reasons= result.reasons
            )
            db= MedicalDatabase()
            db.create_tables()
            db.add_patient(info)
            db.add_record(record)
            st.success("Đã lưu hồ sơ bệnh nhân thành công")

if page == "Tìm kiếm bệnh nhân":
    db= MedicalDatabase()

    st.markdown(
        "<h1 style='text-align: center;'>Tìm kiếm bệnh nhân</h1>",
        unsafe_allow_html=True
    )
    value = st.text_input("Nhập giá trị tìm kiếm")

    search_button = st.button("Search")

    if search_button and value:

        results = db.search_by_field(value)

        if not results:
            st.warning("Không tìm thấy kết quả")
        else:
            st.success(f"Tìm thấy {len(results)} kết quả")
            # st.write(results)

            for r in results:

                with st.expander(f"{r['full_name']}  |  Mã số hồ sơ: {r['record_id']}"):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Thông tin bệnh nhân")
                        st.write("Mã số bệnh nhân:", r['patient_id'])
                        st.write("Ngày sinh:", r["dob"])
                        st.write("Giới tính:", r["gender"])
                        st.write("SĐT:", r["phone_number"])
                        st.write("Địa chỉ:", r["address"])

                    with col2:
                        st.write("Kết quả chẩn đoán")
                        st.write("Chẩn đoán:", r["diagnoses"])
                        st.write("Nguyên nhân và gợi ý:", r["reasons"])

if page == "Xuất báo cáo":
    st.markdown(
        "<h1 style='text-align: center;'>Xuất thông tin và báo cáo</h1>",
        unsafe_allow_html=True
    )

    db = MedicalDatabase()
    record_id = st.text_input("Nhập mã hồ sơ", key="input_record_id")

    # Khởi tạo session state
    if 'loaded_record' not in st.session_state:
        st.session_state.loaded_record = None
        st.session_state.excel_bytes = None
        st.session_state.word_bytes = None
        st.session_state.pdf_bytes = None

    # Nút tải thông tin
    if st.button("Tải thông tin", key="btn_load"):
        record = db.get_record_by_id(record_id)
        
        if record is None:
            st.error("Không tìm thấy hồ sơ")
            st.session_state.loaded_record = None
        else:
            st.session_state.loaded_record = record
            # Tạo file ngay khi tải thông tin
            try:
                st.session_state.excel_bytes = export_excel(record)
                st.session_state.word_bytes = export_word(record)
                st.session_state.pdf_bytes = export_pdf(record)
                st.success("✅ Đã tải thông tin và tạo file thành công!")
            except Exception as e:
                st.error(f"Lỗi tạo file: {str(e)}")

    # Hiển thị thông tin và download buttons (không nằm trong if st.button nữa)
    if st.session_state.loaded_record:
        record = st.session_state.loaded_record
        
        st.subheader("Thông tin bệnh nhân")
        st.write("Họ và tên:", record["full_name"])
        st.write("Ngày sinh:", record["dob"])
        st.write("Giới tính:", record["gender"])
        st.write("SĐT:", record["phone_number"])
        st.write("Địa chỉ:", record["address"])
        st.write("Mã bệnh nhân:", record["patient_id"])
        st.write("Chẩn đoán:", record["diagnoses"])
        st.write("Nguyên nhân và gợi ý:", record["reasons"])

        st.subheader("Chỉ số xét nghiệm")
        df = pd.DataFrame([record])
        st.dataframe(df)
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="Xuất Excel",
                data=st.session_state.excel_bytes,
                file_name=f"diagnosis_{record_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )

        with col2:
            st.download_button(
                label="Xuất Word", 
                data=st.session_state.word_bytes,
                file_name=f"diagnosis_{record_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_word"
            )

        with col3:
            st.download_button(
                label="Xuất PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"diagnosis_{record_id}.pdf",
                mime="application/pdf",
                key="download_pdf"
            )