import streamlit as st 
import os 
import sys 

# mentzer= mcv/rbc
# tsat%= fe*100%/tibc or fe*70.9/transferin
# stfr/fer_idx= stfr/log(ferritin)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from src.dataclass.schema import Input, Output, Config
from src.models.classifier import Classifier

st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Hệ thống chẩn đoán bệnh về hồng cầu nhỏ</h1>",
    unsafe_allow_html=True
)
with st.form("form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Thông tin chung và tiền sử")
        gender = st.radio(
            "Giới tính",
            ["Nam", "Nữ","Khác"]
        )
        enter = st.form_submit_button("Enter")
        if enter:
            if gender == "Nữ":
                kinh_nguyet= st.checkbox("Kinh nguyệt nhiều, kéo dài")
                pregnant= st.checkbox("Có thai")

        da_day= st.checkbox("Viêm loét dạ dày, tá tràng hoặc tiền sử cẳt dạ dày, tá tràng")
        tri= st.checkbox("Trĩ chảy máu")
        diet= st.checkbox("Chế độ ăn kiêng, ăn chay")

        st.subheader("Tiền sử liên quan đến thiếu máu trong tình trạng viêm")
        man_tinh= st.checkbox("Có bệnh nhiễm trùng/viêm mãn tính đang điều trị: gout, viêm khớp, nhiễm khuẩn, lupus ban đỏ hệ thống, bệnh thận mạn, viêm gan mạn ,...")
        cancer= st.checkbox("Ung thư đang điều trị")
        phau_thuat= st.checkbox("Sau phẫu thuật lớn, bỏng, chấn thương")

    with col2:
        st.subheader("Các đặc điểm hồng cầu")
        hb= st.number_input("Hemoglobin (Hb)")
        mcv= st.number_input("MCV")
        mchc= st.number_input("MCHC")
        rbc= st.number_input("RBC")
        rdw= st.number_input("RDW")
        ret_he= st.number_input("Ret-He")

    with col3: 
        st.subheader("Chỉ số hóa sinh máu")
        fe= st.number_input("Định lượng Sắt huyết thanh (Fe)")
        ferritin= st.number_input("Định lượng Ferritin")
        transferrin= st.number_input("Định lượng Transferrin")
        tibc= st.number_input("TIBC (khả năng gắn sắt toàn phần: máy đo)")
        stfr= st.number_input("Nồng độ thụ thể transferrin hòa tan")    
        crp= st.number_input("CRP")

    with col4: 
        st.subheader("Xét nghiệm di huyết sắt tố và đột biến gen")
        hba= st.number_input("HbA")
        hba2= st.number_input("HbA2")
        hbf= st.number_input("HbF")
        hbh= st.number_input("HbH")
        hbe= st.number_input("HbE")
        hbc= st.number_input("HbC")
        hbs= st.number_input("HbS")
        hb_other= st.number_input("Chỉ số Hb khác nếu có")
        dotbiengen= st.checkbox("Đột biến gen thalassemia")

    submitted = st.form_submit_button("Submit")
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
        hb_other= hb_other,

        )
        c= Classifier(patient, config)
        result= c.classify()

        st.write("Kết quả chẩn đoán")
        st.write(result.diagnoses)

        st.write("Nguyên do và gợi ý (nếu có)")
        st.write(result.reasons)



