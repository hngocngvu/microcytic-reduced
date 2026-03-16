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
from src.functions.function import parse_number 

st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Hệ thống chẩn đoán bệnh về hồng cầu nhỏ</h1>",
    unsafe_allow_html=True
)

gender = st.radio(
    "Giới tính",
    ["Nam", "Nữ","Khác"]
        )
if gender == "Nữ":
    kinh_nguyet= st.checkbox("Kinh nguyệt nhiều, kéo dài")
    pregnant= st.checkbox("Có thai")
else:
    kinh_nguyet= False
    pregnant= False



with st.form("form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Thông tin chung và tiền sử")
        da_day= st.checkbox("Viêm loét dạ dày, tá tràng hoặc tiền sử cẳt dạ dày, tá tràng")
        tri= st.checkbox("Trĩ chảy máu")
        diet= st.checkbox("Chế độ ăn kiêng, ăn chay")

        st.subheader("Tiền sử liên quan đến thiếu máu trong tình trạng viêm")
        man_tinh= st.checkbox("Có bệnh nhiễm trùng/viêm mãn tính đang điều trị: gout, viêm khớp, nhiễm khuẩn, lupus ban đỏ hệ thống, bệnh thận mạn, viêm gan mạn ,...")
        cancer= st.checkbox("Ung thư đang điều trị")
        phau_thuat= st.checkbox("Sau phẫu thuật lớn, bỏng, chấn thương")

    with col2:
        st.subheader("Các đặc điểm hồng cầu")
        hb= parse_number(st.text_input("Hemoglobin (Hb)"))
        mcv= parse_number(st.text_input("MCV"))
        mchc= parse_number(st.text_input("MCHC"))
        rbc= parse_number(st.text_input("RBC"))
        rdw= parse_number(st.text_input("RDW-CV"))
        ret_he= parse_number(st.text_input("Ret-He"))

    with col3: 
        st.subheader("Chỉ số hóa sinh máu")
        fe= parse_number(st.text_input("Định lượng Sắt huyết thanh (Fe)"))
        ferritin= parse_number(st.text_input("Định lượng Ferritin"))
        transferrin= parse_number(st.text_input("Định lượng Transferrin"))
        tibc= parse_number(st.text_input("TIBC (khả năng gắn sắt toàn phần: máy đo)"))
        stfr= parse_number(st.text_input("Nồng độ thụ thể transferrin hòa tan"))    
        crp= parse_number(st.text_input("CRP"))

    with col4: 
        st.subheader("Xét nghiệm di huyết sắt tố và đột biến gen")
        hba= parse_number(st.text_input("HbA"))
        hba2= parse_number(st.text_input("HbA2"))
        hbf= parse_number(st.text_input("HbF"))
        hbh= parse_number(st.text_input("HbH"))
        hbe= parse_number(st.text_input("HbE"))
        hbc= parse_number(st.text_input("HbC"))
        hbs= parse_number(st.text_input("HbS"))
        hb_other= parse_number(st.text_input("Chỉ số Hb khác nếu có"))
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



