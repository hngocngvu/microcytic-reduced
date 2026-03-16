import io
import pandas as pd
from docx import Document
from reportlab.pdfgen import canvas

def export_excel(record):
    """Trả về bytes của file Excel"""
    df = pd.DataFrame([record])
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    return buffer.getvalue()  # Trả về bytes

def export_word(record):
    """Trả về bytes của file Word"""
    doc = Document()
    doc.add_heading('Báo cáo chẩn đoán', 0)
    
    for key, value in record.items():
        doc.add_paragraph(f"{key}: {value}")
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def export_pdf(record):
    """Trả về bytes của file PDF"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    
    y = 800
    c.drawString(100, y, "BÁO CÁO CHẨN ĐOÁN")
    y -= 30
    
    for key, value in record.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()