import streamlit as st
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io

from utils.image_processing import process_image

st.set_page_config(page_title="Digitalizador de Documentos", layout="centered")

st.title("📄 Digitalizador de Documentos")
st.write("Envie uma imagem do documento para corrigir perspectiva e exportar em PDF.")

uploaded_file = st.file_uploader("Selecione a imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem original", use_container_width=True)

    processed = process_image(np.array(image))

    st.image(processed, caption="Imagem processada", use_container_width=True)

    # Converter para PDF
    pdf_buffer = io.BytesIO()
    img_pil = Image.fromarray(processed)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    img = ImageReader(img_byte_arr)
    c.drawImage(img, 0, 0, width, height)
    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="📥 Baixar PDF",
        data=pdf_buffer,
        file_name="documento_digitalizado.pdf",
        mime="application/pdf",
    )
