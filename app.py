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

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    import io
    
    # Converter imagem processada para bytes
    img_pil = Image.fromarray(processed)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Criação do PDF
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    page_width, page_height = A4
    
    # Medidas da imagem e proporção
    img = ImageReader(img_bytes)
    img_width, img_height = img_pil.size
    aspect_ratio = img_width / img_height
    
    # Ajustar proporção à página
    max_width = page_width - 40
    max_height = page_height - 40
    if img_width > img_height:
        render_width = max_width
        render_height = render_width / aspect_ratio
    else:
        render_height = max_height
        render_width = render_height * aspect_ratio
    
    # Centralizar na página
    x = (page_width - render_width) / 2
    y = (page_height - render_height) / 2
    
    # Desenhar imagem e salvar
    c.drawImage(img, x, y, render_width, render_height)
    c.showPage()
    c.save()
    
    pdf_buffer.seek(0)
    
    st.download_button(
        label="📥 Baixar PDF",
        data=pdf_buffer,
        file_name="documento_digitalizado.pdf",
        mime="application/pdf",
    )

    st.download_button(
        label="📥 Baixar PDF",
        data=pdf_buffer,
        file_name="documento_digitalizado.pdf",
        mime="application/pdf",
    )
