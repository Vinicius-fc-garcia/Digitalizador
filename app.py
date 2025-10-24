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
st.write("Envie uma imagem de um documento. O app corrigirá perspectiva, removerá bordas e permitirá exportar em PDF.")

uploaded_file = st.file_uploader("Selecione a imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem original", use_container_width=True)

    # Processamento da imagem
    processed = process_image(np.array(image))
    st.image(processed, caption="Imagem processada", use_container_width=True)

    # Botão para gerar PDF
    if st.button("📄 Gerar PDF"):
        # Converter imagem processada para bytes
        img_pil = Image.fromarray(processed)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        # Criar PDF em memória
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        page_width, page_height = A4

        # Dimensões da imagem e ajuste de proporção
        img = ImageReader(img_bytes)
        img_width, img_height = img_pil.size
        aspect_ratio = img_width / img_height

        # Ajustar tamanho mantendo proporção
        max_width = page_width - 40
        max_height = page_height - 40
        if img_width > img_height:
            render_width = max_width
            render_height = render_width / aspect_ratio
        else:
            render_height = max_height
            render_width = render_height * aspect_ratio

        # Centralizar imagem
        x = (page_width - render_width) / 2
        y = (page_height - render_height) / 2

        # Desenhar imagem no PDF
        c.drawImage(img, x, y, render_width, render_height)
        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        # Exibir botão de download com chave única
        st.download_button(
            label="📥 Baixar documento_digitalizado.pdf",
            data=pdf_buffer,
            file_name="documento_digitalizado.pdf",
            mime="application/pdf",
            key="download_pdf_button"
        )

else:
    st.info("👆 Envie uma imagem para começar.")
