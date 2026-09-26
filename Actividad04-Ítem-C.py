import cv2
import numpy as np
import streamlit as st
 
 
st.set_page_config(
    page_title="Control de calidad por thresholding",
    page_icon="🔎",
    layout="wide",
)
 
st.title("Control de calidad industrial")
st.caption("Segmentación de posibles grietas oscuras mediante thresholding")
 
uploaded_file = st.file_uploader(
    "Cargar imagen de una superficie o pieza",
    type=["jpg", "jpeg", "png", "bmp"],
)
 
threshold_value = st.slider(
    "Umbral de intensidad",
    min_value=0,
    max_value=255,
    value=100,
)
 
if uploaded_file is None:
    st.info("Cargue una imagen para iniciar la inspección.")
    st.stop()
 
# Decodifica la imagen cargada.
file_bytes = np.frombuffer(
    uploaded_file.getvalue(),
    dtype=np.uint8,
)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
 
if image is None:
    st.error("No se pudo leer la imagen.")
    st.stop()
 
# Convierte la imagen a escala de grises.
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)
 
# Reduce pequeñas variaciones de intensidad.
blurred = cv2.GaussianBlur(
    gray,
    (3, 3),
    0,
)
 
# Segmenta las regiones oscuras mediante umbralización.
_, mask = cv2.threshold(
    blurred,
    threshold_value,
    255,
    cv2.THRESH_BINARY_INV,
)
 
# Resalta en rojo las regiones detectadas.
result = image.copy()
result[mask > 0] = (0, 0, 255)
 
# OpenCV utiliza BGR; Streamlit muestra imágenes en RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Imagen original")
    st.image(image_rgb, use_container_width=True)
 
    st.subheader("Imagen en escala de grises")
    st.image(gray, use_container_width=True)
 
with col2:
    st.subheader("Máscara de thresholding")
    st.image(mask, use_container_width=True)
 
    st.subheader("Posibles defectos detectados")
    st.image(result_rgb, use_container_width=True)
 
st.caption(
    "Las regiones detectadas son píxeles oscuros según el umbral. "
    "La máscara no confirma por sí sola que sean grietas."
)
