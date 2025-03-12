import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
from fpdf import FPDF
import io
import tempfile
import os

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(
    layout="wide",
    page_title="🧠 Detección y Segmentación de Tumores Cerebrales",
    initial_sidebar_state="collapsed"  # Sidebar oculta al iniciar
)

# Opciones de la sidebar: se agregó "Inicio"
page = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Análisis Craneal", "Análisis del Tumor", "Reporte PDF"])

# Si estamos en la página de inicio, se muestra la portada en grande
if page == "Inicio":
    try:
        st.image("portada.jpg", width=800)
        st.markdown("<h2 style='text-align: center;'>Bienvenido a la aplicación de Diagnóstico</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.warning("No se encontró la imagen de portada.")

# Para secciones de análisis se habilita la carga de imagen y se carga el modelo
if page in ["Análisis Craneal", "Análisis del Tumor"]:
    uploaded_file = st.sidebar.file_uploader("📸 Subir imagen médica (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    st.sidebar.write("📥 Cargando modelo modelo4.keras...")
    model_path = "modelo4.keras"
    try:
        model = load_model(model_path, compile=False)
        st.sidebar.success("✅ Modelo cargado exitosamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

# ---------------------------------------------------------------------------
# Función para Análisis del Tumor

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    
    image_resized = cv2.resize(image, (224, 224))
    if len(image_resized.shape) == 2:
        image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_rgb_resized, axis=0)
    
    st.write("🔍 **Analizando la imagen...**")
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    threshold = 0.7
    tumor_detected = probability >= threshold
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"
    
    st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
    st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")
    
    if tumor_detected:
        st.warning("⚠️ **El modelo ha detectado un posible tumor. Se recomienda un análisis más detallado.**")
    else:
        st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")

# ---------------------------------------------------------------------------
# Procesamiento según la sección seleccionada
if page in ["Análisis Craneal", "Análisis del Tumor"]:
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is not None:
            if page == "Análisis del Tumor":
                analyze_tumor(image, model)
        else:
            st.error("Error al cargar la imagen. Verifica el formato y contenido.")
    else:
        st.info("Por favor, sube una imagen para comenzar el análisis.")
