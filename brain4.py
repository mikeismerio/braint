import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(
    layout="wide",
    page_title="🧠 Detección y Segmentación de Tumores Cerebrales",
    initial_sidebar_state="collapsed"
)

# Definir nombres de clases según el entrenamiento
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitario"]

# Tamaño de imagen esperado por el modelo (ajústalo si es diferente)
IMAGE_SIZE = (80, 80)  # Se ajusta a un tamaño más pequeño para optimización

# Opciones de la sidebar
page = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Análisis del Tumor"])

if page == "Inicio":
    try:
        st.image("portada.jpg", width=400)
        st.markdown("<h2 style='text-align: center;'>Bienvenido a la aplicación de Diagnóstico</h2>", unsafe_allow_html=True)
    except Exception:
        st.warning("No se encontró la imagen de portada.")

if page == "Análisis del Tumor":
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
# Función para Análisis del Tumor con segmentación

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    
    if image is None or image.size == 0:
        st.error("Error: La imagen no pudo ser procesada correctamente.")
        return
    
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        image_resized = cv2.resize(image_rgb, IMAGE_SIZE) / 255.0  # Ajustar al tamaño esperado
        image_array = np.expand_dims(image_resized, axis=0)
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return
    
    st.write("🔍 **Analizando la imagen...**")
    try:
        prediction = model.predict(image_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = tumor_classes[predicted_class_idx]
        probability = prediction[0][predicted_class_idx]
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
        return
    
    st.subheader(f"📌 **Diagnóstico del Modelo:** `{predicted_class}`")
    st.write(f"📊 **Probabilidad de Clasificación:** `{probability:.2%}`")
    
    if predicted_class != "No Tumor":
        st.warning("⚠️ **El modelo ha detectado un posible tumor. Se recomienda un análisis más detallado.**")
    else:
        st.success("✅ **El modelo no detectó un tumor en la imagen.**")
    
    # Segmentación del tumor basada en la detección de bordes
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segmented_image = image_rgb.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.resize(image_rgb, (120, 120)), caption="Imagen Original", use_column_width=True)
    with col2:
        st.image(cv2.resize(segmented_image, (120, 120)), caption="Área Segmentada del Tumor", use_column_width=True)

# ---------------------------------------------------------------------------
# Procesamiento según la sección seleccionada
if page == "Análisis del Tumor":
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if image is not None and image.size > 0:
            analyze_tumor(image, model)
        else:
            st.error("Error al cargar la imagen. Verifica el formato y contenido.")
    else:
        st.info("Por favor, sube una imagen para comenzar el análisis.")
