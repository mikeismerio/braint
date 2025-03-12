import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(
    layout="wide",
    page_title="🧠 Detección y Segmentación de Tumores Cerebrales",
    initial_sidebar_state="collapsed"
)

# Definir nombres de clases según el entrenamiento
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitario"]

# Opciones de la sidebar
page = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Análisis Craneal", "Análisis del Tumor", "Reporte PDF"])

if page == "Inicio":
    try:
        st.image("portada.jpg", width=800)
        st.markdown("<h2 style='text-align: center;'>Bienvenido a la aplicación de Diagnóstico</h2>", unsafe_allow_html=True)
    except Exception:
        st.warning("No se encontró la imagen de portada.")

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
    
    # Convertir la imagen a RGB si es en escala de grises
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar y normalizar la imagen
    image_resized = cv2.resize(image_rgb, (150, 150)) / 255.0
    image_array = np.expand_dims(image_resized, axis=0)
    
    st.write("🔍 **Analizando la imagen...**")
    prediction = model.predict(image_array)
    predicted_class_idx = np.argmax(pred, axis=1)[0]
    predicted_class = tumor_classes[predicted_class_idx]
    probability = pred[0][predicted_class_idx]
    
    st.subheader(f"📌 **Diagnóstico del Modelo:** `{predicted_class}`")
    st.write(f"📊 **Probabilidad de Clasificación:** `{probability:.2%}`")
    
    if predicted_class != "No Tumor":
        st.warning("⚠️ **El modelo ha detectado un posible tumor. Se recomienda un análisis más detallado.**")
    else:
        st.success("✅ **El modelo no detectó un tumor en la imagen.**")
    
    # Mostrar la imagen con el resultado
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.set_title(f"Predicción: {predicted_class}")
    ax.axis('off')
    st.pyplot(fig)

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
