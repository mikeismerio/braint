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
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Opciones de la sidebar
page = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Análisis del Tumor", "Reporte PDF"])

if page == "Inicio":
    try:
        st.image("portada.jpg", width=800)
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
        image_resized = cv2.resize(image_rgb, (100, 100)) / 255.0  # Imagen más pequeña
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
    
    # Segmentación del tumor con umbralización
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=thresh)
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image_rgb)
    ax[0].set_title("Imagen Original")
    ax[0].axis('off')
    ax[1].imshow(segmented_image)
    ax[1].set_title("Área Segmentada del Tumor")
    ax[1].axis('off')
    st.pyplot(fig)

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
