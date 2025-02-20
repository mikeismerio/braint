import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys
import io

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Detección de Tumores Cerebrales")

st.title("🧠 Detección de Tumores Cerebrales con CNN")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")

# =================== CARGAR MODELO ===================
st.write("📥 **Cargando modelo...**")
model_path = "2025-19-02_VGG_model.h5"

try:
    model = load_model(model_path, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

# =================== MOSTRAR RESUMEN DEL MODELO ===================
with st.expander("📜 Ver detalles del modelo"):
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    summary_str = buffer.getvalue()
    buffer.close()
    st.code(summary_str, language="text")

# =================== SUBIR UNA IMAGEN ===================
uploaded_file = st.file_uploader("📸 **Sube una imagen médica (JPG, PNG)**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer la imagen y convertirla en un array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Mostrar imagen original
        st.image(image, caption="Imagen original", width=400)

        # Preprocesar la imagen para el modelo
        image_resized = cv2.resize(image, (224, 224))  # Tamaño compatible con VGG16
        image_array = np.expand_dims(image_resized, axis=0)  # Agregar batch
        image_array = image_array / 255.0  # Normalización

        # =================== REALIZAR PREDICCIÓN ===================
        st.write("🔍 **Analizando la imagen...**")
        prediction = model.predict(image_array)
        
        if prediction.shape[-1] == 1:
            probability = prediction[0][0]  # Para modelos binarios
        else:
            probability = np.argmax(prediction)  # Para modelos de clasificación multiclase

        # Diagnóstico basado en umbral
        threshold = 0.5
        tumor_detected = probability >= threshold
        diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

        # Mostrar resultado de la predicción
        st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
        st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")

        # Mensajes de alerta
        if tumor_detected:
            st.warning("⚠️ **El modelo ha detectado un posible tumor. Se recomienda un análisis más detallado.**")
        else:
            st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")
