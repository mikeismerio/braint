import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import io
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Clasificación de Tumores Cerebrales")

st.title("🧠 Clasificación de Tumores Cerebrales")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")

# =================== CARGAR MODELO ===================
st.write("📥 **Cargando modelo brain-tumor-detection-acc-96-4-cnn.h5...**")
model_path = "brain-tumor-detection-acc-96-4-cnn.h5"

try:
    model = load_model(model_path, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

# =================== CLASES DEL MODELO ===================
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# =================== SUBIR UNA IMAGEN ===================
uploaded_file = st.file_uploader("📸 **Sube una imagen médica (JPG, PNG)**", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Leer la imagen y convertirla en array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:

        # Mostrar imagen original
        st.image(image, caption="Imagen original", width=400)

        # 🔹 Preprocesamiento para el modelo
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        image_array = np.expand_dims(image_rgb, axis=0)

        # =================== REALIZAR PREDICCIÓN ===================
        st.write("🔍 **Analizando la imagen...**")
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_idx]
        probability = predictions[0][predicted_class_idx]

        # Mostrar resultados de la CNN
        st.subheader(f"📌 **Diagnóstico del Modelo:** `{predicted_class}`")
        st.write(f"📊 **Confianza del Modelo:** `{probability:.2%}`")

        # Mostrar distribución de probabilidades
        fig, ax = plt.subplots()
        ax.bar(CLASSES, predictions[0], color=['blue', 'orange', 'green', 'red'])
        ax.set_ylabel("Probabilidad")
        ax.set_title("Distribución de Predicciones")
        st.pyplot(fig)

        # Mensaje final basado en la predicción
        if predicted_class == "No Tumor":
            st.success("✅ **No se detectó presencia de tumor en la imagen.**")
        else:
            st.warning(f"⚠️ **Posible detección de un `{predicted_class}`. Se recomienda análisis clínico detallado.**")
