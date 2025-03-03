import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Cargar el modelo sin compilar para evitar errores con el optimizador
MODEL_PATH = "2025-19-02_VGG_model.h5"
try:
    model = load_model(MODEL_PATH, compile=False)  # Desactiva la compilación
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Configurar la interfaz de usuario
st.title("🧠 Detección de Tumores Cerebrales con VGG")
st.write("Sube una imagen de resonancia magnética para predecir si hay un tumor cerebral.")

# Función para preprocesar la imagen
def preprocess_image(image):
    img = image.resize((224, 224))  # Ajustar al tamaño de entrada del modelo
    img_array = np.array(img) / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para el modelo
    return img_array

# Subir imagen
tumor_classes = ['No Tumor', 'Tumor']
uploaded_file = st.file_uploader("📥 Sube una imagen de MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Imagen cargada", use_column_width=True)
    
    # Procesar imagen y hacer predicción
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Mostrar resultado
    st.subheader("🩺 Diagnóstico:")
    st.write(f"Predicción: **{tumor_classes[predicted_class]}**")
    st.write(f"Confianza: **{confidence:.2f}%**")
