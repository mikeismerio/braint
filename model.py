import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import tempfile

# Cargar el modelo entrenado
model = load_model("2025-19-02_VGG_model.h5", compile=False)

def preprocess_image(image):
    """
    Preprocesar una imagen de MRI para VGG-16.
    """
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Redimensionar a 224x224
    image = np.expand_dims(image, axis=0)  # Añadir dimensión de batch
    image = preprocess_input(image)  # Aplicar preprocesamiento de VGG-16
    return image

# Interfaz en Streamlit
st.title("Detección de Tumores en Imágenes de MRI")
st.write("Sube una imagen de resonancia magnética para detectar si hay un tumor.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Guardar en un archivo temporal para procesamiento con OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name
    
    # Preprocesar la imagen
    processed_image = preprocess_image(image)
    
    # Realizar la predicción
    prediction = model.predict(processed_image)
    label = "🔴 Tumor Detectado" if prediction[0][0] > 0.5 else "🟢 No se detectó tumor"
    
    # Mostrar resultado
    st.subheader("Resultado de la Predicción:")
    st.write(f"**{label}**")
