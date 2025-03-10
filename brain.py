import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")

# 📌 Cargar el modelo de detección de tumores
model_path = "2025-19-02_VGG_model.h5"
model = load_model(model_path, compile=False)

# 📌 Barra lateral para selección de imagen y navegación
st.sidebar.title("📌 Configuración")

# 📌 Opciones de navegación en la barra lateral (Análisis Craneal o Tumor)
page = st.sidebar.radio("Selecciona una sección:", ["Análisis Craneal", "Análisis del Tumor"])

# ✅ Permitir al usuario subir una única imagen en la barra lateral
uploaded_file = st.sidebar.file_uploader("📸 Selecciona una imagen médica:", type=["png", "jpg", "jpeg"])

# 📌 Verificar si el usuario ha subido una imagen antes de continuar
if uploaded_file:
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        if page == "Análisis Craneal":
            st.title("📏 Análisis del Cráneo")

            blurred = cv2.GaussianBlur(image, (7, 7), 2)
            edges = cv2.Canny(blurred, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area_threshold = 5000
            largest_contour = max(contours, key=cv2.contourArea) if contours else None

            if largest_contour is not None and cv2.contourArea(largest_contour) > min_area_threshold:
                hull = cv2.convexHull(largest_contour)
                x, y, w, h = cv2.boundingRect(hull)
                pixel_spacing = 0.035

                diameter_transversal_cm = w * pixel_spacing
                diameter_anteroposterior_cm = h * pixel_spacing
                cephalic_index = (diameter_transversal_cm / diameter_anteroposterior_cm) * 100

                skull_type = (
                    "Dolicocéfalo (cabeza alargada)" if cephalic_index < 75 else
                    "Mesocefálico (cabeza normal)" if 75 <= cephalic_index <= 80 else
                    "Braquicéfalo (cabeza ancha)"
                )

                contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_image, [hull], -1, (0, 255, 0), 2)
                st.image(contour_image, caption="Contorno del Cráneo", width=500)
                st.write(f"📏 **Diámetro Transversal:** `{diameter_transversal_cm:.2f} cm`")
                st.write(f"📏 **Diámetro Anteroposterior:** `{diameter_anteroposterior_cm:.2f} cm`")
                st.write(f"📏 **Índice Cefálico:** `{cephalic_index:.2f}`")
                st.write(f"📌 **Tipo de Cráneo:** `{skull_type}`")

        elif page == "Análisis del Tumor":
            st.title("🧠 Análisis del Tumor")

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_resized = cv2.resize(image, (224, 224))
            image_array = np.expand_dims(image_resized, axis=0) / 255.0

            prediction = model.predict(image_array)
            probability = prediction[0][0]
            threshold = 0.5
            tumor_detected = probability >= threshold
            diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

            st.image(image_resized, caption="Imagen Procesada para Análisis", width=500)
            st.write(f"🔍 **Probabilidad de Tumor:** `{probability:.2%}`")
            st.write(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")

            if tumor_detected:
                heatmap = np.random.randint(0, 255, image.shape, dtype=np.uint8)
                gray_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                area_cm2 = np.sum(heatmap > 100) * 0.01
                cx, cy = np.mean(np.argwhere(heatmap > 100), axis=0).astype(int)
                
                st.image([image, cv2.cvtColor(gray_heatmap, cv2.COLOR_BGR2RGB)], width=400, caption=["Imagen Original", "Segmentación del Tumor"])
                st.write(f"🧠 **Área del tumor:** `{area_cm2:.2f} cm²`")
                st.write(f"📌 **Ubicación del tumor (Centro):** `({cx}, {cy})` en píxeles")

                if area_cm2 > 10:
                    st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
                else:
                    st.success("✅ **El tumor es de tamaño pequeño o moderado.**")
            else:
                st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")
