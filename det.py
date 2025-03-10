import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import io
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Detección y Segmentación de Tumores")

st.title("🧠 Detección y Segmentación de Tumores Cerebrales")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")

# =================== CARGAR MODELO ===================
st.write("📥 **Cargando modelo 2025-19-02_VGG_model.h5...**")
model_path = "2025-19-02_VGG_model.h5"

try:
    model = load_model(model_path, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

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
        prediction = model.predict(image_array)
        probability = prediction[0][0]
        threshold = 0.7
        tumor_detected = probability >= threshold
        diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

        
        # Mostrar resultados de la CNN
        st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
        st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")

       


        if tumor_detected:
            st.warning("⚠️ **El modelo ha detectado un posible tumor. Segmentando...**")
            pixel_spacing = 0.04  # cm/píxel
            blurred = cv2.GaussianBlur(image, (7, 7), 2)
            _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area_threshold = 200
            tumor_contour = max(contours, key=cv2.contourArea) if contours else None

            if tumor_contour is not None and cv2.contourArea(tumor_contour) > min_area_threshold:
                area_pixels = cv2.contourArea(tumor_contour)
                area_cm2 = area_pixels * (pixel_spacing ** 2)

                M = cv2.moments(tumor_contour)
                cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

                tumor_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 2)
                cv2.circle(tumor_image, (cx, cy), 5, (0, 0, 255), -1)

                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.drawContours(mask, [tumor_contour], -1, 255, thickness=cv2.FILLED)
                tumor_region = cv2.bitwise_and(image, image, mask=mask)
                heatmap = cv2.applyColorMap(tumor_region, cv2.COLORMAP_JET)
                heatmap = cv2.addWeighted(tumor_image, 0.6, heatmap, 0.4, 0)


                
                # 📌 Mostrar segmentación
                st.image([image, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)], width=400)


                
                # 📌 Mostrar métricas del tumor
                st.write(f"🧠 **Área del tumor:** `{area_cm2:.2f} cm²`")
                st.write(f"📌 **Ubicación del tumor (Centro):** `({cx}, {cy})` en píxeles")


                
                # 📌 Mostrar resultados finales
                if area_cm2 > 10:
                    st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
                else:
                    st.success("✅ **El tumor es de tamaño pequeño o moderado.**")
            else:
                st.error("❌ No se detectaron tumores en la imagen.")
        else:
            st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")
