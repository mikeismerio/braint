import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")
st.sidebar.title("📌 Configuración")

# Cargar el modelo (¡sí, el cerebro digital ya está en marcha!)
model_path = "2025-19-02_VGG_model.h5"
model = load_model(model_path, compile=False)

# Opciones de navegación
page = st.sidebar.radio("Selecciona una sección:", ["Análisis Craneal", "Análisis del Tumor"])
uploaded_file = st.sidebar.file_uploader("📸 Selecciona una imagen médica:", type=["png", "jpg", "jpeg"])

def analyze_cranio(image):
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
        pixel_spacing = 0.035  # Ajusta según la resolución real de la imagen

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
        cv2.line(contour_image, (x, y + h // 2), (x + w, y + h // 2), (255, 0, 0), 2)
        cv2.line(contour_image, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 0), 2)

        st.image(contour_image, caption="Contorno del Cráneo", width=500)
        st.write(f"📏 **Diámetro Transversal:** `{diameter_transversal_cm:.2f} cm`")
        st.write(f"📏 **Diámetro Anteroposterior:** `{diameter_anteroposterior_cm:.2f} cm`")
        st.write(f"📏 **Índice Cefálico:** `{cephalic_index:.2f}`")
        st.write(f"📌 **Tipo de Cráneo:** `{skull_type}`")
    else:
        st.error("No se encontró un contorno significativo del cráneo. ¿Seguro que la imagen no es una selfie sin filtro?")

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    # Convertimos la imagen a color si es necesario
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

    # Preprocesamiento para el modelo: redimensionar y normalizar
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    tumor_detected = probability >= 0.5
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

    # Segmentación mejorada: umbralización de Otsu + operaciones morfológicas
    gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    ret, tumor_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Crear heatmap de la segmentación
    heatmap = cv2.applyColorMap(tumor_mask, cv2.COLORMAP_JET)

    # Encontrar y dibujar contornos sobre el heatmap
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heatmap_with_contour = heatmap.copy()
    if contours:
        cv2.drawContours(heatmap_with_contour, contours, -1, (0, 0, 255), 2)

    # Calcular área y centroide del tumor
    tumor_area_px = sum(cv2.contourArea(c) for c in contours)
    pixel_spacing = 0.035  # Ajusta según la imagen
    area_cm2 = tumor_area_px * (pixel_spacing ** 2)

    if contours:
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
    else:
        cx, cy = 0, 0

    # Superponer el heatmap con contornos a la imagen original
    overlay = cv2.addWeighted(image_rgb, 0.7, cv2.cvtColor(heatmap_with_contour, cv2.COLOR_BGR2RGB), 0.3, 0)

    st.image([image_rgb, overlay], width=400, caption=["Imagen Original", "Heatmap con contorno del Tumor"])
    st.write(f"🔍 **Probabilidad de Tumor:** `{probability:.2%}`")
    st.write(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
    st.write(f"🧠 **Área del Tumor:** `{area_cm2:.2f} cm²`")
    st.write(f"📌 **Ubicación del Tumor (Centro):** `({cx}, {cy})` en píxeles")

    if tumor_detected:
        if area_cm2 > 10:
            st.warning("⚠️ ¡El tumor está para volverse protagonista! Se recomienda un análisis más detallado.")
        else:
            st.success("✅ Tumor detectado, pero de tamaño razonable. Nada de pánico.")
    else:
        st.success("✅ El modelo no encontró tumor significativo. ¡Sigue disfrutando de tu día sin sobresaltos!")

# Procesamiento de la imagen subida
if uploaded_file:
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    if image is not None:
        if page == "Análisis Craneal":
            analyze_cranio(image)
        elif page == "Análisis del Tumor":
            analyze_tumor(image, model)
    else:
        st.error("Error al cargar la imagen. Verifica el formato y contenido (no todas las selfies son médicas).")
