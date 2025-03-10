import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")
st.sidebar.title("📌 Configuración")

# Cargar el modelo (tu red neuronal entrenada)
model_path = "2025-19-02_VGG_model.h5"
model = load_model(model_path, compile=False)

# Opciones de navegación
page = st.sidebar.radio("Selecciona una sección:", ["Análisis Craneal", "Análisis del Tumor"])
uploaded_file = st.sidebar.file_uploader("📸 Selecciona una imagen médica:", type=["png", "jpg", "jpeg"])

def analyze_cranio(image):
    st.title("📏 Análisis del Cráneo")
    # Verificamos canales de la imagen
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    blurred = cv2.GaussianBlur(gray_image, (7, 7), 2)
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

        contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
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

def segment_tumor_largest_bright_region(image_color):
    """
    Segmenta la región más brillante (o la más oscura si invertimos)
    en la imagen, asumiendo que el tumor es la mayor masa destacada.
    Devuelve la máscara final con la región del tumor.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Ecualización de histograma para resaltar regiones brillantes
    gray_eq = cv2.equalizeHist(gray)

    # Suavizado
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Umbral con Otsu (THRESH_BINARY)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos para ver si el "fondo" es lo más grande
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    for c in conts:
        area = cv2.contourArea(c)
        if area > largest_area:
            largest_area = area

    # Si la región más grande es mayor a la mitad de la imagen, invertimos
    img_area = gray.shape[0] * gray.shape[1]
    if largest_area > (img_area / 2):
        # Probablemente el fondo se tomó como "blanco"
        mask = cv2.bitwise_not(mask)

    # Operaciones morfológicas para limpiar agujeros/ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Reevaluar contornos para dibujar solo el más grande
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return np.zeros_like(mask)

    # Seleccionar el contorno más grande
    largest_contour = max(conts, key=cv2.contourArea)

    # Crear una nueva máscara solo con el contorno más grande
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)

    return final_mask

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    # Aseguramos que la imagen tenga 3 canales para visualización
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

    # Preprocesamiento para el modelo (redimensionar y normalizar)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    tumor_detected = probability >= 0.5
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

    # >>> Nueva función: segmentar la región más brillante (o invertida) <<<
    tumor_mask = segment_tumor_largest_bright_region(image_color)

    # Hallar contornos en la máscara final
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear heatmap de la máscara
    heatmap = cv2.applyColorMap(tumor_mask, cv2.COLORMAP_JET)
    heatmap_with_contour = heatmap.copy()

    # Dibujar el contorno (en verde) sobre el heatmap
    if contours:
        cv2.drawContours(heatmap_with_contour, contours, -1, (0, 255, 0), 2)
    else:
        st.warning("No se detectó contorno en la segmentación del tumor.")

    # Calcular área y centroide
    tumor_area_px = sum(cv2.contourArea(c) for c in contours)
    pixel_spacing = 0.035  # Ajusta según la resolución real de la imagen
    area_cm2 = tumor_area_px * (pixel_spacing ** 2)

    if contours:
        # Tomamos el contorno más grande para centroide
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
    else:
        cx, cy = 0, 0

    # Superponer heatmap al 30% sobre la imagen original
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
