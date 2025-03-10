import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")
st.sidebar.title("📌 Configuración")

# Cargar tu modelo entrenado (ejemplo: clasificación de tumor/no tumor)
model_path = "2025-19-02_VGG_model.h5"
model = load_model(model_path, compile=False)

page = st.sidebar.radio("Selecciona una sección:", ["Análisis Craneal", "Análisis del Tumor"])
uploaded_file = st.sidebar.file_uploader("📸 Selecciona una imagen médica:", type=["png", "jpg", "jpeg"])

def analyze_cranio(image):
    st.title("📏 Análisis del Cráneo")
    # Convertir a gris si es de 3 canales
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
        pixel_spacing = 0.035  # Ajusta según tu imagen

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
        st.error("No se encontró un contorno significativo del cráneo.")

def watershed_segmentation(image_bgr):
    """
    Segmenta la imagen usando Watershed:
    1) Convertir a gris + ecualización adaptativa (CLAHE) para realzar contraste.
    2) Umbral (Otsu) y operaciones morfológicas.
    3) Distance Transform para separar foreground del background.
    4) Aplicar Watershed.
    5) Devolver máscara binaria con las regiones > 1 (no fondo).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)

    # Umbral Otsu tras suavizado
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance Transform para encontrar foreground seguro
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Ajusta el factor 0.7 si la segmentación se come o deja partes
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores para Watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)

    # Crear máscara final: marcamos como 255 todo lo que sea > 1
    # (1 es el label del fondo)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    return mask

def keep_largest_connected_component(mask):
    """
    De la máscara binaria, nos quedamos solo con el componente conectado más grande.
    Esto evita que aparezcan múltiples zonas pequeñas que no sean tumor.
    """
    # Conectamos componentes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Label 0 es fondo. Empezamos a buscar desde label 1
    largest_label = 0
    largest_area = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label_id

    # Creamos una máscara nueva con solo el componente más grande
    final_mask = np.zeros_like(mask)
    if largest_label > 0:
        final_mask[labels == largest_label] = 255

    return final_mask

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Clasificación (probabilidad de tumor)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    tumor_detected = probability >= 0.5
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

    # 1) Watershed
    raw_mask = watershed_segmentation(image_bgr)

    # 2) Conectado más grande (para descartar ruiditos)
    final_mask = keep_largest_connected_component(raw_mask)

    # (Opcional) Operación morfológica para pulir bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contornos de la máscara final
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Heatmap
    heatmap = cv2.applyColorMap(final_mask, cv2.COLORMAP_JET)
    heatmap_with_contour = heatmap.copy()

    if contours:
        cv2.drawContours(heatmap_with_contour, contours, -1, (0, 255, 0), 2)
    else:
        st.warning("No se detectó contorno en la segmentación del tumor.")

    # Área y centroide
    tumor_area_px = sum(cv2.contourArea(c) for c in contours)
    pixel_spacing = 0.035  # Ajusta según tu resolución
    area_cm2 = tumor_area_px * (pixel_spacing ** 2)

    cx, cy = 0, 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

    # Superponer el heatmap al 30% sobre la imagen original
    overlay = cv2.addWeighted(image_rgb, 0.7, cv2.cvtColor(heatmap_with_contour, cv2.COLOR_BGR2RGB), 0.3, 0)

    st.image([image_rgb, overlay], width=400, caption=["Imagen Original", "Watershed + Mayor Componente"])
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
        st.error("Error al cargar la imagen. Verifica el formato y contenido.")
