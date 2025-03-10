import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")
st.sidebar.title("📌 Configuración")

# Cargar tu modelo entrenado (cualquiera que sea)
model_path = "2025-19-02_VGG_model.h5"
model = load_model(model_path, compile=False)

# Opciones de navegación
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
        pixel_spacing = 0.035  # Ajusta según la resolución real de la imagen

        diameter_transversal_cm = w * pixel_spacing
        diameter_anteroposterior_cm = h * pixel_spacing
        cephalic_index = (diameter_transversal_cm / diameter_anteroposterior_cm) * 100

        skull_type = (
            "Dolicocéfalo (cabeza alargada)" if cephalic_index < 75 else
            "Mesocefálico (cabeza normal)" if 75 <= cephalic_index <= 80 else
            "Braquicéfalo (cabeza ancha)"
        )

        # Convertir a BGR para dibujar contornos
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

def watershed_tumor_segmentation(image_bgr):
    """
    Segmenta el tumor usando Watershed:
    1. Convertir a gris + CLAHE para mejorar contraste.
    2. Umbral Otsu para primer plano.
    3. Morfología para refinar.
    4. Distance Transform para marcar foreground.
    5. Watershed.
    Devuelve la máscara final (región del tumor).
    """
    # 1) Convertir a gris + CLAHE (ecualización adaptativa)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)

    # 2) Suavizado y umbral Otsu
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Morfología para eliminar ruido y cerrar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Fondo seguro (sure background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4) Distance Transform para foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Ajusta el factor 0.7 según tu imagen
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5) Watershed
    # Etiquetas para cada región
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed sobre la imagen original BGR
    markers = cv2.watershed(image_bgr, markers)

    # Cualquier píxel marcado como -1 es frontera
    # Creamos máscara final: marcamos como tumor todo lo que no sea fondo
    # OJO: Dependiendo de la anatomía, es posible que requieras filtrar áreas muy pequeñas o muy grandes
    mask = np.zeros_like(gray, dtype=np.uint8)
    # Asigna 255 a todo lo que sea mayor que 1 (1 es el label del fondo)
    mask[markers > 1] = 255

    # (Opcional) Podrías filtrar áreas muy pequeñas
    # conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in conts:
    #     if cv2.contourArea(c) < 500:  # por ejemplo
    #         cv2.drawContours(mask, [c], -1, 0, -1)

    return mask

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    # Convertir la imagen a color si viene en 1 canal
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Inferencia con tu modelo de clasificación
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    tumor_detected = probability >= 0.5
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

    # >>> Segmentar el tumor con Watershed <<<
    tumor_mask = watershed_tumor_segmentation(image_bgr)

    # Buscar contornos en la máscara final
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear un heatmap de la máscara
    heatmap = cv2.applyColorMap(tumor_mask, cv2.COLORMAP_JET)
    heatmap_with_contour = heatmap.copy()

    # Dibujar contornos en verde
    if contours:
        cv2.drawContours(heatmap_with_contour, contours, -1, (0, 255, 0), 2)
    else:
        st.warning("No se detectó contorno en la segmentación del tumor.")

    # Calcular área y centroide
    tumor_area_px = sum(cv2.contourArea(c) for c in contours)
    pixel_spacing = 0.035  # Ajusta según tu resolución
    area_cm2 = tumor_area_px * (pixel_spacing ** 2)

    # Tomamos el contorno más grande para el centroide
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] else 0
    else:
        cx, cy = 0, 0

    # Superponer el heatmap al 30% sobre la imagen original
    overlay = cv2.addWeighted(image_rgb, 0.7, cv2.cvtColor(heatmap_with_contour, cv2.COLOR_BGR2RGB), 0.3, 0)

    st.image([image_rgb, overlay], width=400, caption=["Imagen Original", "Watershed con contorno del Tumor"])
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
