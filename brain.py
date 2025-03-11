import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
from fpdf import FPDF
import io
import tempfile
import os

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Detección y Segmentación de Tumores Cerebrales")
st.sidebar.title("📌 Configuración")

# Selección de página
page = st.sidebar.radio("Selecciona una sección:", ["Análisis Craneal", "Análisis del Tumor", "Reporte PDF"])

# Subida de imagen para análisis (no se usa en Reporte PDF)
if page != "Reporte PDF":
    uploaded_file = st.sidebar.file_uploader("📸 Subir imagen médica (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# =================== CARGAR MODELO ===================
st.sidebar.write("📥 Cargando modelo 2025-19-02_VGG_model.h5...")
model_path = "2025-19-02_VGG_model.h5"
try:
    model = load_model(model_path, compile=False)
    st.sidebar.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

# ---------------------------------------------------------------------------
# Función para Análisis Craneal (estructura anterior)
def analyze_cranio(image):
    st.title("📏 Análisis del Cráneo")
    # Convertir a escala de grises si es una imagen a color
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

        # Convertir el gris a BGR para dibujar contornos
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

# ---------------------------------------------------------------------------
# Función para Análisis del Tumor: incluye predicción con CNN y segmentación
def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    # Asegurarse de trabajar con una imagen en 3 canales para visualización
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    
    # Preprocesamiento para el modelo: redimensionar a 224x224 y convertir a RGB
    image_resized = cv2.resize(image, (224, 224))
    if len(image_resized.shape) == 2:
        image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_rgb_resized, axis=0)
    
    # Realizar la predicción con la CNN
    st.write("🔍 **Analizando la imagen...**")
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    threshold = 0.7
    tumor_detected = probability >= threshold
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"
    
    st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
    st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")
    
    if tumor_detected:
        st.warning("⚠️ **El modelo ha detectado un posible tumor. Segmentando...**")
        pixel_spacing = 0.04  # cm/píxel (ajusta según la resolución)
        
        # Para segmentar, primero convertir la imagen a escala de grises (si no lo está)
        if len(image.shape) == 3:
            gray_seg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_seg = image.copy()
            
        # Segmentación: suavizado y umbral fijo
        blurred = cv2.GaussianBlur(gray_seg, (7, 7), 2)
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
            
            # Dibujar contornos y centroide sobre la imagen original (convertida a BGR)
            tumor_image = cv2.cvtColor(gray_seg, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 2)
            cv2.circle(tumor_image, (cx, cy), 5, (0, 0, 255), -1)
            
            # Crear una máscara del tumor y generar heatmap
            mask = np.zeros_like(gray_seg, dtype=np.uint8)
            cv2.drawContours(mask, [tumor_contour], -1, 255, thickness=cv2.FILLED)
            tumor_region = cv2.bitwise_and(gray_seg, gray_seg, mask=mask)
            heatmap = cv2.applyColorMap(tumor_region, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(tumor_image, 0.6, heatmap, 0.4, 0)
            
            st.image([gray_seg, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)], width=400,
                     caption=["Imagen Original (Grayscale)", "Segmentación del Tumor"])
            st.write(f"🧠 **Área del Tumor:** `{area_cm2:.2f} cm²`")
            st.write(f"📌 **Ubicación del Tumor (Centro):** `({cx}, {cy})` en píxeles")
            
            if area_cm2 > 10:
                st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
            else:
                st.success("✅ **El tumor es de tamaño pequeño o moderado.**")
        else:
            st.error("❌ No se detectaron contornos significativos para el tumor.")
    else:
        st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")

# ---------------------------------------------------------------------------
# Función para generar el reporte PDF
def generate_pdf_report(patient_data, images):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Reporte Médico", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Nombre: {patient_data['nombre']}", ln=True)
    pdf.cell(0, 10, f"Edad: {patient_data['edad']}", ln=True)
    pdf.cell(0, 10, f"Sexo: {patient_data['sexo']}", ln=True)
    pdf.cell(0, 10, f"Fecha de estudio: {patient_data['fecha']}", ln=True)
    pdf.multi_cell(0, 10, f"Observaciones: {patient_data['observaciones']}")
    pdf.ln(10)

    if images:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Imágenes del Estudio", ln=True)
        for img in images:
            # Guardar la imagen en un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(img.getvalue())
                tmp_file.flush()
                image_path = tmp_file.name
            pdf.add_page()
            try:
                pdf.image(image_path, x=10, y=10, w=pdf.w - 20)
            except Exception as e:
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Error al cargar imagen: {str(e)}", ln=True)
            os.remove(image_path)
    else:
        pdf.cell(0, 10, "No se han subido imágenes para este reporte.", ln=True)

    # Devolver el PDF en bytes
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

# Función para la página de Reporte PDF
def pdf_report():
    st.title("📝 Generación de Reporte PDF")
    st.write("Ingresa los datos del paciente y sube las imágenes que deseas incluir en el reporte.")

    with st.form("pdf_form"):
        nombre = st.text_input("Nombre del Paciente")
        edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
        sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
        fecha = st.date_input("Fecha de estudio")
        observaciones = st.text_area("Observaciones")
        images = st.file_uploader("Sube las imágenes para el reporte", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submit = st.form_submit_button("Generar Reporte")

    if submit:
        if not nombre:
            st.error("El nombre es obligatorio. ¡No te saltes los datos del paciente!")
            return
        patient_data = {
            "nombre": nombre,
            "edad": edad,
            "sexo": sexo,
            "fecha": fecha.strftime("%Y-%m-%d"),
            "observaciones": observaciones
        }
        pdf_bytes = generate_pdf_report(patient_data, images)
        st.success("Reporte PDF generado exitosamente. ¡Ahora sí, a presumir de reporte profesional!")
        st.download_button("Descargar Reporte PDF", pdf_bytes, file_name="reporte.pdf", mime="application/pdf")

# ---------------------------------------------------------------------------
# Procesamiento de la imagen subida para Análisis Craneal y Tumor
if page == "Análisis Craneal" or page == "Análisis del Tumor":
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is not None:
            if page == "Análisis Craneal":
                analyze_cranio(image)
            elif page == "Análisis del Tumor":
                analyze_tumor(image, model)
        else:
            st.error("Error al cargar la imagen. Verifica el formato y contenido.")
    else:
        st.info("Por favor, sube una imagen desde la barra lateral para comenzar el análisis.")
elif page == "Reporte PDF":
    pdf_report()
