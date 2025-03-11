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

# Subida de imagen (solo para análisis)
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
# Función para Análisis Craneal
def analyze_cranio(image):
    st.title("📏 Análisis del Cráneo")
    # Convertir a escala de grises si es necesario
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
        pixel_spacing = 0.035  # Ajusta según la resolución real

        diameter_transversal_cm = w * pixel_spacing
        diameter_anteroposterior_cm = h * pixel_spacing
        cephalic_index = (diameter_transversal_cm / diameter_anteroposterior_cm) * 100

        skull_type = (
            "Dolicocéfalo (cabeza alargada)" if cephalic_index < 75 else
            "Mesocefálico (cabeza normal)" if 75 <= cephalic_index <= 80 else
            "Braquicéfalo (cabeza ancha)"
        )

        # Dibujar contornos sobre la imagen
        contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, [hull], -1, (0, 255, 0), 2)
        cv2.line(contour_image, (x, y + h // 2), (x + w, y + h // 2), (255, 0, 0), 2)
        cv2.line(contour_image, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 0), 2)

        st.image(contour_image, caption="Contorno del Cráneo", width=500)
        st.write(f"📏 **Diámetro Transversal:** `{diameter_transversal_cm:.2f} cm`")
        st.write(f"📏 **Diámetro Anteroposterior:** `{diameter_anteroposterior_cm:.2f} cm`")
        st.write(f"📏 **Índice Cefálico:** `{cephalic_index:.2f}`")
        st.write(f"📌 **Tipo de Cráneo:** `{skull_type}`")

        # Guardar resultados en session_state para el reporte PDF
        st.session_state.cranio_metrics = {
            "Diámetro Transversal": f"{diameter_transversal_cm:.2f} cm",
            "Diámetro Anteroposterior": f"{diameter_anteroposterior_cm:.2f} cm",
            "Índice Cefálico": f"{cephalic_index:.2f}",
            "Tipo de Cráneo": skull_type
        }
        st.session_state.cranio_image = contour_image
    else:
        st.error("No se encontró un contorno significativo del cráneo.")

# ---------------------------------------------------------------------------
# Función para Análisis del Tumor
def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    # Asegurar imagen en 3 canales para visualización
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    
    # Preprocesamiento para el modelo
    image_resized = cv2.resize(image, (224, 224))
    if len(image_resized.shape) == 2:
        image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb_resized = cv2.resize(image_rgb, (224, 224))
    image_array = np.expand_dims(image_rgb_resized, axis=0)
    
    st.write("🔍 **Analizando la imagen...**")
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    threshold = 0.7
    tumor_detected = probability >= threshold
    diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"
    
    st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
    st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")
    
    # Guardar datos básicos del tumor
    st.session_state.tumor_metrics = {
        "Diagnóstico": diagnosis,
        "Probabilidad": f"{probability:.2%}"
    }
    
    if tumor_detected:
        st.warning("⚠️ **El modelo ha detectado un posible tumor. Segmentando...**")
        pixel_spacing = 0.04  # Ajusta según la resolución
        
        if len(image.shape) == 3:
            gray_seg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_seg = image.copy()
            
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
            
            tumor_image = cv2.cvtColor(gray_seg, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 2)
            cv2.circle(tumor_image, (cx, cy), 5, (0, 0, 255), -1)
            
            mask = np.zeros_like(gray_seg, dtype=np.uint8)
            cv2.drawContours(mask, [tumor_contour], -1, 255, thickness=cv2.FILLED)
            tumor_region = cv2.bitwise_and(gray_seg, gray_seg, mask=mask)
            heatmap = cv2.applyColorMap(tumor_region, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(tumor_image, 0.6, heatmap, 0.4, 0)
            
            st.image([gray_seg, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)], width=400,
                     caption=["Imagen Original (Grayscale)", "Segmentación del Tumor"])
            st.write(f"🧠 **Área del Tumor:** `{area_cm2:.2f} cm²`")
            st.write(f"📌 **Ubicación del Tumor (Centro):** `({cx}, {cy})` en píxeles")
            
            st.session_state.tumor_metrics.update({
                "Área del Tumor": f"{area_cm2:.2f} cm²",
                "Ubicación (Centro)": f"({cx}, {cy})"
            })
            st.session_state.tumor_image = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            if area_cm2 > 10:
                st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
            else:
                st.success("✅ **El tumor es de tamaño pequeño o moderado.**")
        else:
            st.error("❌ No se detectaron contornos significativos para el tumor.")
    else:
        st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")

# ---------------------------------------------------------------------------
# Función auxiliar: coloca imagen a la izquierda y métricas a la derecha
def add_section_with_image_and_metrics(pdf, fill_color, title, image, metrics):
    # Encabezado de sección
    r, g, b = fill_color
    pdf.set_fill_color(r, g, b)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, title, ln=True, fill=True)
    pdf.ln(1)
    
    # Coordenadas y tamaño de la imagen
    start_y = pdf.get_y()
    x_image = 10
    image_width = 60
    image_height = 60  
    line_height = 6
    
    # Guardar imagen temporal y colocarla en PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, image)
        image_path = tmp_file.name
    pdf.image(image_path, x=x_image, y=start_y, w=image_width, h=image_height)
    os.remove(image_path)

    # Métricas a la derecha
    x_text = x_image + image_width + 5
    pdf.set_xy(x_text, start_y)
    pdf.set_font("Arial", "", 12)

    for key, value in metrics.items():
        pdf.set_x(x_text)  
        pdf.cell(80, line_height, f"{key}: {value}", ln=True, align='L')

    text_block_height = len(metrics) * line_height
    new_y = start_y + max(image_height, text_block_height) + 5
    pdf.set_y(new_y)
    pdf.ln(1)

# ---------------------------------------------------------------------------
# Función para generar el reporte PDF (con portada en la 1ra página)
def generate_pdf_report(patient_data):
    pdf = FPDF()
    
    # === PÁGINA DE PORTADA ===
    pdf.add_page()
    try:
        # Ajusta x, y y tamaño de acuerdo a tu portada
        pdf.image("portada.jpg", x=0, y=0, w=210)  # Ancho completo A4
    except Exception as e:
        # Si no se encuentra la imagen, simplemente sigue
        pass
    
    # Saltar a la siguiente página para el contenido
    pdf.add_page()
    
    # === COMIENZA EL REPORTE ===
    # Cabecera con fondo de color
    pdf.set_fill_color(50, 150, 250)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Reporte Médico", ln=True, align="C", fill=True)
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    
    # Datos del paciente
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Datos del Paciente", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 6, f"Nombre: {patient_data['nombre']}", ln=True)
    pdf.cell(0, 6, f"Edad: {patient_data['edad']}", ln=True)
    pdf.cell(0, 6, f"Sexo: {patient_data['sexo']}", ln=True)
    pdf.cell(0, 6, f"Fecha de estudio: {patient_data['fecha']}", ln=True)
    pdf.multi_cell(0, 6, f"Observaciones: {patient_data['observaciones']}")
    pdf.ln(3)
    
    # Medición del Cráneo
    if "cranio_metrics" in st.session_state:
        add_section_with_image_and_metrics(
            pdf,
            fill_color=(200, 220, 255),
            title="Medición del Cráneo",
            image=st.session_state.cranio_image,
            metrics=st.session_state.cranio_metrics
        )
    else:
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(0, 8, "Medición del Cráneo", ln=True, fill=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 6, "No se realizaron análisis del cráneo.", ln=True)
        pdf.ln(5)
    
    # Segmentación del Tumor
    if "tumor_metrics" in st.session_state:
        add_section_with_image_and_metrics(
            pdf,
            fill_color=(255, 200, 200),
            title="Segmentación del Tumor",
            image=st.session_state.tumor_image,
            metrics=st.session_state.tumor_metrics
        )
    else:
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(255, 200, 200)
        pdf.cell(0, 8, "Segmentación del Tumor", ln=True, fill=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 6, "No se realizaron análisis del tumor.", ln=True)
        pdf.ln(5)
    
    # Generar bytes del PDF
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

# ---------------------------------------------------------------------------
# Función para la página de Reporte PDF
def pdf_report():
    st.title("📝 Generación de Reporte PDF")
    st.write("Ingresa los datos del paciente para generar el reporte con los análisis realizados.")
    with st.form("pdf_form"):
        nombre = st.text_input("Nombre del Paciente")
        edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
        sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
        fecha = st.date_input("Fecha de estudio")
        observaciones = st.text_area("Observaciones")
        submit = st.form_submit_button("Generar Reporte")
    if submit:
        if not nombre:
            st.error("El nombre es obligatorio.")
            return
        patient_data = {
            "nombre": nombre,
            "edad": edad,
            "sexo": sexo,
            "fecha": fecha.strftime("%Y-%m-%d"),
            "observaciones": observaciones
        }
        pdf_bytes = generate_pdf_report(patient_data)
        st.success("Reporte PDF generado exitosamente.")
        st.download_button("Descargar Reporte PDF", pdf_bytes, file_name="reporte.pdf", mime="application/pdf")

# ---------------------------------------------------------------------------
# Procesamiento según la sección seleccionada
if page in ["Análisis Craneal", "Análisis del Tumor"]:
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
        st.info("Por favor, sube una imagen para comenzar el análisis.")
elif page == "Reporte PDF":
    pdf_report()
