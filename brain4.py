import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(
    layout="wide",
    page_title="🧠 Detección y Segmentación de Tumores Cerebrales",
    initial_sidebar_state="collapsed"
)

# Definir nombres de clases según el entrenamiento
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitario"]

# Sidebar para cargar la imagen y el modelo
page = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Análisis del Tumor"])

if page == "Inicio":
    try:
        st.image("portada.jpg", width=800)
        st.markdown("<h2 style='text-align: center;'>Bienvenido a la aplicación de Diagnóstico</h2>", unsafe_allow_html=True)
    except Exception:
        st.warning("No se encontró la imagen de portada.")

if page == "Análisis del Tumor":
    uploaded_file = st.sidebar.file_uploader("📸 Subir imagen médica (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    st.sidebar.write("📥 Cargando modelo modelo4.keras...")
    model_path = "modelo4.keras"
    try:
        model = load_model(model_path, compile=False)
        st.sidebar.success("✅ Modelo cargado exitosamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

# ---------------------------------------------------------------------------
# Función para Análisis del Tumor

def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    
    if image is None or image.size == 0:
        st.error("Error: La imagen no pudo ser procesada correctamente.")
        return
    
    # Convertir a RGB si la imagen está en escala de grises
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar la imagen al tamaño que espera el modelo
    image_resized = cv2.resize(image_rgb, (224, 224)) / 255.0
    image_array = np.expand_dims(image_resized, axis=0)

    st.write("🔍 **Analizando la imagen...**")
    try:
        prediction = model.predict(image_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = tumor_classes[predicted_class_idx]
        probability = prediction[0][predicted_class_idx]
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
        return
    
    st.subheader(f"📌 **Diagnóstico del Modelo:** `{predicted_class}`")
    st.write(f"📊 **Probabilidad de Clasificación:** `{probability:.2%}`")
    
    if predicted_class != "No Tumor":
        st.warning("⚠️ **El modelo ha detectado un posible tumor. Segmentando el área...**")
        
        # Convertir a escala de grises para segmentación
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Aplicar desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(gray_image, (7, 7), 2)
        
        # Umbralización adaptativa para segmentación
        _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

        # Encontrar contornos del tumor
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Seleccionar el contorno más grande (posible tumor)
            tumor_contour = max(contours, key=cv2.contourArea)
            
            # Dibujar contorno en la imagen original
            tumor_image = image_rgb.copy()
            cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 2)
            
            # Crear máscara de la región del tumor
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            cv2.drawContours(mask, [tumor_contour], -1, 255, thickness=cv2.FILLED)
            segmented_tumor = cv2.bitwise_and(gray_image, gray_image, mask=mask)

            # Aplicar mapa de calor
            heatmap = cv2.applyColorMap(segmented_tumor, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

            # Mostrar imágenes en paralelo
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Imagen Original", width=300)
            with col2:
                st.image(heatmap, caption="Segmentación del Tumor", width=300)

        else:
            st.error("❌ No se detectaron contornos significativos para el tumor.")
    else:
        st.success("✅ **El modelo no detectó un tumor en la imagen.**")

# ---------------------------------------------------------------------------
# Procesamiento según la sección seleccionada
if page == "Análisis del Tumor":
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if image is not None and image.size > 0:
            analyze_tumor(image, model)
        else:
            st.error("Error al cargar la imagen. Verifica el formato y contenido.")
    else:
        st.info("Por favor, sube una imagen para comenzar el análisis.")
