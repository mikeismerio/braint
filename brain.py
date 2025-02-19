import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")

# 📌 Barra lateral para selección de imagen y navegación
st.sidebar.title("📌 Configuración")

# 📌 Opciones de navegación en la barra lateral (Primero Cráneo, luego Tumor)
page = st.sidebar.radio("Selecciona una sección:", ["Análisis Craneal", "Análisis del Tumor"])

# ✅ Permitir al usuario subir una única imagen en la barra lateral
uploaded_file = st.sidebar.file_uploader("📸 Selecciona una imagen médica:", type=["png", "jpg", "jpeg"])

# 📌 Verificar si el usuario ha subido una imagen antes de continuar
if uploaded_file:
    # ✅ Leer la imagen en memoria
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # =================== PÁGINA 1: ANÁLISIS CRANEAL ===================
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

                # 📌 Dibujar contornos y líneas azules en la imagen procesada
                contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_image, [hull], -1, (0, 255, 0), 2)  # Verde para el contorno
                cv2.line(contour_image, (x, y + h // 2), (x + w, y + h // 2), (255, 0, 0), 2)  # Línea horizontal
                cv2.line(contour_image, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 0), 2)  # Línea vertical

                # 📌 Mostrar la imagen procesada
                st.image(contour_image, caption="Contorno del Cráneo", width=600)
                st.write(f"📏 **Diámetro Transversal:** `{diameter_transversal_cm:.2f} cm`")
                st.write(f"📏 **Diámetro Anteroposterior:** `{diameter_anteroposterior_cm:.2f} cm`")
                st.write(f"📏 **Índice Cefálico:** `{cephalic_index:.2f}`")
                st.write(f"📌 **Tipo de Cráneo:** `{skull_type}`")

        # =================== PÁGINA 2: ANÁLISIS DEL TUMOR ===================
        elif page == "Análisis del Tumor":
            st.title("🧠 Análisis del Tumor")

            # 📌 Aplicar suavizado Gaussiano y detección de tumor
            pixel_spacing = 0.035  # cm/píxel
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

                tumor_contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(tumor_contour_image, [tumor_contour], -1, (0, 255, 0), 2)
                cv2.circle(tumor_contour_image, (cx, cy), 5, (0, 0, 255), -1)

                st.image(tumor_contour_image, caption="Detección de Tumor", width=600)
                st.write(f"🧠 **Área del tumor:** `{area_cm2:.2f} cm²`")
                st.write(f"📌 **Ubicación del tumor (Centro):** `({cx}, {cy})` en píxeles")

                if area_cm2 > 10:
                    st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
                else:
                    st.success("✅ **El tumor es de tamaño pequeño o moderado.**")

            else:
                st.error("❌ No se detectaron tumores en la imagen.")
