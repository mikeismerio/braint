import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="Detección y Análisis de Imágenes Médicas")

# 📌 Barra lateral para seleccionar la página
st.sidebar.title("📌 Navegación")
page = st.sidebar.radio("Selecciona una sección:", ["Análisis del Tumor", "Análisis Craneal"])

# ✅ Permitir al usuario subir una imagen
uploaded_file = st.file_uploader("📸 Selecciona una imagen médica:", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # ✅ Leer la imagen en memoria
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # ✅ Verificar si la imagen se cargó correctamente
    if image is None:
        st.error("❌ No se pudo cargar la imagen.")
    else:
        st.success("✅ Imagen cargada correctamente.")

        # =================== PÁGINA 1: ANÁLISIS DEL TUMOR ===================
        if page == "Análisis del Tumor":
            st.title("🧠 Análisis del Tumor")

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
                cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

                tumor_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 1)
                cv2.circle(tumor_image, (cx, cy), 5, (0, 0, 255), -1)

                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.drawContours(mask, [tumor_contour], -1, 255, thickness=cv2.FILLED)
                tumor_region = cv2.bitwise_and(image, image, mask=mask)
                heatmap = cv2.applyColorMap(tumor_region, cv2.COLORMAP_JET)
                heatmap = cv2.addWeighted(tumor_image, 0.6, heatmap, 0.4, 0)

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(image, cmap="gray")
                axs[0].set_title("Imagen Original")
                axs[0].axis("off")
                axs[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                axs[1].set_title("Segmentación del Tumor con Heatmap")
                axs[1].axis("off")
                st.pyplot(fig)

                st.write(f"🧠 **Área del tumor:** `{area_cm2:.2f} cm²`")
                st.write(f"📌 **Ubicación:** `({cx}, {cy})` en píxeles")

                if area_cm2 > 10:
                    st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
                else:
                    st.success("✅ **El tumor es de tamaño pequeño o moderado.**")
            else:
                st.error("❌ No se detectaron tumores.")

        # =================== PÁGINA 2: ANÁLISIS CRANEAL ===================
        elif page == "Análisis Craneal":
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

                estimated_cranial_height = 13
                volume_cm3 = (4/3) * np.pi * (diameter_transversal_cm / 2) * (diameter_anteroposterior_cm / 2) * (estimated_cranial_height / 2)

                fig = plt.figure(figsize=(6, 6))
                plt.imshow(image, cmap="gray")
                plt.axis("off")
                plt.title("Imagen Procesada del Cráneo")
                st.pyplot(fig)

                st.write(f"📏 **Diámetro Transversal:** `{diameter_transversal_cm:.2f} cm`")
                st.write(f"📏 **Diámetro Anteroposterior:** `{diameter_anteroposterior_cm:.2f} cm`")
                st.write(f"📏 **Índice Cefálico:** `{cephalic_index:.2f}`")
                st.write(f"📌 **Tipo de Cráneo:** `{skull_type}`")
                st.write(f"🧠 **Volumen craneal estimado:** `{volume_cm3:.2f} cm³`")

                if not (1200 <= volume_cm3 <= 1700):
                    st.warning("⚠️ **El volumen craneal podría no ser correcto.**")
            else:
                st.error("❌ No se detectaron contornos del cráneo.")
