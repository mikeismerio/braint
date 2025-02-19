import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# =================== INTERFAZ EN STREAMLIT ===================
st.title("🧠 Detección y Segmentación de Tumores en Imágenes Médicas")

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

        # 📌 Definir el tamaño estimado del píxel (depende de la resolución de la imagen médica)
        pixel_spacing = 0.04  # cm/píxel

        # 📌 Aplicar suavizado Gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(image, (7, 7), 2)

        # 📌 Aplicar umbralización adaptativa para resaltar regiones anómalas (posibles tumores)
        _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

        # 📌 Encontrar contornos en la imagen segmentada
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 📌 Filtrar el contorno más grande, asumiendo que es el tumor
        min_area_threshold = 200  # Filtrar áreas demasiado pequeñas (ruido)
        tumor_contour = max(contours, key=cv2.contourArea) if contours else None

        if tumor_contour is not None and cv2.contourArea(tumor_contour) > min_area_threshold:
            # 📌 Calcular área del tumor en píxeles
            area_pixels = cv2.contourArea(tumor_contour)

            # 📌 Convertir el área a cm² usando el Pixel Spacing
            area_cm2 = area_pixels * (pixel_spacing ** 2)

            # 📌 Calcular el centro del tumor
            M = cv2.moments(tumor_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # Coordenada X del centro
                cy = int(M["m01"] / M["m00"])  # Coordenada Y del centro
            else:
                cx, cy = 0, 0

            # 📌 Dibujar la imagen con el contorno del tumor resaltado
            tumor_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(tumor_image, [tumor_contour], -1, (0, 255, 0), 1)  # Línea verde más delgada
            cv2.circle(tumor_image, (cx, cy), 5, (0, 0, 255), -1)  # Punto rojo en el centro

            # 📌 Crear una máscara basada en la intensidad del tumor
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.drawContours(mask, [tumor_contour], -1, 255, thickness=cv2.FILLED)

            # 📌 Aplicar el heatmap basado en la intensidad de la imagen original dentro del tumor
            tumor_region = cv2.bitwise_and(image, image, mask=mask)  # Extraer la región del tumor
            heatmap = cv2.applyColorMap(tumor_region, cv2.COLORMAP_JET)  # Aplicar mapa de calor
            heatmap = cv2.addWeighted(tumor_image, 0.6, heatmap, 0.4, 0)  # Fusionar con la imagen original

            # 📌 Mostrar ambas imágenes en Streamlit
            st.subheader("🖼️ Segmentación del Tumor y Heatmap")
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # 📌 Imagen sin procesar
            axs[0].imshow(image, cmap="gray")
            axs[0].set_title("Imagen Original")
            axs[0].axis("off")

            # 📌 Imagen con segmentación y heatmap basado en intensidad
            axs[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Segmentación del Tumor con Heatmap (Brillo)")
            axs[1].axis("off")

            st.pyplot(fig)

            # 📌 Mostrar resultados finales
            st.subheader("📊 Resultados del Análisis")
            st.write(f"🧠 **Área del tumor:** `{area_cm2:.2f} cm²`")
            st.write(f"📌 **Ubicación del tumor (Centro):** `({cx}, {cy})` en píxeles")

            # 📌 Validar si el área del tumor es preocupante
            if area_cm2 > 10:
                st.warning("⚠️ **El tumor es grande. Se recomienda un análisis más detallado.**")
            else:
                st.success("✅ **El tumor es de tamaño pequeño o moderado.**")

        else:
            st.error("❌ No se detectaron tumores en la imagen.")
