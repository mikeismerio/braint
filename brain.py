def analyze_tumor(image, model):
    st.title("🧠 Análisis del Tumor")
    # Aseguramos que la imagen tenga 3 canales
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

    # Segmentación: suavizado y prueba de umbralización normal y, si es necesario, invertida
    gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Primer intento: THRESH_BINARY
    ret, tumor_mask = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Si no se detecta nada, probamos con THRESH_BINARY_INV
    if not contours:
        ret, tumor_mask = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mejorar la máscara con operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Crear heatmap a partir de la máscara segmentada
    heatmap = cv2.applyColorMap(tumor_mask, cv2.COLORMAP_JET)

    # Dibujar el contorno sobre el heatmap
    heatmap_with_contour = heatmap.copy()
    if contours:
        cv2.drawContours(heatmap_with_contour, contours, -1, (0, 0, 255), 2)
    else:
        st.warning("No se detectó contorno en la segmentación del tumor.")

    # Calcular área y centroide del tumor
    tumor_area_px = sum(cv2.contourArea(c) for c in contours)
    pixel_spacing = 0.035  # Ajusta este valor según la imagen
    area_cm2 = tumor_area_px * (pixel_spacing ** 2)

    if contours:
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
    else:
        cx, cy = 0, 0

    # Superponer el heatmap con contorno a la imagen original
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
