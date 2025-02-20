import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys
import io

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="🧠 Detección de Tumores Cerebrales")

st.title("🧠 Detección de Tumores Cerebrales con CNN")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")

# =================== CARGAR MODELO ===================
st.write("📥 **Cargando modelo...**")
model_path = "2025-19-02_VGG_model.h5"

try:
    model = load_model(model_path, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

# =================== MOSTRAR RESUMEN DEL MODELO ===================
with st.expander("📜 Ver detalles del modelo"):
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    summary_str = buffer.getvalue()
    buffer.close()
    st.code(summary_str, language="text")

# =================== SUBIR UNA IMAGEN ===================
uploaded_file = st.file_uploader("📸 **Sube una imagen médica (JPG, PNG)**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer la imagen y convertirla en un array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Mostrar imagen original
        st.image(image, caption="Imagen original", width=400)

        # Preprocesar la imagen para el modelo
        image_resized = cv2.resize(image, (224, 224))  # Tamaño compatible con VGG16
        image_array = np.expand_dims(image_resized, axis=0)  # Agregar batch
        image_array = image_array / 255.0  # Normalización

        # =================== REALIZAR PREDICCIÓN ===================
        st.write("🔍 **Analizando la imagen...**")
        prediction = model.predict(image_array)
        
        # Verificar si el modelo tiene salida binaria o multiclase
        if prediction.shape[-1] == 1:
            probability = prediction[0][0]  # Para modelos binarios
        else:
            probability = np.max(prediction)  # Para modelos de clasificación multiclase

        # 🔧 Ajustar umbral dinámicamente si el modelo está sesgado
        threshold = st.slider("📊 Ajustar umbral de detección:", 0.1, 0.9, 0.5)
        tumor_detected = probability >= threshold
        diagnosis = "Tumor Detectado" if tumor_detected else "No se detectó Tumor"

        # Mostrar resultado de la predicción
        st.subheader(f"📌 **Diagnóstico del Modelo:** `{diagnosis}`")
        st.write(f"📊 **Probabilidad de Tumor:** `{probability:.2%}`")

        # 📊 Mostrar histograma de predicciones (para detectar sesgo)
        st.write("📊 **Distribución de Probabilidad de Tumor**")
        fig, ax = plt.subplots()
        ax.hist(prediction.flatten(), bins=10, color="blue", alpha=0.7)
        ax.axvline(threshold, color="red", linestyle="dashed", label=f"Umbral: {threshold:.2f}")
        ax.set_title("Distribución de Predicciones")
        ax.set_xlabel("Probabilidad de Tumor")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)

        # Mensajes de alerta
        if tumor_detected:
            st.warning("⚠️ **El modelo ha detectado un posible tumor. Se recomienda un análisis más detallado.**")
        else:
            st.success("✅ **El modelo no detectó un tumor significativo en la imagen.**")

        # 🔥 Mostrar Mapa de Activación (CAM) para entender qué detecta el modelo
        def get_heatmap(image_array, model, layer_name="vgg16"):
            """Genera un mapa de calor basado en las activaciones de una capa del modelo."""
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(image_array)
                loss = predictions[:, 0]
            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_output = conv_output.numpy()[0]
            pooled_grads = pooled_grads.numpy()
            for i in range(pooled_grads.shape[-1]):
                conv_output[:, :, i] *= pooled_grads[i]
            heatmap = np.mean(conv_output, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            return heatmap

        # Generar y visualizar el heatmap
        st.write("🔥 **Mapa de Activación del Modelo (CAM)**")
        heatmap = get_heatmap(image_array, model)
        plt.figure(figsize=(6, 6))
        plt.imshow(image_resized)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")
        st.pyplot(plt)
