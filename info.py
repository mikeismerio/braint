import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os
import io

# =================== CONFIGURACIÓN DE LA PÁGINA ===================
st.set_page_config(layout="wide", page_title="📊 Análisis del Modelo VGG")
st.title("📊 Análisis del Modelo VGG - 2025-19-02")

# =================== CARGAR MODELO ===================
st.write("📥 **Cargando modelo...**")
model_path = "2025-19-02_VGG_model.h5"

if not os.path.exists(model_path):
    st.error(f"❌ No se encontró el archivo `{model_path}`. Asegúrate de subirlo.")
    st.stop()

try:
    model = load_model(model_path, compile=False)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

# =================== INFORMACIÓN DEL MODELO ===================
st.header("📌 Información General del Modelo")
st.write(f"**Nombre del modelo:** `{model.name}`")
st.write(f"**Número de capas:** `{len(model.layers)}`")

# Obtener número de parámetros
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
total_params = trainable_params + non_trainable_params

st.write(f"**Parámetros entrenables:** `{trainable_params:,}`")
st.write(f"**Parámetros no entrenables:** `{non_trainable_params:,}`")
st.write(f"**Total de parámetros:** `{total_params:,}`")


# =================== DETALLES DE CAPAS ===================
st.header("🔍 Detalle de las Capas del Modelo")

layers_data = []
for layer in model.layers:
    try:
        output_shape = str(layer.output_shape)
    except AttributeError:
        output_shape = "No disponible"

    layers_data.append({
        "Nombre de la Capa": layer.name,
        "Tipo": layer.__class__.__name__,
        "Salida": output_shape,
        "Parámetros": layer.count_params()
    })

df_layers = pd.DataFrame(layers_data)
st.dataframe(df_layers)

# =================== RESUMEN DEL MODELO ===================
st.header("📜 Resumen del Modelo")
with io.StringIO() as buf:
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    summary_text = buf.getvalue()
st.text(summary_text)

st.success("🚀 Aplicación lista para analizar modelos en Streamlit!")
