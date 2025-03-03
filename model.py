import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os

# Título de la aplicación
st.title("Cargar y visualizar un modelo de TensorFlow")

# Subir archivo de modelo
uploaded_file = st.file_uploader("Selecciona un archivo de modelo (.h5)", type=["h5"])

if uploaded_file is not None:
    try:
        # Guardar el archivo subido en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Cargar el modelo desde el archivo temporal
        model = load_model(tmp_file_path)
        st.success("¡Modelo cargado correctamente!")

        # Mostrar resumen del modelo
        st.subheader("Resumen del modelo")
        st.text("A continuación se muestra la arquitectura del modelo:")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))  # Capturar el resumen
        st.text("\n".join(model_summary))  # Mostrar el resumen en Streamlit

        # Mostrar detalles de las capas
        st.subheader("Detalles de las capas")
        for layer in model.layers:
            st.write(f"**Nombre de la capa:** {layer.name}")
            st.write(f"**Tipo de la capa:** {layer.__class__.__name__}")
            st.write(f"**Forma de salida:** {layer.output_shape}")
            st.write(f"**Número de parámetros:** {layer.count_params()}")
            st.write("------")

        # Eliminar el archivo temporal después de usarlo
        os.remove(tmp_file_path)

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.warning("Por favor, sube un archivo de modelo (.h5).")
