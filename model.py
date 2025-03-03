import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

# Título de la aplicación
st.title("Cargar y visualizar un modelo de TensorFlow")

# Subir archivo de modelo
uploaded_file = st.file_uploader("Selecciona un archivo de modelo (.h5)", type=["h5"])

if uploaded_file is not None:
    try:
        # Convertir el archivo subido a un objeto BytesIO
        model_bytes = BytesIO(uploaded_file.read())

        # Cargar el modelo desde el archivo subido
        model = load_model(model_bytes)
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

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.warning("Por favor, sube un archivo de modelo (.h5).")
