import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Título de la aplicación
st.title("Cargar y visualizar un modelo de TensorFlow")

# Cargar el modelo
model_path = st.text_input("Introduce la ruta del modelo (.h5):", "2025-19-02_VGG_model.h5")

if st.button("Cargar modelo"):
    try:
        # Cargar el modelo
        model = load_model(model_path)
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
