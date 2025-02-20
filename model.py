import streamlit as st
from tensorflow.keras.models import load_model
import sys
import io

st.title("🧠 Detección de Tumores Cerebrales")
st.write(f"📌 **Versión de Python en Streamlit Cloud:** `{sys.version}`")

# 📌 Ruta del modelo
model_path = "2025-19-02_VGG_model.h5"  # Asegúrate de que el archivo está en la misma carpeta

# 📌 Cargar el modelo
st.write("📥 **Cargando modelo...**")
try:
    model = load_model(model_path)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")

# 📌 Mostrar el resumen del modelo en Streamlit
if "model" in locals():
    st.subheader("📜 Resumen del Modelo")
    
    # Capturar el resumen en un buffer de texto
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    summary_str = buffer.getvalue()
    buffer.close()

    # Mostrarlo en Streamlit con formato de código
    st.code(summary_str, language="text")
