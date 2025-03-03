import tensorflow as tf
from tensorflow.keras.models import load_model

# Ruta del archivo del modelo
model_path = "2025-19-02_VGG_model.h5"

# Cargar el modelo sin compilar
model = load_model(model_path, compile=False)

# Resumen del modelo
print("Resumen del modelo:")
model.summary()
