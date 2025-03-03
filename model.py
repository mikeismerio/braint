import tensorflow as tf
from tensorflow.keras.models import load_model

# Ruta del archivo del modelo
model_path = "2025-19-02_VGG_model.h5"

# Cargar el modelo
model = load_model(model_path)

# Resumen del modelo
print("Resumen del modelo:")
model.summary()

# Obtener información de los pesos
print("\nCapas y pesos del modelo:")
for layer in model.layers:
    print(f"Capa: {layer.name}, Tipo: {layer.__class__.__name__}")
    for weight in layer.weights:
        print(f"  - {weight.name}, Forma: {weight.shape}")

# Revisar la configuración del modelo
config = model.get_config()
print("\nConfiguración del modelo:")
print(config)
