import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys

model_path = "2025-19-02_VGG_model.h5"

if os.path.exists(model_path):
    print(f"El archivo {model_path} existe y tiene un tamaño de {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    sys.stdout.flush()
    
    # Cargar el modelo sin compilar
    model = load_model(model_path, compile=False)
    print("Modelo cargado correctamente.")
    sys.stdout.flush()

    # Mostrar el resumen del modelo
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    model.summary()
    sys.stdout = old_stdout

    print(mystdout.getvalue())  # Forzar impresión del resumen
    sys.stdout.flush()
else:
    print("El archivo no existe o está corrupto.")
    sys.stdout.flush()
