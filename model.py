import os
import sys

model_path = "2025-19-02_VGG_model.h5"

if os.path.exists(model_path):
    print(f"El archivo {model_path} existe y tiene un tamaño de {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    sys.stdout.flush()
else:
    print("El archivo no existe o está corrupto.")
    sys.stdout.flush()
