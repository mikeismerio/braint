import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# Cargar el modelo entrenado
model = load_model("2025-19-02_VGG_model.h5")

def preprocess_image(image_path):
    """
    Cargar y preprocesar una imagen de MRI para VGG-16.
    """
    image = cv2.imread(image_path)  # Leer la imagen
    image = cv2.resize(image, (224, 224))  # Redimensionar a 224x224
    image = np.expand_dims(image, axis=0)  # Añadir dimensión de batch
    image = preprocess_input(image)  # Aplicar preprocesamiento de VGG-16
    return image

def predict_image(image_path):
    """
    Predecir si una imagen de MRI contiene un tumor.
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    
    # Convertir la predicción a una etiqueta
    label = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    
    # Mostrar la imagen con la predicción
    plt.imshow(cv2.imread(image_path))
    plt.title(label)
    plt.axis("off")
    plt.show()

    print(f"Prediction: {label}")

# Ejemplo de uso
image_path = "/kaggle/input/braintumormri/test_mri.JPG"  # Reemplazar con la ruta de la imagen
predict_image(image_path)
