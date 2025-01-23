import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

def load_model():
    """Carga el modelo preentrenado desde un archivo comprimido."""
    filename = "streamlit1/model_trained.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    """Preprocesa la imagen cargada para ser compatible con el modelo."""
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def get_fashion_mnist_label(class_index):
    """Convierte el índice de clase a su etiqueta de Fashion MNIST."""
    labels = [
        "Camiseta/Top", "Pantalón", "Suéter", "Vestido", "Abrigo", 
        "Sandalia", "Camisa", "Zapatilla deportiva", "Bolso", "Botines"
    ]
    return labels[class_index] if 0 <= class_index < len(labels) else "Clase desconocida"

def main():
    """Función principal para la ejecución de la aplicación."""
    # Título y descripción
    st.title("Clasificación de Imágenes de Fashion MNIST")
    st.markdown(
        """
        **Sube una imagen** para que sea clasificada por un modelo preentrenado de Fashion MNIST.
        El modelo predice la categoría de la prenda dentro de las siguientes opciones:
        - Camiseta/Top
        - Pantalón
        - Suéter
        - Vestido
        - Abrigo
        - Sandalia
        - Camisa
        - Zapatilla deportiva
        - Bolso
        - Botines
        """
    )

    # Área para subir imágenes
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

        # Procesar la imagen
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)

        # Mostrar la imagen preprocesada
        st.image(preprocessed_image[0], caption="Imagen preprocesada (28x28 px, escala de grises)", use_column_width=True, clamp=True)

        # Botón para clasificar
        if st.button("Clasificar Imagen"):
            st.markdown("## Resultado de la Clasificación")
            model = load_model()
            
            # Realizar la predicción
            prediction = model.predict(preprocessed_image.reshape(1, -1))
            predicted_class = np.argmax(prediction)  # Obtener la clase con mayor probabilidad

            # Mostrar la predicción
            label = get_fashion_mnist_label(predicted_class)
            st.success(f"La imagen fue clasificada como: **{label}** (Clase {predicted_class})")

if __name__ == "__main__":
    # Ajustar el diseño de la aplicación
    st.set_page_config(page_title="Clasificación Fashion MNIST", page_icon="👕", layout="centered")
    main()
