import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn

def load_model():
  filename = "streamlit1/model_trained.pkl.gz"
  with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
  return model
  

def preprocess_image(image):
  image = image.convert('L') # convertir a escala de grises
  image = image.resize((28,28)) 
  image_array = img_to_array(image) / 255.0
  image_array = np.expand_dims(image_array, axis=0)
  return image_array

def main():
  """ Función que se encargará de la ejecución principal"""
  st.title("Clasificación de Imágenes con un Modelo Preentrenado")

  st.markdown("Sube una imagen para clasificar")

  # widget que permite subir archivos
  uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file) # gaurdamos la imagen

    st.image(image, caption="imagen subida")
    
    preprocessed_image = preprocess_image(image)

    st.image(preprocessed_image[0], caption="imagen preprocesada")

    if st.button("Clasificar imagen"):
      st.markdown("Imagen clasificada")
      model = load_model()

      prediction = model.predict(preprocessed_image.reshape(1,-1))

      st.markdown(f"La imagen fue clasificada como: {prediction}")

  
if __name__ == "__main__":
  main()
