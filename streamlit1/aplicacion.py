import streamlit as st
from PIL import Image

def main():
  """ Función que se encargará de la ejecución principal"""
  st.title("Clasificación de Imágenes con un Modelo Preentrenado")

  st.markdown("Sube una imagen para clasificar")

  # widget que permite subir archivos
  uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)

if __name__ == "__main__":
  main()
  
