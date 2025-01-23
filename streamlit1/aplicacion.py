import streamlit as st

def main():
  """ Función que se encargará de la ejecución principal"""
  st.title("Clasificación de Imágenes con un Modelo Preentrenado")

  st.markdown("Sube una imagen para clasificar")

  # widget que permite subir archivos
  upload_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
  

if __name__ == "__main__":
  main()
  
