import os
import shutil
from PIL import Image

entrada_manzana = "datasets/manzana_original/"
entrada_banana = "datasets/banana_original/"

salida_manzana = "datasets/entrenamiento/manzana_escalada/"
salida_banana = "datasets/entrenamiento/banana_escalada/"

def reset_folder(path: str):

    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Carpeta eliminada: {path}")

    os.makedirs(path)
    print(f"Carpeta creada: {path}")

def img_resizing(entrada: str, salida: str):

    for archivo in os.listdir(entrada):

        ruta = os.path.join(entrada, archivo)

        try:
            img = Image.open(ruta)
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            img.save(os.path.join(salida, archivo))
            print(f"OK: {archivo}")

        except Exception as e:
            print(f"Error en {archivo}: {e}")

def main():
    # limpiar carpetas
    reset_folder(salida_manzana)
    reset_folder(salida_banana)

    # redimensionar dataset
    img_resizing(entrada_manzana, salida_manzana)
    img_resizing(entrada_banana, salida_banana)

if __name__ == "__main__":
    main()