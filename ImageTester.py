import numpy as np
import os
from utilities import convert, predecir

numero_capas = 4

clases = ["manzana", "banana"]

carpeta_imagenes = "PruebasMezcladas/"

# CARGAR MODELO


modelo = np.load("modelo.npy", allow_pickle=True)

pesos = modelo[0]
bias = modelo[1]

# PREDICCIÓN

for archivo in os.listdir(carpeta_imagenes):

    if archivo.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

        ruta = os.path.join(carpeta_imagenes, archivo)

        entrada = convert(ruta)

        clase, salida = predecir(
            entrada,
            pesos,
            bias,
            numero_capas
        )

        print("\nImagen:", archivo)
        print("Salida red:")
        print(salida)
        print("Clase predicha:", clases[clase])