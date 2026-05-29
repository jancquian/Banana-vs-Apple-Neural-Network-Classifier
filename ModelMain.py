import numpy as np
import random

from utilities import (
    load_image_paths,
    train,
    convert,
    matriz_inicializing,
    inicializar_pesos,
    inicializar_bias,
    predecir
)

# DATOS
manzana_resizing_path = "datasets/entrenamiento/manzana_escalada/"
banana_resizing_path = "datasets/entrenamiento/banana_escalada/"
manzana_prueba_path = "datasets/prueba/manzana/"
banana_prueba_path = "datasets/prueba/banana/"

route_dataset = []

# CONFIGURACIÓN
numero_capas = 4
tasa_aprendizaje = 0.01

arquitectura_capas = [32*32, 64, 32, 2] # 1024, 64, 32, 2

epocas = 250

errores = []
salidas = []
pesos = []
bias = []

def main():

    global route_dataset
    global errores
    global salidas
    global pesos
    global bias

    # DATASET

    route_dataset = load_image_paths(route_dataset, manzana_resizing_path, 'manzana')
    route_dataset = load_image_paths(route_dataset, banana_resizing_path, 'banana')

    # INIT RED

    errores = matriz_inicializing(arquitectura_capas, errores)
    salidas = matriz_inicializing(arquitectura_capas, salidas)

    pesos = inicializar_pesos(arquitectura_capas, pesos)
    bias = inicializar_bias(arquitectura_capas, bias)

    # TRAIN

    for epoca in range(epocas):

        random.shuffle(route_dataset)

        for ruta, clase in route_dataset:

            if clase == 'manzana':
                objetivo = np.array([[1.0], [0.0]])
            else:
                objetivo = np.array([[0.0], [1.0]])

            entrada = convert(ruta)

            errores, salidas, pesos, bias = train(
                errores,
                salidas,
                pesos,
                bias,
                objetivo,
                tasa_aprendizaje,
                entrada,
                numero_capas
            )

    # GUARDAR MODELO

    np.save("modelo.npy", np.array([pesos, bias], dtype=object))

    # PRUEBAS

    test_manzana = load_image_paths([], manzana_prueba_path, 'manzana')
    test_banana = load_image_paths([], banana_prueba_path, 'banana')

    ruta_manzana, _ = test_manzana[0]
    ruta_banana, _ = test_banana[0]

    entrada_manzana = convert(ruta_manzana)
    entrada_banana = convert(ruta_banana)

    clases = ["manzana", "banana"]

    clase1, salida1 = predecir(entrada_manzana, pesos, bias, numero_capas)
    clase2, salida2 = predecir(entrada_banana, pesos, bias, numero_capas)

    print("\n====RESULTADO PRUEBA DE MANZANA")
    print(salida1)
    print("Predicción:", clases[clase1])

    print("\n====RESULTADO PRUEBA DE BANANA")
    print("Salida: ", salida2)
    print("Predicción:", clases[clase2])


if __name__ == "__main__":
    main()