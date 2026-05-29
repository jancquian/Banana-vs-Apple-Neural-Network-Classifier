import numpy as np
from PIL import Image
import scipy.special
import os

# DATASET

def load_image_paths(route_dataset, entrada, clase):

    for archivo in os.listdir(entrada):
        ruta = os.path.join(entrada, archivo)
        route_dataset.append((ruta, clase))

    return route_dataset

# INPUT (32x32)

def convert(nombre):

    img = Image.open(nombre).resize((32, 32))
    img = np.array(img.convert('L')) / 255.0

    return img.reshape(1024, 1)


def sigmoid(x):
    return scipy.special.expit(x)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


# Inicialización de matriz

def matriz_inicializing(arquitectura_capas, matriz):
    for n in arquitectura_capas:
        matriz.append(np.zeros((n, 1)))

    return matriz


def inicializar_pesos(arquitectura_capas, matriz_pesos):
    for i in range(len(arquitectura_capas) - 1):
        limit = np.sqrt(6 / (arquitectura_capas[i] + arquitectura_capas[i+1]))

        w = np.random.uniform(
            -limit,
            limit,
            (arquitectura_capas[i+1], arquitectura_capas[i])
        )

        matriz_pesos.append(w)

    return matriz_pesos


def inicializar_bias(arquitectura_capas, bias):
    for i in range(1, len(arquitectura_capas)):
        b = np.random.uniform(
            -0.1,
            0.1,
            (arquitectura_capas[i], 1)
        )

        bias.append(b)

    return bias

#  Entrenamiento

def train(errores, salidas, pesos, bias, objetivo, lr, entrada, capas):

    # forward
    salidas[0] = entrada

    for i in range(1, capas):
        z = np.dot(pesos[i-1], salidas[i-1]) + bias[i-1]
        if i == capas - 1:
            salidas[i] = sigmoid(z)
        else:
            salidas[i] = relu(z)

    # output error
    errores[capas-1] = (objetivo - salidas[capas-1]) * salidas[capas-1] * (1 - salidas[capas-1])

    # backprop
    for i in range(capas-2, -1, -1):
        errores[i] = np.dot(pesos[i].T, errores[i+1])

        if i > 0:
            errores[i] *= relu_deriv(salidas[i])

    # update
    for i in range(capas-1):
        pesos[i] += lr * np.dot(errores[i+1], salidas[i].T)
        bias[i] += lr * errores[i+1]

    return errores, salidas, pesos, bias

# Predicción

def predecir(entrada, pesos, bias, capas):

    a = entrada
    for i in range(capas - 1):
        z = np.dot(pesos[i], a) + bias[i]
        if i == capas - 2:
            a = sigmoid(z)
        else:
            a = relu(z)

    return np.argmax(a), a