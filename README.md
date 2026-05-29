# Banana-vs-Apple-Neural-Network-Classifier
A simple image classification project built from scratch using pure NumPy and Python. This project trains a basic multilayer neural network capable of distinguishing between apples and bananas using resized image datasets.

# Features

- Image preprocessing pipeline
- Dataset cleaning and resizing
- Fully connected neural network implemented manually
- Forward propagation
- Backpropagation
- Model persistence using `.npy`
- Prediction system for new images
- No TensorFlow or PyTorch required

---

## Project Structure

```text
project/
│
├── ImgResizer.py
├── ModelMain.py
├── ImageTester.py
├── utilities.py
├── modelo.npy
│
├── datasets/
│   ├── manzana_original/
│   ├── banana_original/
│   │
│   ├── entrenamiento/
│   │   ├── manzana_escalada/
│   │   └── banana_escalada/
│   │
│   └── prueba/
│       ├── manzana/
│       └── banana/
│
└── PruebasMezcladas/
```

---

# How It Works

The project consists of three main stages:

## 1. Image Preprocessing (`ImgResizer.py`)

This script:

- Removes old processed datasets
- Resizes all images to `32x32`
- Saves processed images into training folders

### Input

```text
datasets/manzana_original/
datasets/banana_original/
```

### Output

```text
datasets/entrenamiento/manzana_escalada/
datasets/entrenamiento/banana_escalada/
```

---

## 2. Neural Network Training (`ModelMain.py`)

This script:

- Loads the processed dataset
- Initializes the neural network
- Trains using backpropagation
- Saves the trained model
- Tests predictions with sample images

### Network Architecture

```text
1024 → 64 → 32 → 2
```

Where:

- `1024` = 32x32 image pixels
- `64` = hidden layer
- `32` = hidden layer
- `2` = output classes

### Output Classes

```text
[1, 0] = apple
[0, 1] = banana
```

### Saved Model

```text
modelo.npy
```

---

## 3. Prediction (`ImageTester.py`)

This script:

- Loads the trained model
- Reads images from a folder
- Predicts whether each image is an apple or banana

### Prediction Folder

```text
PruebasMezcladas/
```

---

# Requirements

Install dependencies:

```bash
pip install pillow numpy
```

---

# Running the Project

## Step 1 — Resize Dataset

```bash
python ImgResizer.py
```

---

## Step 2 — Train the Neural Network

```bash
python ModelMain.py
```

---

## Step 3 — Test Predictions

```bash
python ImageTester.py
```

---

# Example Output

```text
Image: banana1.jpg
Output:
[[0.02]
 [0.97]]

Predicted class: banana
```

---

# Technologies Used

- Python
- NumPy
- Pillow (PIL)

---

# Educational Purpose

This project was developed as an educational implementation of a neural network without using machine learning frameworks such as TensorFlow or PyTorch.

The objective is to understand:

- Neural network fundamentals
- Matrix operations
- Backpropagation
- Gradient descent
- Image preprocessing
- Model persistence

---

# Possible Improvements

Future enhancements may include:

- Convolutional Neural Networks (CNNs)
- OpenCV preprocessing
- Data augmentation
- Softmax activation
- Cross-entropy loss
- GPU acceleration
- Real-time image detection

---

# License

This project is open-source and available under the MIT License.
