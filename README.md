# 🧠 Noisy MNIST Classification

This project focuses on classifying noisy, transformed images of handwritten digits from the MNIST dataset using Deep Learning techniques. The images have undergone transformations to make the task more challenging. The goal is to compare the performance of two different neural network architectures: a Fully Connected (Dense) Neural Network and a Convolutional Neural Network (CNN).

## 📊 Dataset
The dataset used in this project is the **MNIST dataset**, consisting of 70,000 images of handwritten digits (0 to 9). Each image is a **28x28 grayscale** image. The dataset is split into:
- **60,000 training images**
- **10,000 test images**

The images have been transformed with random alterations such as:
- 🔄 **Rotation**
- ↕️ **Translation**
- 🔍 **Zoom**
- ☀️ **Varying brightness**

These transformations were applied using the `ImageDataGenerator` class from Keras to make the task more challenging.

## 🛠️ Models
Two different neural network architectures were used for the classification task:

### 1. ⚙️ Fully Connected (Dense) Neural Network
This model consists of:
- 3 Dense layers with **ReLU** activation
- **Dropout layers** for regularization

### 2. 🖼️ Convolutional Neural Network (CNN)
This model consists of:
- 2 Convolutional layers followed by Max Pooling layers
- 2 Dense layers with **ReLU** activation
- **Dropout layers** for regularization

## 📈 Results
The models were trained on the noisy MNIST dataset and evaluated on the test set. Here's a comparison of the two architectures based on accuracy:

- **CNN** achieved an accuracy of around **97%** on the test set 🏆
- **Fully Connected Neural Network** achieved an accuracy of around **82%**

Clearly, the CNN outperformed the fully connected model due to its ability to capture spatial features in images.

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/Essmaw/noisy-mnist-classification.git
cd noisy-mnist-classification
```

### Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### 2. Create a Conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate the Conda environment

```bash
conda activate mnist_env
```

### 4. Run the Jupyter notebook

```bash
jupyter notebook
```
Open the `noisy_mnist_classification.ipynb` notebook to run the analyze by yourself.

### 5. 🌐 Try it on Google Colab

You can also try running the notebook on Google Colab without needing to set up the environment locally. Click the badge below to get started:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tvZtvmC6X1vtMZ4DrU6Do81TQ-g9TWo8?hl=fr)
