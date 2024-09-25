# noisy-mnist-classification

This project focuses on classifying noisy, transformed images of handwritten digits from the MNIST dataset using Deep Learning techniques. The images have undergone transformations to make the task more challenging. The goal is to compare the performance of two different neural network architectures: a fully connected (dense) neural network and a Convolutional Neural Network (CNN).

## Dataset
The dataset used in this project is the MNIST dataset, which consists of 70,000 images of handwritten digits from 0 to 9. Each image is a 28x28 grayscale image. The dataset is split into 60,000 training images and 10,000 test images.

The images in the dataset have been transformed with random transformations such as rotation, translation, zooming, and varying brightness to make the task more challenging. The transformations were applied using the `ImageDataGenerator` class from Keras.


## Models
Two different neural network architectures were used for the classification task:
1. Fully Connected (Dense) Neural Network
2. Convolutional Neural Network (CNN)

The fully connected neural network consists of 3 dense layers with ReLU activation functions and dropout layers for regularization. The CNN consists of 2 convolutional layers followed by max pooling layers, and then 2 dense layers with ReLU activation functions and dropout layers.

## Results
The models were trained on the noisy MNIST dataset and evaluated on the test set. The performance of the models was compared based on the accuracy metric. The CNN model outperformed the fully connected neural network, achieving an accuracy of around 97% on the test set, while the fully connected neural network achieved an accuracy of around 82%.


## Usage

### Clone the repository

```bash
git clone https://github.com/Essmaw/noisy-mnist-classification.git
cd PyClustal
```

### Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create a Conda environment

```bash
conda env create -f environment.yml
```

### Activate the Conda environment

```bash
conda activate mnist_env
```

### Run the Jupyter notebook

```bash
jupyter notebook
```

Open the `noisy_mnist_classification.ipynb` notebook to run the analyze by yourself.


