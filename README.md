# Project Title

Enhancing Image Classification Performance using Deep Convolutional Neural Networks

## Project Description

This project focuses on improving the accuracy of image classification tasks by leveraging the power of deep learning and convolutional neural networks (CNNs). By employing advanced techniques and architectures, the project aims to achieve state-of-the-art performance in classifying complex visual data. The project explores various aspects of CNNs, such as network architectures, hyperparameter tuning, and data augmentation, to optimize the model's performance. Additionally, it investigates different evaluation metrics and visualization techniques to gain insights into the model's predictions. Through this project, the goal is to showcase the potential of deep learning for image classification tasks and provide valuable insights for further research and applications.

## Features

- Implement a k-nearest neighbor classifier with 1 and 3 nearest neighbors.
- Implement a centroid-based classifier.
- Read training and test data from a chosen database.
- Calculate and compare the performance of the classifiers.

## Dependencies and Programming Language

- **Programming Language:** Python

The project relies on the following libraries and frameworks:

- **TensorFlow** : A powerful deep learning library for building and training neural networks.
- **Keras** : A high-level neural networks API that runs on top of TensorFlow, simplifying the process of building and training models.
- **NumPy** : A fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices.
- **Matplotlib** : A plotting library for creating static, animated, and interactive visualizations in Python.
- **Scikit-learn** : A machine learning library that provides efficient tools for data preprocessing, model selection, and evaluation.


The project was developed using **Google Colab**, a cloud-based Jupyter notebook environment, as it provided a convenient platform for running the code and leveraging the computational resources required for deep learning tasks.

Ensure that you have these dependencies installed in your Python environment before running the project. You can use tools like pip or conda to install them:



# Classifier Performance Analysis

This project involves building and analyzing the performance of two classifiers: K Nearest Neighbors (KNN) and Nearest Centroid. The code compares the performance of these classifiers using the MNIST dataset, which consists of handwritten digit images.

## Dataset

The MNIST dataset is loaded using the `mnist.load_data()` function from the `keras.datasets` module. It is then transformed to reshape the images into a flattened format suitable for training and testing the classifiers. The training set is represented by `train_X` and `train_y`, while the test set is represented by `test_X` and `test_y`.

## K Nearest Neighbors (KNN) Classifier

The KNN classifier is evaluated with two different values for the number of neighbors (`k`): 1 and 3. For each `k`, the following steps are performed:

1. The KNN classifier is instantiated with `KNeighborsClassifier(k)` from the `sklearn.neighbors` module.
2. The training data (`train_X` and `train_y`) is fitted to the model using the `fit()` method.
3. The time taken to fit the data is measured using the `time()` function.
4. The testing data (`test_X`) is used to predict the labels using the `predict()` method.
5. The time taken to predict the labels is measured using the `time()` function.
6. The performance of the classifier is evaluated by displaying the classification report using the `classification_report()` function from the `sklearn.metrics` module.
7. The confusion matrix is visualized using the `ConfusionMatrixDisplay` class from `sklearn.metrics` and plotted using `matplotlib.pyplot`.

## Nearest Centroid Classifier

The Nearest Centroid classifier is evaluated using the following steps:

1. The Nearest Centroid classifier is instantiated with `NearestCentroid()` from the `sklearn.neighbors` module.
2. The training data (`train_X` and `train_y`) is fitted to the model using the `fit()` method.
3. The time taken to fit the data is measured using the `time()` function.
4. The testing data (`test_X`) is used to predict the labels using the `predict()` method.
5. The time taken to predict the labels is measured using the `time()` function.
6. The performance of the classifier is evaluated by displaying the classification report using the `classification_report()` function from the `sklearn.metrics` module.
7. The confusion matrix is visualized using the `ConfusionMatrixDisplay` class from `sklearn.metrics` and plotted using `matplotlib.pyplot`.

## Usage

Feel free to experiment with different values of `k` for the KNN classifier or explore other classifiers to further analyze their performance using the MNIST dataset.



