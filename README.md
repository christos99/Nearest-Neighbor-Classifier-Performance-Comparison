# Project Title

Nearest Neighbor Classifier Performance Comparison

## Project Description

This project involves developing a program in any desired programming language to compare the performance of the k-nearest neighbor classifier with 1 and 3 nearest neighbors against the performance of the centroid-based classifier. The program will read training and test data from a selected database and evaluate the accuracy of the classifiers.

## Features

- Implement a k-nearest neighbor classifier with 1 and 3 nearest neighbors.
- Implement a centroid-based classifier.
- Read training and test data from a chosen database.
- Calculate and compare the performance of the classifiers.

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

To run the code, ensure that you have the necessary dependencies installed. You can then execute the script and observe the performance analysis of the KNN and Nearest Centroid classifiers.

Feel free to experiment with different values of `k` for the KNN classifier or explore other classifiers to further analyze their performance using the MNIST dataset.



