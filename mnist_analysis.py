import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from time import time

def load_and_preprocess_data():
    """
    Load the MNIST dataset and preprocess it for model training and evaluation.
    Returns the scaled and reshaped training and test sets.
    """
    # Load MNIST dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Reshape and scale the data
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X.reshape(-1, 28*28))
    test_X = scaler.transform(test_X.reshape(-1, 28*28))

    return train_X, train_y, test_X, test_y

def train_and_evaluate_classifier(clf, train_X, train_y, test_X, test_y):
    """
    Train a classifier and evaluate its performance.
    Outputs the training time, prediction time, and classification metrics.
    """
    start_time = time()
    clf.fit(train_X, train_y)
    fit_time = time() - start_time

    start_time = time()
    predictions = clf.predict(test_X)
    predict_time = time() - start_time

    # Display the performance metrics
    print(f"Classifier: {clf.__class__.__name__} (Time to fit: {fit_time:.2f}s, Time to predict: {predict_time:.2f}s)")
    print(classification_report(test_y, predictions))
    ConfusionMatrixDisplay.from_predictions(test_y, predictions)
    plt.title(f"Confusion Matrix - {clf.__class__.__name__}")
    plt.show()

def main():
    """
    Main function to execute the training and evaluation process.
    Trains K-Nearest Neighbors and Nearest Centroid classifiers.
    """
    train_X, train_y, test_X, test_y = load_and_preprocess_data()

    # List of classifiers to train and evaluate
    classifiers = [
        KNeighborsClassifier(n_neighbors=k)
        for k in [1, 3]
    ] + [NearestCentroid()]

    for clf in classifiers:
        train_and_evaluate_classifier(clf, train_X, train_y, test_X, test_y)

if __name__ == "__main__":
    main()
