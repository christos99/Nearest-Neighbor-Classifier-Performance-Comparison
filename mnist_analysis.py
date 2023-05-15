import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from keras.datasets import mnist
from time import time

# Data Transform
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] ** 2)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] ** 2)

# KNN Classifier
neighbors = [1, 3]
for k in neighbors:
    # Fitting the KNearest Neighbor Classifier
    knn = KNeighborsClassifier(k)
    
    # Starting timer
    t = time()
    
    # Fitting the training data to the model
    knn.fit(train_X, train_y)
    
    # Creating first time stamp for time elapsed to fit the data
    time_fit = time() - t
    
    t = time()
    
    # Using testing data to evaluate the model
    predicted = knn.predict(test_X)
    
    # Creating second time stamp for time elapsed to predict
    time_predict = time() - t
    
    print(f"Analysis for K Nearest Neighbor = {k}")
    
    # Displaying the time it took to fit, predict the data, and the total runtime of the process.
    print(f"Time to fit: {time_fit:.2f}s - Time to predict: {time_predict:.2f}s\nTotal time: {(time_fit + time_predict):.2f}s")
    
    # Displaying the scores for each classification of the model.
    print(f"Classification report for classifier {knn}:\n{metrics.classification_report(test_y, predicted)}\n")
    
    # Using Confusion Matrix to display the Model's performance
    display = metrics.ConfusionMatrixDisplay.from_predictions(test_y, predicted)
    display.figure_.suptitle("Confusion Matrix")
    plt.show()

# Nearest Centroid
# Fitting the Nearest Centroid Classifier
NCentroid = NearestCentroid()

# Starting timer
t = time()
NCentroid.fit(train_X, train_y)

# Creating first time stamp for time elapsed to fit the data
time_fit = time() - t
t = time()

# Using testing data to evaluate the model
predicted = NCentroid.predict(test_X)

# Creating second time stamp for time elapsed to predict
predict_time = time() - t

print(f"Analysis for Nearest Centroid")

# Displaying the time it took to fit, predict the data, and the total runtime of the process.
print(f"Time to fit: {time_fit:.2f}s - Time to predict: {time_predict:.2f}s\nTotal time: {(time_fit + time_predict):.2f}s")

# Displaying the scores for each classification of the model.
print(f"Classification report for classifier {NCentroid}:\n{metrics.classification_report(test_y, predicted)}\n")

# Using Confusion Matrix to display the Model's performance
display = metrics.ConfusionMatrixDisplay.from_predictions(test_y, predicted)
display.figure_.suptitle("Confusion Matrix")
plt.show()
