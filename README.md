# MNIST Classifier Comparison

## Project Overview
This project focuses on implementing and evaluating machine learning classifiers, specifically K-Nearest Neighbors (KNN) and Nearest Centroid, using the MNIST dataset. The MNIST dataset, renowned for its collection of 70,000 images of handwritten digits, serves as a benchmark for image classification tasks in machine learning.

## Key Features

### Data Preprocessing
- **Normalization**: Converts the image pixel values for optimal processing.
- **Flattening**: Reshapes the images from 2D arrays into 1D arrays suitable for input into the classifiers.

### Classifier Training and Evaluation
- **K-Nearest Neighbors (KNN)**: Implemented for `k=1` and `k=3`, this classifier labels a data point based on how its neighbors are classified.
- **Nearest Centroid**: Classifies data points based on the closest class centroid, providing a simple yet effective classification method.

### Performance Analysis
- Measures including accuracy, precision, recall, and F1-score are computed.
- Time metrics for model fitting and prediction are recorded and reported.

### Visualization
- Confusion matrices are generated for each classifier, offering an insightful view of the model's performance in classifying each digit.

## Installation Requirements
The project is developed using Python and requires the following libraries:
- `numpy`
- `matplotlib`
- `keras`
- `scikit-learn`

To install these libraries, use the following pip command:
```sh
pip install numpy matplotlib keras scikit-learn
```
## Running the Project

Execute the main script (`main.py`) to start the process. The script will handle the following tasks automatically:

- Data loading and preprocessing.
- Classifier training and evaluation.
- Result visualization.

## Contributions and Feedback

Contributions to this project are welcome. Please send your feedback and suggestions to enhance the project's functionality and efficiency.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
