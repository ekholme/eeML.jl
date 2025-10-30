# eeML

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ekholme.github.io/eeML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ekholme.github.io/eeML.jl/dev/)
[![Build Status](https://github.com/ekholme/eeML.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ekholme/eeML.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/ekholme/eeML.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ekholme/eeML.jl)

## Roadmap (suggested by Gemini)

### Phase 1: Foundations and Core Utilities
*Before implementing any specific models, you should establish a solid foundation. This phase focuses on the fundamental components that will be used throughout the library.*

*   **Project Structure**: Set up a Julia package with a clear directory structure.
    *   `src/`: Main source code.
    *   `src/models/`: Directory for individual model implementations.
    *   `src/utilities/`: Directory for helper functions.
    *   `test/`: Unit tests.
    *   `docs/`: Documentation.
*   **Data Structures**: Define a common interface for your models. A good starting point is to create a `struct` for each model that stores its hyperparameters and learned parameters.
    *   *Example*: `struct LinearRegressionModel { ... }`
*   **Core Functions**: Implement basic functions that all models will need.
    *   `fit(model, X, y)`: A function that takes a model instance and data, trains the model, and returns the updated model.
    *   `predict(model, X)`: A function that takes a trained model and new data, and returns predictions.
    *   `score(model, X, y)`: A function to evaluate the model's performance.

### Phase 2: Supervised Learning - Linear Models
*Start with the simplest, most interpretable models. This phase will introduce you to key concepts like loss functions, optimization, and gradient descent.*

*   **Linear Regression**:
    *   Implement the core `fit` and `predict` functions.
    *   Use the closed-form solution (*normal equation*) for a direct solution.
    *   Implement *gradient descent* as an alternative optimization method. This is crucial for later, more complex models.
    *   Use metrics like *Mean Squared Error (MSE)* and *R²* to evaluate performance.
*   **Logistic Regression**:
    *   Extend your knowledge of linear models to classification.
    *   Implement the *sigmoid function*.
    *   Define the *cross-entropy loss function*.
    *   Implement *gradient descent* to find the optimal parameters.
    *   Use metrics like *accuracy*, *precision*, and *recall*.

### Phase 3: Supervised Learning - Non-Linear & Ensemble Methods
*Once you are comfortable with linear models, you can move on to more powerful and flexible algorithms.*

*   **K-Nearest Neighbors (k-NN)**:
    *   This is a simple but effective non-parametric method.
    *   Implement a function to calculate the distance between data points (e.g., *Euclidean distance*).
    *   The `fit` step is trivial (just storing the data), so focus on the `predict` function's logic.
*   **Decision Trees**:
    *   Implement a function to find the best split point (e.g., using *Gini impurity* or *information gain*).
    *   Recursively build the tree.
    *   Implement both classification and regression versions of the tree.
*   **Ensemble Methods (Random Forest)**:
    *   Combine your decision tree implementation with *bootstrapping* (random sampling with replacement) to build a random forest.
    *   Implement the voting/averaging mechanism for predictions.

### Phase 4: Unsupervised Learning & Preprocessing
*Machine learning is not just about prediction. Unsupervised methods are vital for exploring data and feature engineering.*

*   **K-Means Clustering**:
    *   Implement the core algorithm: randomly initialize centroids, assign points to the nearest centroid, and update the centroids.
    *   Handle the initialization process carefully (e.g., *K-means++*).
*   **Principal Component Analysis (PCA)**:
    *   This is a dimensionality reduction technique.
    *   Implement the algorithm using *Singular Value Decomposition (SVD)* from Julia's built-in linear algebra libraries.
*   **Preprocessing Utilities**:
    *   Create helper functions for common data preparation tasks.
    *   **StandardScaler**: Standardize features by removing the mean and scaling to unit variance.
    *   **MinMaxScaler**: Scale features to a given range, typically `[0,1]`.

### Phase 5: Neural Networks (A First Taste)
*This is a more advanced topic, but implementing a simple neural network will bring together many of the concepts you've learned.*

*   **Feed-Forward Neural Network**:
    *   Define a network structure with layers, weights, and biases.
    *   Implement common activation functions (e.g., *ReLU*, *sigmoid*, *softmax*).
    *   Implement the *forward pass* to compute predictions.
    *   Implement the *backward pass* (*backpropagation*) to calculate gradients.
    *   Use a more advanced optimizer like *Adam* or *RMSprop* (or stick with *stochastic gradient descent* for simplicity).

### Summary of Suggested Build Order
1.  **Phase 1**: Focus on setting up the project and defining your core `fit`, `predict`, and `score` functions.
2.  **Linear Regression**: Your first complete model. Start with the normal equation, then implement gradient descent.
3.  **Logistic Regression**: Your first classification model. This will force you to think about different loss functions and metrics.
4.  **K-Means Clustering**: A simple unsupervised learning algorithm. It doesn't use `fit` and `predict` in the same way, which is a good learning point.
5.  **Decision Trees**: A non-linear model that helps you understand tree-based algorithms.
6.  **Random Forest**: Build on your decision tree implementation to create an ensemble model.
7.  **PCA & Preprocessing Utilities**: Add these essential tools to make your models more effective.
8.  **Simple Neural Network**: The final, most complex step. This will tie together optimization, linear algebra, and calculus in a powerful way.