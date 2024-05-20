# Implementation and Analysis of Logistic Regression

## Introduction

This report details the implementation of the Logistic Regression algorithm within a Python environment, using the scikit-learn library. The algorithm is applied for binary classification tasks on a specified dataset, focusing on model training, parameter tuning, and evaluation.

## Data Preparation

The process begins with data loading and preprocessing:
- **Dataset Loading**: Two CSV files are imported, one for training and another for testing.
- **Feature Scaling**: The `StandardScaler` is employed to normalize feature values, ensuring that each feature contributes equally during distance calculation.
- **Label Adjustment**: For binary classification, labels are converted to a binary format, where any non-zero class label is set to 1, and 0 remains unchanged.

## Model Configuration and Evaluation

The dataset is split into training and validation sets, followed by:
- **Hyperparameter Tuning**: We adjust the parameter `C`, `max_iter` to explore its impact on model performance.
- **Cross-Validation**: Employing a 5-fold cross-validation to estimate the effectiveness of the model under different configurations.
- **Performance Metrics**: Accuracy, recall, and F1-score are calculated to assess model performance across the training and validation datasets.

## Hyperparameters and Metrics

### Key Parameters Tuned
- `C`: Controls the regularization strength of the model.
- `max_iter`: Maximum number of iterations taken for the solvers to converge.

### Solver Used
1. **liblinear**:
   - A good choice for small datasets.
   - The update formula for each coefficient is not easily expressed in a simple mathematical form as it depends on the specific coordinate descent algorithm.

2. **newton-cg**:
   - uses Newton's method combined with conjugate gradients for optimization.
   - Formula: θ^(k+1) = θ^(k) − α(H^(−1)∇J)
   - θ^(k)is the parameter vector at iteration 𝑘, α is the step size (learning rate), ∇J is the gradient of the cost function, H is the Hessian matrix.

3. **LBFGS**:
   - A limited-memory variant of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm.
   - The update formula for each iteration is similar to that of Newton's method but with an approximation of the inverse Hessian matrix.

4. **SAG/Saga**:
   - SAG (Stochastic Average Gradient) and Saga (SAGA) are stochastic gradient descent (SGD) methods that use an average gradient to update the coefficients.
   - The update formula for each iteration involves computing the gradient of the cost function for a subset of the training data (mini-batch) and adjusting the coefficients accordingly.

### Hyperparameter Influence
- **C**: controls the regularization strength of the model. Too high might lead to overfitting, too low could lead underfitting.

## Results and Visualization

The performance of the model across different hyperparameter settings is visualized through plots that show the accuracy and standard deviation of the model during cross-validation:
- **Accuracy and Validation Plots**: These plots help identify the best `C` and other parameters for optimal model performance.
- **Confusion Matrices**: Generated for both training and validation phases to visualize the model's effectiveness in classifying each class correctly.

## Conclusion

The application of the Logistic Regression algorithm has demonstrated effective classification capability in binary settings. The thorough exploration of hyperparameters and their impact on model accuracy provides valuable insights, potentially guiding further optimizations.

## Future Work

To enhance the current model:
- **Feature Engineering**: Advanced feature selection and reduction techniques could be implemented to refine the model input.
- **Algorithm Optimization**: Exploring more efficient data structures or parallel processing could improve computation time and scalability.
- **Hybrid Models**: Combining Logistic Regression with other algorithms could address its weaknesses and enhance overall predictive power.

