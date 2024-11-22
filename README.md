# Multi-Classification Model on Synthetic Dataset

This project implements a multi-class classification model using PyTorch to classify data points generated with sklearn's `make_blobs`. It visualizes the dataset and the decision boundaries learned by the model.

---

## Files & Dependencies
1. **Python Environment**: Requires Python 3.7+.
2. **Libraries**:
   - `torch`: For defining the neural network and performing computations.
   - `matplotlib`: For visualizing the dataset and decision boundaries.
   - `sklearn`: For generating synthetic datasets and splitting data.
   - `numpy`: For numerical operations.

---

## Code Explanation

### Data Generation
- The `make_blobs` function generates a synthetic dataset with:
  - 4 classes (`NUM_CLASSES=4`).
  - 2 features (`NUM_FEATURES=2`).
  - Controlled randomness (`RANDOM_SEED=13`).
- The dataset is split into training and testing sets using `train_test_split`.

---

### Visualization of Data
- The scatter plot visualizes the points in a 2D plane, colored by their class.

---

### Model Architecture
- `MultiClassificationModel`:
  - Three fully connected (`Linear`) layers.
  - ReLU activation function after the first two layers.
  - Outputs 4 logits corresponding to the classes.

---

### Decision Boundary
- A grid is created spanning the data's feature space.
- The model predicts the class for each grid point.
- The predictions are visualized using a `contourf` plot, showing decision boundaries.

---

### Training and Prediction
- The model is set to evaluation mode for inference.
- `torch.inference_mode` is used to disable gradient computations for efficiency.

---

### Output
- The decision boundary plot shows how the model classifies different regions of the feature space.
- Data points are overlaid on the boundary plot to verify the model's learned boundaries.

---

# Customization Guide

## Dataset Parameters
- Modify `NUM_CLASSES`, `NUM_FEATURES`, or `cluster_std` in the `make_blobs` function to create different datasets.

## Model
- Adjust the number of layers or neurons in the `MultiClassificationModel` to experiment with the network's complexity.
- Try different activation functions (e.g., `LeakyReLU`, `Tanh`, `Sigmoid`) or optimizers (e.g., `Adam`, `SGD`) for performance variations.

## Visualization
- Change the `cmap` parameter in `plt.scatter` and `plt.contourf` to apply different color schemes to the plots.


---
