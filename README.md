# 🧠 Neural Network from Scratch (MNIST)

This project implements a fully-connected neural network from scratch using only NumPy and Matplotlib.
The model is trained and evaluated on the MNIST handwritten digits dataset (784 features, 10 classes).

## 📂 Project Structure

### Data Loading & Preprocessing

- Load MNIST training and test data.
- Normalize input features to [0, 1].
- One-hot encode labels into categorical format.

### Exploratory Data Analysis

- Display random MNIST digit samples.
- Inspect dataset dimensions and label distribution.

### Model Architecture

- Support for multiple hidden layers.
- Implemented activation functions:
  - Sigmoid
  - ReLU
  - Tanh
  - Leaky ReLU
- Softmax for output layer.

### Forward Propagation

- Matrix multiplication + chosen activation per layer.
- Final output probabilities via softmax.

### Loss Function

- Cross-entropy loss for multi-class classification.

### Backpropagation

- Compute gradients for weights and biases:
  - 𝑑𝑊 = (1/𝑚)𝑑𝑍⋅𝐴ᵀ
  - 𝑑𝑏 = (1/𝑚)∑𝑑𝑍
- Propagate errors backward through the network.

### Training

- Update parameters with gradient descent.
- Train over defined epochs with adjustable learning rate.
- Track cost and accuracy (train/test).

### Evaluation

- Accuracy function to measure performance.
- Predictions on unseen test data.
- Plot cost vs. epochs to visualize convergence.

## 🚀 Usage

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/neural-nets-from-scratch.git
cd neural-nets-from-scratch
pip install -r requirements.txt
```

### Run

Run the training script:

```bash
python neural_nets.py
```

### Example: Train a ReLU Network

```python
PARAMS = [X_train, y_train, X_test, y_test, "relu", 10, [128, 32]]
nn_relu = NN(*PARAMS)

epochs = 200
lr = 0.03

nn_relu.fit(lr=lr, epochs=epochs)
nn_relu.plot_cost(lr)
```

## 📈 Results

- **Architecture**: 2 hidden layers (128, 32 neurons).
- **Activation**: ReLU.
- **Epochs**: 200.
- **Learning Rate**: 0.03.
- **Accuracy**: Achieved high classification accuracy (>90%) on MNIST test data.
