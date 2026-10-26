# Handwritten Digit Recognizer — Neural Network from Scratch (NumPy)

A two-layer neural network that classifies handwritten digits (0–9), built using only **NumPy** and math — no TensorFlow, no PyTorch. Every part of the pipeline is implemented by hand: forward propagation, backpropagation, and gradient descent.

On a held-out validation set of 1,000 images, the model reaches **90.4% accuracy** after 500 iterations.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How It Works](#how-it-works)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations and Future Work](#limitations-and-future-work)
- [Credits](#credits)

---

## Overview

I built this project to understand what deep learning frameworks do under the hood. Instead of calling `model.fit()`, I wrote each step myself — weight initialization, activation functions, the softmax output, the gradients for backpropagation, and the parameter updates — and trained the network on the Kaggle MNIST digit data.

## Dataset

The data comes from the Kaggle **[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)** competition (based on MNIST).

- `train.csv` — 42,000 labelled images. Each row has a `label` column followed by 784 pixel columns (a flattened 28 × 28 grayscale image, values 0–255).
- `test.csv` — 28,000 unlabelled images (used for Kaggle submissions).
- `sample_submission.csv` — the submission format.

The dataset is **not included** in this repository. See [Getting Started](#getting-started) for how to download it.

**Data split used in the notebook:**

| Split      | Samples | Purpose                         |
|------------|---------|---------------------------------|
| Training   | 41,000  | Learning the weights            |
| Validation | 1,000   | Measuring accuracy on unseen data |

The rows are shuffled before splitting, and pixel values are normalized from `[0, 255]` to `[0, 1]`.

## Model Architecture

```
Input layer        Hidden layer         Output layer
784 neurons   ──►  10 neurons (ReLU) ──► 10 neurons (Softmax)
(28×28 pixels)                          (one per digit 0–9)
```

| Layer   | Shape of weights | Activation |
|---------|------------------|------------|
| Hidden  | W1: 10 × 784, b1: 10 × 1 | ReLU |
| Output  | W2: 10 × 10, b2: 10 × 1  | Softmax |

## How It Works

**Forward propagation**

$$Z^{[1]} = W^{[1]} X + b^{[1]}, \qquad A^{[1]} = \text{ReLU}(Z^{[1]})$$

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}, \qquad A^{[2]} = \text{softmax}(Z^{[2]})$$

**Backpropagation** (with one-hot encoded labels $Y$ and $m$ training examples)

$$dZ^{[2]} = A^{[2]} - Y$$

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}, \qquad db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \odot \text{ReLU}'(Z^{[1]})$$

$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^{T}, \qquad db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

**Parameter update** (gradient descent with learning rate $\alpha$)

$$W := W - \alpha \, dW, \qquad b := b - \alpha \, db$$

## Implementation Details

A few choices I made to keep training stable and the code correct:

- **He initialization** — weights are drawn from a normal distribution scaled by $\sqrt{2 / n_{\text{in}}}$, which suits ReLU layers and avoids starting with activations that are too large or too small. Biases start at zero.
- **Numerically stable softmax** — the column-wise maximum is subtracted from $Z$ before exponentiating, so `np.exp` doesn't overflow.
- **Input normalization** — pixel values are scaled to `[0, 1]` so gradients stay in a reasonable range.
- **Correct bias gradients** — bias gradients are summed per neuron (`axis=1, keepdims=True`), so they keep the shape `(10, 1)` instead of collapsing into a single scalar.
- **Prediction visualization** — a helper function displays any training image with its predicted and true label side by side.

**Hyperparameters**

| Parameter      | Value |
|----------------|-------|
| Iterations     | 500   |
| Learning rate  | 0.1   |
| Batch          | Full batch (all 41,000 training samples per step) |

## Results

| Metric                                  | Accuracy |
|-----------------------------------------|----------|
| Training accuracy (iteration 450)       | ~90.6%   |
| Validation accuracy (1,000 held-out images) | **90.4%** |

Training accuracy across iterations:

| Iteration | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 |
|-----------|---|----|-----|-----|-----|-----|-----|-----|-----|-----|
| Accuracy  | 9.4% | 77.3% | 83.8% | 86.3% | 88.0% | 88.9% | 89.5% | 90.0% | 90.3% | 90.6% |

Training and validation accuracy are close, so the model is not noticeably overfitting. Accuracy was still rising at the end, which suggests more iterations would help.

> Note: results will vary slightly between runs because the data shuffle and weight initialization are random.

## Repository Structure

```
Digit-recognizer/
├── data/                       # Place Kaggle CSV files here (not tracked by git)
│   └── .gitkeep
├── notebooks/
│   └── digit_recognizer.ipynb  # Full implementation: data prep, training, evaluation
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/debugging-life08/Digit-recognizer.git
cd Digit-recognizer
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download the data from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and place `train.csv` (and optionally `test.csv`) in the `data/` folder.

Or, using the Kaggle CLI:

```bash
kaggle competitions download -c digit-recognizer -p data/
unzip data/digit-recognizer.zip -d data/
```

**4. Run the notebook**

```bash
jupyter notebook notebooks/digit_recognizer.ipynb
```

Run all cells. The notebook trains the network, prints accuracy every 50 iterations, shows a sample prediction, and reports the final validation accuracy.

## Limitations and Future Work

- **Small hidden layer** — 10 hidden neurons is a tight bottleneck. A larger hidden layer (e.g., 64 or 128) or a second hidden layer would likely raise accuracy.
- **Full-batch gradient descent** — every update uses all 41,000 samples. Mini-batch gradient descent would update more often and usually converge faster.
- **No fixed random seed** — setting `np.random.seed(...)` would make results exactly reproducible.
- **No loss tracking** — logging cross-entropy loss alongside accuracy would give a clearer picture of convergence.
- **Kaggle submission** — the model has not yet been run on `test.csv` to produce a submission file.
- **Other ideas** — confusion matrix and per-digit error analysis, learning rate schedules, L2 regularization, and comparing against a scikit-learn or PyTorch baseline.

## Credits

- This project was inspired by the video [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://youtu.be/w8yWXqWQYmU) by [Samson Zhang](https://www.youtube.com/@SamsonZhangTheSalmon).
- Dataset: [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) on Kaggle, derived from the MNIST database by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
