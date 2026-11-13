# Credit Card Fraud Detection – Unsupervised Anomaly Detection

## 📌 Project Overview

This project explores **unsupervised machine learning and deep learning techniques for credit card fraud detection**, with a particular focus on anomaly detection using autoencoders.

The main objective is to identify fraudulent transactions by learning the patterns of legitimate transactions and detecting deviations from normal behavior.

The project investigates multiple approaches, ranging from manually implemented neural networks to optimized deep learning architectures and classical anomaly detection algorithms.

It is organized into three complementary notebooks covering feature analysis, autoencoder architectures, hyperparameter optimization, and comparative performance evaluation.

## 🎯 Project Objectives

- Explore and analyze highly imbalanced financial transaction datasets.
- Investigate feature selection and the discriminative power of transaction attributes.
- Implement autoencoders for unsupervised anomaly detection.
- Compare Dense, Denoising, and Variational Autoencoders (VAE).
- Build an autoencoder from scratch using NumPy.
- Optimize model hyperparameters using Optuna and Bayesian optimization.
- Compare neural network approaches with classical anomaly detection algorithms.
- Evaluate models using metrics adapted to imbalanced classification.
- Analyze reconstruction errors and anomaly detection thresholds.

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset**, available on Kaggle:

[Credit Card Fraud Detection – ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The original dataset contains:

| Characteristic | Value |
|---|---:|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Input features | 30 |
| Fraud rate | 0.173% |

### Dataset Features

- `V1` to `V28`: anonymized features obtained through PCA.
- `Time`: elapsed time between transactions.
- `Amount`: transaction amount.
- `Class`: target variable (0 = legitimate, 1 = fraudulent).

The extreme class imbalance makes accuracy alone insufficient for evaluating model performance.

The project therefore emphasizes Precision, Recall, F1-score, and ROC-AUC.

## 🧠 Methodology

### 1. Exploratory Data Analysis and Feature Investigation

The first stage focuses on understanding the dataset and identifying relevant patterns.

Activities include:
- Analysis of class distribution and imbalance.
- Exploration of transaction amounts.
- Feature correlation analysis.
- Investigation of relationships between individual features and fraudulent transactions.
- Testing feature combinations using Logistic Regression and ROC-AUC to identify potentially discriminative feature pairs.

### 2. Autoencoder-Based Anomaly Detection

An autoencoder is a neural network trained to reconstruct its input through a compressed latent representation.

The underlying principle is that a model trained primarily on legitimate transactions learns their normal patterns and produces larger reconstruction errors for anomalous transactions.

The reconstruction error is used as an anomaly score:

\[
MSE(x)=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
\]

A threshold is then applied to distinguish normal transactions from potential fraud.

The project explores several architectures:

| Architecture | Description |
|---|---|
| Dense Autoencoder | Fully connected encoder-decoder architecture |
| Denoising Autoencoder | Learns robust representations from corrupted inputs |
| Variational Autoencoder (VAE) | Learns a probabilistic latent representation |
| NumPy Autoencoder | Neural network implemented manually without deep learning frameworks |

### 3. Hyperparameter Optimization with Optuna

Optuna is used to automate the search for better model configurations through the Tree-structured Parzen Estimator (TPE).

Depending on the notebook, the optimization explores parameters such as:

- Learning rate.
- Latent dimension.
- Number of hidden layers.
- Number of neurons.
- Batch size.
- Dropout rate.
- Number of epochs.
- Anomaly detection threshold.

The objective is to improve anomaly detection performance, particularly the F1-score.

### 4. Classical Anomaly Detection Algorithms

To establish meaningful comparisons, the project also investigates classical approaches:

- **Isolation Forest:** Detects anomalies through random isolation trees.
- **Local Outlier Factor (LOF):** Identifies observations with significantly different local densities.
- **One-Class SVM:** Learns a decision boundary around normal observations.
- **PCA Reconstruction:** Uses linear dimensionality reduction and reconstruction errors to detect anomalies.

### 5. Model Evaluation

The different approaches are evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of correctly identified fraud among detected anomalies |
| Recall | Proportion of actual fraud detected |
| F1-score | Balance between Precision and Recall |
| ROC-AUC | Ability to distinguish normal and fraudulent transactions |

Additional visualizations include:
- Training and validation loss curves.
- Reconstruction error distributions.
- Anomaly detection thresholds.
- Confusion matrices.
- ROC and Precision-Recall curves.
- Hyperparameter optimization progress.
- Comparative model performance charts.

## 📓 Project Notebooks

### Notebook 1 – Dense, Denoising and Variational Autoencoders

`Dense+Denoising+VA_E.ipynb`

Explores different autoencoder architectures for anomaly detection.

Main topics:
- Dense Autoencoder implementation.
- Denoising Autoencoder.
- Variational Autoencoder (VAE).
- Model training and reconstruction.
- Reconstruction error analysis.
- Threshold-based anomaly detection.
- Comparative evaluation of architectures.

**Goal:** Investigate how different latent representations and reconstruction strategies influence anomaly detection.

### Notebook 2 – Autoencoder Optimization and Classical Baselines

`AE+Optuna.ipynb`

Focuses on improving autoencoder performance and comparing it with traditional anomaly detection algorithms.

Main topics:
- Exploratory Data Analysis and preprocessing.
- Baseline Autoencoder using TensorFlow/Keras.
- Hyperparameter optimization using Optuna (50 trials).
- Final model training with optimized parameters.
- Isolation Forest, LOF, and One-Class SVM.
- Comparative performance evaluation.

**Goal:** Improve the precision-recall trade-off through automated optimization and establish comparisons with classical algorithms.

### Notebook 3 – Feature Testing and NumPy Autoencoder

`anomaly_detection_features_testing.ipynb`

Explores feature combinations and implements an autoencoder entirely from scratch using NumPy.

Main topics:
- Feature pair investigation using Logistic Regression.
- Feature discrimination analysis using ROC-AUC.
- Manual neural network implementation.
- Encoder-decoder architecture with a compressed latent space.
- ReLU and linear activation functions.
- Mean Squared Error loss.
- Manual mini-batch Stochastic Gradient Descent.
- He weight initialization.
- Reconstruction-based anomaly detection.
- Hyperparameter optimization using Optuna (25 trials).
- Comparison with Isolation Forest, LOF, One-Class SVM, and PCA.

**Goal:** Understand the mathematical foundations of autoencoders, investigate feature relevance, and evaluate the impact of hyperparameter optimization.

## 📈 Selected Experimental Results

The following results are taken from the individual notebook experiments. They should be interpreted separately because the preprocessing, architectures, thresholds, and experimental configurations differ.

### Autoencoder Optimization – Notebook 2

| Metric | Baseline AE | AE + Optuna |
|---|---:|---:|
| Accuracy | 96.72% | 99.03% |
| Precision | 4.23% | 13.11% |
| Recall | 83.67% | 82.65% |
| F1-score | 0.081 | 0.226 |
| ROC-AUC | 0.954 | 0.962 |

Optuna improves the F1-score significantly while maintaining a high fraud detection rate.

### NumPy Autoencoder – Notebook 3

| Metric | Baseline AE | AE + Optuna |
|---|---:|---:|
| Accuracy | 94.98% | 94.98% |
| Precision | 2.93% | 2.97% |
| Recall | 87.20% | 88.82% |
| F1-score | 0.0566 | 0.0576 |
| ROC-AUC | 0.9561 | 0.9613 |

The optimized architecture obtained the following configuration:

- Architecture: `30 → 16 → 32 → 4 → 32 → 16 → 30`
- Learning rate: `0.000752`
- Latent dimension: `4`
- Training epochs: `120`
- Optimization trials: `25`

The experiment demonstrates how a neural network can be implemented and optimized without relying on TensorFlow or PyTorch.

**Note:** These results belong to different experimental configurations and are not directly interchangeable.

## 🛠️ Technologies and Libraries

- **Python 3**
- **NumPy** – Numerical computing and manual neural network implementation.
- **Pandas** – Data manipulation and preprocessing.
- **Scikit-learn** – Classical anomaly detection algorithms, preprocessing, and evaluation.
- **TensorFlow / Keras** – Deep learning and autoencoder architectures.
- **Optuna** – Hyperparameter optimization.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical visualization.
- **Jupyter Notebook** – Interactive experimentation.

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Dense+Denoising+VA_E.ipynb
│
├── AE+Optuna.ipynb
│
├── anomaly_detection_features_testing.ipynb
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation and Execution

### 1. Clone the repository

```bash
git clone <repository-url>
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow optuna jupyter
```

### 3. Download the dataset

Download `creditcard.csv` from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the dataset in the appropriate directory and update its path in the notebooks.

**Note:** The notebooks use different data-loading configurations. Check their dataset paths before execution. The third notebook also contains a Windows-specific local dataset path that must be adapted.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks and execute their cells sequentially.

## 🔍 Key Concepts

- Unsupervised Machine Learning
- Anomaly Detection
- Credit Card Fraud Detection
- Exploratory Data Analysis
- Feature Selection and Feature Engineering
- Artificial Neural Networks
- Dense Autoencoders
- Denoising Autoencoders
- Variational Autoencoders
- Neural Networks from Scratch
- Backpropagation and Gradient Descent
- Latent Space Representation
- Singular and Statistical Reconstruction Errors
- Bayesian Hyperparameter Optimization
- Optuna and TPE
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- PCA Reconstruction
- Imbalanced Classification
- Precision-Recall Trade-off

## 🚀 Future Improvements

- Investigate more advanced autoencoder architectures.
- Explore alternative anomaly threshold calibration strategies.
- Experiment with semi-supervised anomaly detection.
- Improve feature selection and dimensionality reduction.
- Investigate model explainability techniques.
- Evaluate generalization on additional real-world fraud datasets.
- Explore hybrid approaches combining reconstruction errors and classical anomaly scores.

---

*Academic project exploring machine learning and deep learning approaches for financial fraud detection, from mathematical implementation to model optimization and comparative evaluation.*
