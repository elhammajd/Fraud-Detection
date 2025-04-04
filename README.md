# Fraud Detection Package

## Overview

The Fraud Detection Package is a modular Python toolkit designed for detecting fraudulent banking transactions. It leverages ensemble methods that combine both neural network and traditional machine learning models to improve predictive performance. The package includes data preprocessing, balancing using SMOTE, model training, evaluation metrics, visualization tools (boxplots), and statistical testing (Wilcoxon signed–rank test). It is intended to serve as a robust starting point for developing and deploying fraud detection solutions.

## Features

- **Data Processing:** Load, preprocess, and split datasets; balance imbalanced data using SMOTE.
- **Neural Network Ensemble:** Build and train a neural network using TensorFlow/Keras.
- **Non-Neural Ensemble:** Combine multiple classic machine learning models (RandomForest, AdaBoost, ExtraTrees) via soft voting.
- **Combined Ensemble:** Fuse the outputs of neural and non-neural models using a logistic regression meta–classifier.
- **Evaluation Metrics:** Calculate accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
- **Visualization:** Create boxplots to analyze data distributions.
- **Statistical Analysis:** Perform the Wilcoxon signed–rank test for paired data comparisons.
- **Extensible Design:** Easily customize or extend the package for new models, metrics, or tests.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Fraud-Detection-main
