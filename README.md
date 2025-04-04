# Fraud Detection

## Overview

The Fraud Detection Package is a modular Python toolkit for detecting fraudulent banking transactions. By leveraging ensemble methods that combine both neural network and traditional machine learning models, the package delivers enhanced predictive performance for real-world fraud detection scenarios. It comes with comprehensive tools for data preprocessing, balancing (using SMOTE), model training, performance evaluation, visualization, and statistical testing (Wilcoxon signed–rank test). This package is an excellent starting point for developing and deploying robust fraud detection solutions in financial applications.

## Features

- **Data Processing:** 
  - Load, clean, and preprocess datasets.
  - Split data into training and testing sets.
  - Balance imbalanced data using SMOTE.

- **Neural Network Ensemble:**
  - Build and train a neural network using TensorFlow/Keras.
  - Utilize dropout and dense layers for improved generalization.

- **Non-Neural Ensemble:**
  - Combine traditional machine learning models (RandomForest, AdaBoost, ExtraTrees) using a soft voting strategy.

- **Combined Ensemble:**
  - Fuse predictions from the neural network and non-neural ensemble using a logistic regression meta–classifier.

- **Evaluation Metrics:**
  - Compute accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix for thorough model assessment.

- **Visualization:**
  - Generate boxplots to analyze data distributions and model outputs.

- **Statistical Analysis:**
  - Perform the Wilcoxon signed–rank test for paired data comparisons.

- **Extensible Design:**
  - Easily integrate new models, evaluation metrics, or visualization techniques.

## Package Structure

fraud_detection/ 

├── init.py # Package initialization 

├── data_processing.py # Data loading, preprocessing, splitting, and balancing 

├── neural_network.py # Neural network model construction and training 

├── non_neural_network_ensemble.py # Building and training of traditional ensemble classifiers 

├── ensemble.py # Combining neural and non-neural ensembles using a meta-classifier 

├── metrics.py # Performance metrics evaluation 

└── visualization.py # Data visualization and statistical testing setup.py # Package installation configuration example_usage.py # Example script demonstrating package usage

bash
Copy

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd Fraud-Detection-main
Set Up a Virtual Environment (Recommended)

Create and activate a virtual environment to manage dependencies:

bash
Copy
python -m venv venv
# Activate on Linux/Mac:
source venv/bin/activate
# Activate on Windows:
venv\Scripts\activate
Install the Package

Install the package in editable mode:

bash
Copy
pip install -e .
This will install the following dependencies:

pandas

numpy

scikit-learn

imbalanced-learn

tensorflow

matplotlib

scipy

vbnet

