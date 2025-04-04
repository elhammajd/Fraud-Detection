import pandas as pd
from fraud_detection.data_processing import load_data, preprocess_data, split_data, balance_data
from fraud_detection.neural_network import build_nn_model
from fraud_detection.non_neural_ensemble import build_non_nn_ensemble
from fraud_detection.ensemble import build_combined_ensemble, train_combined_ensemble
from fraud_detection.metrics import evaluate_model
from fraud_detection.visualization import plot_boxplot, perform_wilcoxon_test
from sklearn.linear_model import LogisticRegression

# Link to sample data:
print("Download sample data from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
data_path = "creditcard.csv"  # Update this path to where the dataset is stored

# Load and preprocess the data
data = load_data(data_path)
X, y = preprocess_data(data, target_column='Class')
X_train, X_test, y_train, y_test = split_data(X, y)

# Balance the training data
X_train_bal, y_train_bal = balance_data(X_train, y_train)

# Build models
nn_model = build_nn_model(X_train_bal.shape[1])
non_nn_ensemble = build_non_nn_ensemble()
meta_model = LogisticRegression()  # Meta-model for combining predictions

# Train the combined ensemble
meta_model, trained_nn_model, trained_non_nn_model = train_combined_ensemble(
    meta_model, nn_model, non_nn_ensemble, X_train_bal, y_train_bal, epochs=20, batch_size=32)

# Evaluate the combined ensemble on the test set
metrics = evaluate_model(None, X_test, y_test, nn_model=trained_nn_model, non_nn_model=trained_non_nn_model, meta_model=meta_model)
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Plot a boxplot for the first 10 features in the test set
plot_boxplot(X_test.iloc[:, :10], title="Boxplot of First 10 Features", xlabel="Features", ylabel="Value")

# Perform a Wilcoxon test between the first two features
feature1 = X_test.iloc[:, 0]
feature2 = X_test.iloc[:, 1]
stat, p_value = perform_wilcoxon_test(feature1, feature2)
print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
