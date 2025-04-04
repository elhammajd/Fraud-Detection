import numpy as np
from sklearn.linear_model import LogisticRegression

# Build a meta-classifier to combine NN and non-NN ensemble predictions.   
# The meta-model uses the two model predictions as features.
def build_combined_ensemble(nn_model, non_nn_model):
    meta_model = LogisticRegression()
    return meta_model

    """Train combined ensemble:
       - Train the neural network and non-neural network ensemble separately.
       - Create meta-features from their predictions.
       - Train the meta-model on these meta-features.
    """
def train_combined_ensemble(meta_model, nn_model, non_nn_model, X_train, y_train, epochs=20, batch_size=32):
    # For NN training, further split training data
    from sklearn.model_selection import train_test_split
    X_nn_train, X_nn_val, y_nn_train, y_nn_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Train neural network
    from .neural_network import build_nn_model, train_nn_model
    nn_input_dim = X_train.shape[1]
    nn_model = build_nn_model(nn_input_dim)
    nn_model, _ = train_nn_model(nn_model, X_nn_train, y_nn_train, X_nn_val, y_nn_val, epochs=epochs, batch_size=batch_size)
    
    # Train non-neural ensemble
    from .non_neural_ensemble import train_non_nn_ensemble
    non_nn_model = train_non_nn_ensemble(non_nn_model, X_train, y_train)
    
    # Create meta-features from predictions on training data
    nn_preds = nn_model.predict(X_train)
    non_nn_preds = non_nn_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    meta_features = np.hstack([nn_preds, non_nn_preds])
    
    meta_model.fit(meta_features, y_train)
    return meta_model, nn_model, non_nn_model
