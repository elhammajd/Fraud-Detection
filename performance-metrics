from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test, nn_model=None, non_nn_model=None, meta_model=None):
    if meta_model is not None and nn_model is not None and non_nn_model is not None:
        nn_preds = nn_model.predict(X_test)
        non_nn_preds = non_nn_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
        meta_features = np.hstack([nn_preds, non_nn_preds])
        preds = meta_model.predict(meta_features)
    else:
        preds = model.predict(X_test)
        # If using a Keras model, threshold the probabilities
        if preds.ndim > 1:
            preds = (preds > 0.5).astype(int).flatten()
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, preds),
        'confusion_matrix': confusion_matrix(y_test, preds)
    }
    return metrics
