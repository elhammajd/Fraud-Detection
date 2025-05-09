import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
# Separate features and target
def preprocess_data(data, target_column='Class'):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

# Split data into training and testing sets
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# using SMOTE
def balance_data(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
