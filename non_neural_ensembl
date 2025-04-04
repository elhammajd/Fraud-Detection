from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

# Build an ensemble of non-neural network classifiers
def build_non_nn_ensemble():
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf3 = ExtraTreesClassifier(n_estimators=100, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', clf1), ('ada', clf2), ('et', clf3)], voting='soft')
    return ensemble
# Train the non-neural network ensemble
def train_non_nn_ensemble(ensemble, X_train, y_train):
    ensemble.fit(X_train, y_train)
    return ensemble
