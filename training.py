"""5. Latih Decision Tree pada data training yang sudah seimbang."""

from sklearn.tree import DecisionTreeClassifier


def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
