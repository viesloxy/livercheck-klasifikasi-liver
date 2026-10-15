"""6. Prediksi data testing asli dengan model yang telah dilatih."""


def test_model(model, X_test):
    return model.predict(X_test)
