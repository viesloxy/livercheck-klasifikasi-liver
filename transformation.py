"""3. Median training, encoding Gender, dan Z-score numerik training."""

from sklearn.preprocessing import StandardScaler


def transform_data(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    numeric = X_train.columns.drop('Gender')
    for frame in (X_train, X_test):
        frame['Gender'] = frame['Gender'].map({'Female': 0, 'Male': 1})
    if X_train['Gender'].isna().any() or X_test['Gender'].isna().any():
        raise ValueError('Gender harus Female atau Male')

    # Median dan StandardScaler fit hanya pada training, bukan testing.
    medians = X_train[numeric].median()
    X_train[numeric] = X_train[numeric].fillna(medians)
    X_test[numeric] = X_test[numeric].fillna(medians)
    scaler = StandardScaler()
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])
    return {'X_train': X_train.to_numpy(dtype=float),
            'X_test': X_test.to_numpy(dtype=float),
            'medians': medians.to_dict(),
            'scaler': scaler}
