"""4. Seimbangkan kelas hanya pada data training (ROS atau SMOTE)."""

from imblearn.over_sampling import RandomOverSampler, SMOTENC


def resample_training(X_train, y_train, method):
    if method == 'ROS':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'SMOTE':
        # SMOTENC adalah varian SMOTE agar Gender sintetis tetap 0/1.
        sampler = SMOTENC(categorical_features=[1], random_state=42)
    else:
        raise ValueError(f'Metode resampling tidak dikenal: {method}')
    return sampler.fit_resample(X_train, y_train)
