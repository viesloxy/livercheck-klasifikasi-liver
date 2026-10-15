"""1. Baca data, deteksi missing/duplikat/outlier, lalu buang duplikat."""

import pandas as pd


def load_and_clean(path):
    frame = pd.read_csv(path)
    initial_rows = len(frame)
    missing = frame.isna().sum().to_dict()
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    # IQR hanya untuk deteksi: nilai ekstrem klinis tidak dibuang.
    numeric = frame.drop(columns=['Gender', 'Dataset'])
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    outliers = ((numeric < q1 - 1.5 * (q3 - q1)) |
                (numeric > q3 + 1.5 * (q3 - q1))).sum().astype(int).to_dict()
    return {'frame': frame, 'initial_rows': initial_rows,
            'missing': missing, 'duplicates': duplicates, 'outliers': outliers}
