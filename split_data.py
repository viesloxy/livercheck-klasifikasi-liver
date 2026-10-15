"""2. Split 80:20; jalankan file ini langsung untuk mengekspor dua CSV."""

from pathlib import Path

from sklearn.model_selection import train_test_split


def split_data(frame):
    X = frame.drop(columns='Dataset')
    y = (frame['Dataset'] == 1).astype(int)  # 1 = liver; 0 = non-liver.
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def export_split(frame, output_dir):
    """Simpan baris asli hasil split, sebelum imputasi/Z-score/resampling."""
    X_train, X_test, _, _ = split_data(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / 'training.csv'
    test_path = output_dir / 'testing.csv'
    frame.loc[X_train.index].to_csv(train_path, index=False)
    frame.loc[X_test.index].to_csv(test_path, index=False)
    return train_path, test_path


if __name__ == '__main__':
    from preprocessing import load_and_clean

    dataset_dir = Path(__file__).resolve().parent / 'Dataset'
    cleaned = load_and_clean(dataset_dir / 'indian_liver_patient.csv')['frame']
    train_path, test_path = export_split(cleaned, dataset_dir)
    X_train, X_test, _, _ = split_data(cleaned)
    print(f'Training: {train_path} ({len(X_train)} baris)')
    print(f'Testing : {test_path} ({len(X_test)} baris)')
    print('CSV berisi data setelah deduplikasi, sebelum imputasi/transformasi.')
    print('Dataset asli tetap di indian_liver_patient.csv.')
