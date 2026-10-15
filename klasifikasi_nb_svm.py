# Tugas TM 8 Klasifikasi : Naive Bayes, SVM, dan Decision Tree
# Kelompok 2 : 434241083, 434241084, 434241085

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)

from preprocessing import load_and_clean
from split_data import split_data
from transformation import transform_data
from resampling import resample_training

# 1. Pipeline persis seperti UTS : bersihkan, split 80:20, transformasi
cleaned = load_and_clean('Dataset/indian_liver_patient.csv')
X_train, X_test, y_train, y_test = split_data(cleaned['frame'])
transformed = transform_data(X_train, X_test)
X_train_t = transformed['X_train']
X_test_t = transformed['X_test']
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print('=' * 72)
print('KLASIFIKASI PENYAKIT LIVER : NAIVE BAYES vs SVM vs DECISION TREE')
print('Pipeline UTS : ILPD, split 80:20, Z-score, resampling ROS & SMOTE')
print('=' * 72)
print(f"Data awal              : {cleaned['initial_rows']} baris")
print(f"Duplikat dibuang       : {cleaned['duplicates']} baris")
print(f"Data bersih            : {len(X_train) + len(X_test)} baris")
print(f"Data training          : {len(X_train_t)} baris "
      f"({np.bincount(y_train)[0]} non-liver : {np.bincount(y_train)[1]} liver)")
print(f"Data testing           : {len(X_test_t)} baris "
      f"({np.bincount(y_test)[0]} non-liver : {np.bincount(y_test)[1]} liver)")

# 2. Tiga model yang dibandingkan
model_dict = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
}
resampling_list = ['ROS', 'SMOTE']

hasil = []
model_terbaik = None
akurasi_terbaik = -1

for metode_resampling in resampling_list:
    X_balanced, y_balanced = resample_training(X_train_t, y_train, metode_resampling)
    print('\n' + '=' * 72)
    print(f"RESAMPLING {metode_resampling} : data training seimbang "
          f"{np.bincount(y_balanced)[0]} : {np.bincount(y_balanced)[1]}")
    print('=' * 72)

    for nama, model in model_dict.items():
        model.fit(X_balanced, y_balanced)
        prediksi = model.predict(X_test_t)

        cm = confusion_matrix(y_test, prediksi, labels=[0, 1])
        TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        akurasi = accuracy_score(y_test, prediksi)
        presisi = precision_score(y_test, prediksi, zero_division=0)
        recall = recall_score(y_test, prediksi, zero_division=0)
        f1 = f1_score(y_test, prediksi, zero_division=0)
        sensitivitas = TP / (TP + FN) * 100
        spesifisitas = TN / (TN + FP) * 100
        label = f'{nama} ({metode_resampling})'

        print('\n' + f'CLASSIFICATION REPORT {label}'.center(72, '='))
        print(classification_report(y_test, prediksi, zero_division=0))
        print(f'Confusion Matrix : TN={TN}  FP={FP}  FN={FN}  TP={TP}')
        print(f'Akurasi       : {akurasi * 100:.2f} %')
        print(f'Presisi       : {presisi:.4f}')
        print(f'Recall        : {recall:.4f}')
        print(f'F1-Score      : {f1:.4f}')
        print(f'Sensitivity   : {sensitivitas:.2f} %')
        print(f'Specificity   : {spesifisitas:.2f} %')

        hasil.append({'Metode': nama, 'Resampling': metode_resampling,
                      'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
                      'Akurasi': round(akurasi * 100, 2),
                      'Presisi': round(presisi, 4),
                      'Recall': round(recall, 4),
                      'F1': round(f1, 4),
                      'Sensitivity': round(sensitivitas, 2),
                      'Specificity': round(spesifisitas, 2)})

        if akurasi > akurasi_terbaik:
            akurasi_terbaik = akurasi
            model_terbaik = {'label': label, 'model': model,
                             'scaler': transformed['scaler'],
                             'medians': transformed['medians'],
                             'features': list(X_train.columns)}

# 3. Tabel perbandingan seluruh kombinasi
tabel = pd.DataFrame(hasil)
print('\n' + '=' * 72)
print('PERBANDINGAN KINERJA METODE')
print('=' * 72)
print(tabel.to_string(index=False))

terbaik = tabel.sort_values(['Akurasi', 'F1'], ascending=False).iloc[0]
print(f"\nMetode terbaik : {terbaik['Metode']} ({terbaik['Resampling']}) "
      f"dengan akurasi {terbaik['Akurasi']} %")

tabel.to_csv('hasil_perbandingan.csv', index=False)
print('Tabel perbandingan disimpan ke hasil_perbandingan.csv')

# 4. Heatmap keenam confusion matrix
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, baris in zip(axes.ravel(), hasil):
    matriks = [[baris['TN'], baris['FP']], [baris['FN'], baris['TP']]]
    sns.heatmap(matriks, annot=True, fmt='d', cmap='rocket_r', ax=ax, cbar=False)
    ax.set_title(f"{baris['Metode']} ({baris['Resampling']})\n"
                 f"Akurasi {baris['Akurasi']} %", fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('hasil_confusion_matrix.png', dpi=130, bbox_inches='tight')
print('Heatmap confusion matrix disimpan ke hasil_confusion_matrix.png')

# 5. Simpan model terbaik beserta perlengkapannya untuk aplikasi Streamlit
joblib.dump(model_terbaik, 'model_terbaik.joblib')
print(f"Model terbaik ({model_terbaik['label']}) disimpan ke model_terbaik.joblib")
