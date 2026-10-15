# Eksperimen UI Streamlit versi 2 : LiverCheck
# Style mengikuti blog editorial : Inter, putih, tombol pill hitam,
# link hover underline, section berborder abu, aksen biru saat hover.
# CATATAN : file eksperimen pribadi, bukan bagian folder pengumpulan.

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import base64
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title='LiverCheck Klasifikasi Penyakit Liver',
                   page_icon=None, layout='wide',
                   initial_sidebar_state='collapsed')

CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;800&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #FFFFFF; color: #151519; }
header[data-testid="stHeader"] { display: none; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none; }
.block-container { padding-top: 1rem; max-width: 1150px; }

/* header logo dan nav */
.logochip { width: 47px; }
.logochip img { width: 100%; display: block; }
[data-testid="stImage"] img { display: block; }
[data-baseweb="segmented-control"] {
    background: transparent; border: none; padding: 0; box-shadow: none;
    gap: 6px;
}
[data-baseweb="segmented-control"] label {
    border-radius: 8px; padding: 6px 14px; font-weight: 500;
    font-size: 15px; color: #52525B; border: none !important; background: transparent !important; outline: none !important; box-shadow: none !important;
}
[data-baseweb="segmented-control"] label:hover {
    color: #151519; text-decoration: underline; text-underline-offset: 6px;
}
[data-baseweb="segmented-control"] label[data-checked="true"] {
    background: transparent !important; color: #151519 !important;
    font-weight: 700; text-decoration: underline;
    text-underline-offset: 6px; text-decoration-thickness: 2px;
}

/* tombol pill hitam ala blogy */
.stButton > button {
    background: #151519; color: #FFFFFF; border: none;
    border-radius: 200px; padding: 11px 25px; font-weight: 500;
    font-size: 15px; font-family: 'Inter', sans-serif;
}
.stButton > button:hover { background: #F2CA4B; color: #221F1A; }

/* tipografi hero editorial */
.hero-title { font-size: 42px; font-weight: 300; line-height: 1.15;
    color: #151519; letter-spacing: -0.02em; }
.hero-title strong { font-weight: 800; }
.hero-text { font-size: 15px; color: #52525B; line-height: 1.85;
    margin-top: 16px; }

/* section border */
.garis { border-bottom: 1px solid #E3E3E8; margin: 26px 0; }

/* kartu */
.card {
    background: #FFFFFF; border: 1px solid #E3E3E8; border-radius: 16px;
    padding: 24px 26px; height: 100%;
}
.card h3 { font-size: 15px; font-weight: 700; color: #151519;
    margin: 0 0 10px 0; }
.card p { font-size: 13.5px; color: #52525B; line-height: 1.85; margin: 6px 0; }
.tag {
    display: inline-block; background: #F3F1F4; color: #151519;
    font-size: 12px; font-weight: 500; padding: 3px 10px;
    border-radius: 200px; margin: 0 6px 8px 0;
}
.taglink { text-decoration: none; }
.taglink:hover .tag { background: #F2CA4B; color: #FFFFFF; }

/* form dan input */
[data-testid="stForm"] { background: #FFFFFF; border: 1px solid #E3E3E8;
    border-radius: 16px; padding: 1.5rem 1.7rem; }
[data-testid="stNumberInput"] input {
    background: #F3F1F4; border: 1px solid #E3E3E8; border-radius: 14px;
    color: #151519 !important; font-weight: 500; font-family: 'Inter',
    sans-serif; padding-right: 10px;
}
[data-testid="stNumberInput"] button { display: none; }
[data-testid="stNumberInput"] input:focus {
    border-color: #F2CA4B; outline: none; box-shadow: none; }
[data-testid="stSelectbox"] > div > div {
    background: #F3F1F4; border: 1px solid #E3E3E8; border-radius: 14px;
    color: #151519 !important;
}
[data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label {
    font-weight: 500; color: #151519; font-size: 13.5px; }

/* progress peluang */
.track { background: #F3F1F4; border-radius: 200px; height: 12px; width: 100%; }
.fill { background: #F2CA4B; border-radius: 200px; height: 12px; }

/* link */
a { color: #151519; text-decoration: underline; text-underline-offset: 4px; }
a:hover { color: #B8912F; }

h1 { display: none; }
.footer { font-size: 12.5px; color: #52525B; padding: 14px 0 2px 0; }
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

# ================== MODEL DAN DATA ==================
artefak = joblib.load('model_terbaik.joblib')
model = artefak['model']
scaler = artefak['scaler']
fitur = artefak['features']
fitur_numerik = [f for f in fitur if f != 'Gender']
data = pd.read_csv('indian_liver_patient.csv')
logo_b64 = base64.b64encode(open('assets/logo_header.png', 'rb').read()).decode()
liver_img = Image.open('assets/liver.png')

if 'nav' not in st.session_state:
    st.session_state['nav'] = 'Beranda'


def ke_prediksi():
    st.session_state['nav'] = 'Prediksi'


def ke_beranda():
    st.session_state['nav'] = 'Beranda'


# ================== HEADER ==================
h1, h2, h3 = st.columns([0.9, 2.6, 1.1], vertical_alignment='center')
with h1:
    tag_logo = ('<div class="logochip"><img src="data:image/png;base64,' + logo_b64 + '"></div></div>')
    st.markdown(tag_logo, unsafe_allow_html=True)
with h2:
    st.segmented_control('nav', ['Beranda', 'Dataset', 'Prediksi'],
                         key='nav', label_visibility='collapsed')
with h3:
    st.button('Coba Prediksi', on_click=ke_prediksi,
              use_container_width=True)

garis = '<div style="border-bottom:1px solid #E3E3E8;margin:14px 0 26px 0"></div>'
st.markdown(garis, unsafe_allow_html=True)

# ================== BERANDA ==================
if st.session_state['nav'] == 'Beranda':
    kiri, kanan = st.columns([1.4, 1])
    with kiri:
        st.markdown('<div class="hero-title"><strong>Kenali kondisi '
                    'livermu.</strong> Cukup dari sepuluh angka '
                    'laboratorium biasa.</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-text">LiverCheck adalah alat bantu '
                    'klasifikasi penyakit liver yang bekerja atas dasar '
                    'Indian Liver Patient Dataset. Model SVM dengan '
                    'penyeimbangan SMOTE membaca sepuluh nilai '
                    'laboratorium, lalu memberi perkiraan kelas beserta '
                    'peluangnya dalam hitungan detik. Seluruh proses '
                    'berjalan lokal di perangkat ini.</div>',
                    unsafe_allow_html=True)
        st.write('')
        st.button('Mulai Prediksi', on_click=ke_prediksi)
    with kanan:
        st.image(liver_img, use_container_width=True)

    st.markdown(garis, unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Angka Yang Perlu Diketahui</h3>'
                '<p>Model terbaik pada eksperimen ini adalah SVM dengan '
                'resampling SMOTE yang mencatat akurasi 74.56 persen dan '
                'F1 Score 0.80 pada 114 data testing, tertinggi dibanding '
                'Naive Bayes dan Decision Tree pada pipeline yang '
                'sama.</p></div>', unsafe_allow_html=True)
    st.write('')
    k1, k2 = st.columns(2)
    with k1:
        st.markdown('<div class="card"><h3>Tentang Dataset</h3>'
                    '<p>Data yang dipakai adalah Indian Liver Patient '
                    'Dataset hasil pemeriksaan laboratorium pasien, '
                    'berisi sepuluh fitur seperti umur, jenis kelamin, '
                    'bilirubin, enzim hati, protein, albumin, dan rasio '
                    'albumin globulin. Dataset aslinya tersedia terbuka '
                    'di Kaggle.</p>'
                    '<a class="taglink" href="https://www.kaggle.com/'
                    'datasets/uciml/indian-liver-patient-records">'
                    '<span class="tag">Buka dataset di Kaggle</span></a>'
                    '</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="card"><h3>Tentang Model</h3>'
                    '<p>SVM mencari garis pemisah terbaik antar kelas '
                    'dengan margin maksimal, lalu kernel RBF membantu '
                    'memetakan data ke ruang yang lebih mudah dipisah. '
                    'Sebelum dilatih, data training diseimbangkan dengan '
                    'SMOTE sehingga kelas non liver dan liver sama '
                    'banyaknya.</p>'
                    '<span class="tag">SVM Kernel RBF</span>'
                    '<span class="tag">SMOTENC</span>'
                    '<span class="tag">Z Score</span>'
                    '<span class="tag">Split 80 : 20</span>'
                    '</div>', unsafe_allow_html=True)
    st.markdown(garis, unsafe_allow_html=True)

# ================== DATASET ==================
elif st.session_state['nav'] == 'Dataset':
    st.markdown('<div class="hero-title"><strong>Indian Liver Patient '
                'Dataset.</strong> Lima ratus delapan puluh tiga catatan '
                'laboratorium nyata.</div>', unsafe_allow_html=True)
    st.write('')
    para = ('Dataset berasal dari pemeriksaan pasien di India timur laut. '
            'Setelah pembersihan, tersisa 570 baris dengan kolom target '
            'bernilai 1 untuk pasien liver dan 0 untuk non liver. Empat '
            'nilai kosong diisi median data training, tiga belas baris '
            'duplikat dibuang, dan seluruh fitur numerik distandarisasi '
            'dengan Z Score yang dipasang hanya pada data training.')
    st.markdown(f'<div class="card"><p>{para}</p>'
                '<a class="taglink" href="https://www.kaggle.com/datasets/'
                'uciml/indian-liver-patient-records">'
                '<span class="tag">Sumber dataset di Kaggle</span></a>'
                '<span class="tag">583 baris awal</span>'
                '<span class="tag">570 baris bersih</span>'
                '</div>', unsafe_allow_html=True)
    st.write('')
    k1, k2 = st.columns([1, 1.6])
    with k1:
        l1 = data['Dataset'].map({1: 'Liver', 2: 'Non Liver'}).value_counts()
        maks = l1.max()
        bar_html = ''
        for nama, jumlah in l1.items():
            lebar = int(jumlah / maks * 100)
            bar_html += ('<p style="margin:8px 0 4px 0;font-weight:700;'
                         'color:#151519;font-size:13px">' + nama + ' : '
                         + str(jumlah) + ' baris</p>'
                         '<div class="track"><div class="fill" style="width:'
                         + str(lebar) + '%"></div></div>')
        st.markdown('<div class="card"><h3>Sebaran Kelas</h3>' + bar_html
                    + '</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="card"><h3>Sepuluh Baris Pertama</h3>'
                    '</div>', unsafe_allow_html=True)
        tampil = data.copy()
        tampil['Dataset'] = tampil['Dataset'].map({1: 'Liver', 2: 'Non Liver'})
        st.dataframe(tampil.head(10), use_container_width=True,
                     hide_index=True)

# ================== PREDIKSI ==================
elif st.session_state['nav'] == 'Prediksi':
    kiri, kanan = st.columns([1.5, 1])
    with kiri:
        with st.form('form_pasien', border=False):
            st.markdown('<h3 style="font-weight:700;color:#151519;'
                        'margin:0 0 4px 0">Form Nilai Laboratorium Pasien'
                        '</h3>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                umur = st.number_input('Umur (tahun)', 4, 100, 45)
                gender = st.selectbox('Jenis Kelamin', ['Female', 'Male'])
                tb = st.number_input('Total Bilirubin (mg dL)', 0.0, 80.0,
                                     1.0, 0.1)
                db = st.number_input('Direct Bilirubin (mg dL)', 0.0, 25.0,
                                     0.3, 0.1)
                alk = st.number_input('Alkaline Phosphotase (IU/L)', 0.0,
                                      3000.0, 200.0)
            with c2:
                sgpt = st.number_input('Alamine Aminotransferase SGPT '
                                       '(IU/L)', 0.0, 2500.0, 30.0)
                sgot = st.number_input('Aspartate Aminotransferase SGOT '
                                       '(IU/L)', 0.0, 3000.0, 35.0)
                protein = st.number_input('Total Protiens (g dL)', 0.0,
                                          10.0, 6.7, 0.1)
                albumin = st.number_input('Albumin (g dL)', 0.0, 10.0, 3.3,
                                          0.1)
                rasio = st.number_input('Albumin and Globulin Ratio', 0.0,
                                        10.0, 1.0, 0.05)
            st.write('')
            prediksi_btn = st.form_submit_button('Jalankan Prediksi',
                                                use_container_width=True)

    with kanan:
        st.markdown('<div class="card"><h3>Hasil Klasifikasi</h3>'
                    '<p>Isi form di sebelah kiri lalu tekan tombol '
                    'jalankan prediksi. Hasil dan peluang kelas liver akan '
                    'tampil pada panel ini.</p></div>',
                    unsafe_allow_html=True)

    if prediksi_btn:
        pasien = pd.DataFrame(
            [[umur, 1 if gender == 'Male' else 0, tb, db, alk, sgpt, sgot,
              protein, albumin, rasio]], columns=fitur)
        pasien[fitur_numerik] = scaler.transform(pasien[fitur_numerik])
        pasien = pasien.to_numpy(dtype=float)
        hasil = model.predict(pasien)[0]
        proba = model.predict_proba(pasien)[0]
        peluang = proba[1] * 100

        teks_hasil = 'POSITIF LIVER' if hasil == 1 else 'NEGATIF, NON LIVER'
        st.write('')
        panel = ('<div class="card"><h3>Hasil Klasifikasi</h3>'
                 '<div style="font-size:28px;font-weight:800;color:'
                 '#151519;margin:6px 0 14px 0">' + teks_hasil + '</div>'
                 '<p style="margin:0 0 6px 0">Peluang kelas liver</p>'
                 '<div class="track"><div class="fill" style="width:'
                 + f'{peluang:.1f}' + '%"></div></div>'
                 '<p style="margin-top:8px">Kelas liver : '
                 + f'{peluang:.1f}' + ' persen, kelas non liver : '
                 + f'{proba[0] * 100:.1f}' + ' persen</p></div>')
        st.markdown(panel, unsafe_allow_html=True)
        st.info('Hasil prediksi bersifat edukasi dan bukan diagnosis '
                'medis resmi.')

# ================== FOOTER ==================
st.markdown(garis, unsafe_allow_html=True)
f1, f2, f3 = st.columns([1.4, 1, 1])
with f1:
    tag_logo_kaki = ('<div class="logochip" style="width:56px;'
                     'padding:6px;border-radius:16px"><img src="data:image/'
                     'png;base64,' + logo_b64 + '"></div>')
    st.markdown(tag_logo_kaki, unsafe_allow_html=True)
    st.markdown('<div class="footer">LiverCheck, alat bantu klasifikasi '
                'penyakit liver berbasis Indian Liver Patient Dataset. '
                'Hasil prediksi bersifat edukasi dan bukan diagnosis '
                'medis resmi.</div>', unsafe_allow_html=True)
with f2:
    st.markdown('<div class="footer"><strong>Halaman</strong><br>'
                'Beranda<br>Dataset<br>Prediksi</div>',
                unsafe_allow_html=True)
with f3:
    st.markdown('<div class="footer"><strong>Tautan</strong><br>'
                '<a href="https://www.kaggle.com/datasets/uciml/'
                'indian-liver-patient-records">Dataset di Kaggle</a><br>'
                'Streamlit<br>scikit learn</div>', unsafe_allow_html=True)
