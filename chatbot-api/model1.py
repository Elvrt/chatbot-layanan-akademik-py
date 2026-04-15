import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. INISIALISASI SASTRAWI ---
print("Memuat Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess_text(text):
    text = str(text).lower()
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# --- 2. MEMBACA FILE CSV ---
df_nlu = pd.read_csv('dataset-nlu-clean.csv', sep=';')
df_kategori = pd.read_csv('kategori-intent.csv', sep=';')

# --- 3. PRE-PROCESSING DATA ---
print("Menerapkan Sastrawi pada dataset NLU...")
df_nlu['pertanyaan_bersih'] = df_nlu['pertanyaan_variasi'].apply(preprocess_text)

X = df_nlu['pertanyaan_bersih'].tolist()
Y = df_nlu['id_kategori'].tolist()

# --- 4. PERSIAPAN MODEL ---
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.20, min_df=2),
    MultinomialNB(alpha=0.5)
)

# --- 5. EVALUASI LENGKAP DENGAN 10-FOLD ---
K = 10
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

print(f"\nMelakukan Evaluasi {K}-Fold Cross Validation...")
# 1. Menghitung Akurasi
scores = cross_val_score(model, X, Y, cv=skf, scoring='accuracy')
print(f"Rata-rata Akurasi Model: {np.mean(scores) * 100:.2f}%\n")

# 2. Mendapatkan tebakan prediksi untuk seluruh data uji di tiap fold
print("Menghitung Precision, Recall, dan F1-Score...")
Y_pred = cross_val_predict(model, X, Y, cv=skf)

# Ambil daftar nama kategori dari dataframe agar hasil laporannya mudah dibaca
daftar_nama_kategori = df_kategori['nama_kategori'].tolist()

# 3. Mencetak Classification Report
print("\n=== LAPORAN EVALUASI MODEL (CLASSIFICATION REPORT) ===")
print(classification_report(Y, Y_pred, target_names=daftar_nama_kategori))

# --- 6. MEMBUAT CONFUSION MATRIX ---
print("\nMenghasilkan grafik Confusion Matrix...")

# Menghitung nilai matriks dari label asli (Y) dan label tebakan (Y_pred)
cm = confusion_matrix(Y, Y_pred)

# Membuat objek visualisasi matriks
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=daftar_nama_kategori
)

# Mengatur ukuran jendela grafik agar luas (10x8 inch)
fig, ax = plt.subplots(figsize=(10, 8))

# Menggambar grafik (menggunakan tema warna Biru dan memiringkan teks agar terbaca)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

# Menambahkan judul dan merapikan layout
plt.title("Confusion Matrix - AEGIS 2.0 (10-Fold CV)", fontsize=14)
plt.tight_layout()

# Menampilkan grafik ke layar
plt.show()