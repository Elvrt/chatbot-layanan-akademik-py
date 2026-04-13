import os
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

from sklearn.metrics.pairwise import cosine_similarity

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
df_knowledge = pd.read_csv('knowledge.csv', sep=';')

# TAMBAHKAN BARIS INI UNTUK MEMPERBAIKI ERROR:
df_kategori.columns = ['id_kategori', 'nama_kategori']

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

# =====================================================================
# --- 7. PELATIHAN FINAL & FITUR TANYA JAWAB (CHATBOT) ---
# =====================================================================

print("\nMelatih model final menggunakan seluruh dataset...")
model.fit(X, Y)
print("Model AEGIS 2.0 siap digunakan!")

def get_response(user_input):
    # 1. Bersihkan input user pakai Sastrawi
    clean_input = preprocess_text(user_input)

    # 2. Cek Probabilitas (Keyakinan) Model Naive Bayes
    probabilitas = model.predict_proba([clean_input])[0]
    nilai_yakin = np.max(probabilitas)
    pred_id = model.classes_[np.argmax(probabilitas)]

    # Threshold: Jika bot sangat tidak yakin (< 30%)
    if nilai_yakin < 0.30:
        return "🤖 Maaf, AEGIS belum mengerti pertanyaan tersebut. Coba gunakan kata kunci yang lebih jelas."

    # 3. Dapatkan Nama Kategori
    nama_kat = df_kategori[df_kategori['id_kategori'] == pred_id]['nama_kategori'].values[0]

    # 4. Filter knowledge base sesuai kategori
    jawaban_terkait = df_knowledge[df_knowledge['id_kategori'] == pred_id].copy()

    # --- PERBAIKAN UTAMA DI SINI ---
    # Kita bersihkan juga pertanyaan di database agar "nyambung" dengan input user
    jawaban_terkait['pertanyaan_baku_bersih'] = jawaban_terkait['pertanyaan_baku'].apply(preprocess_text)

    # 5. Cosine Similarity (Pencocokan Kalimat)
    vec_kb = TfidfVectorizer()
    # Menggunakan pertanyaan yang SUDAH BERSIH untuk dihitung
    kb_tfidf = vec_kb.fit_transform(jawaban_terkait['pertanyaan_baku_bersih'])
    user_tfidf = vec_kb.transform([clean_input])

    skor_mirip = cosine_similarity(user_tfidf, kb_tfidf).flatten()
    jawaban_terkait['skor_mirip'] = skor_mirip

    # Ambil 1 JAWABAN PALING MIRIP (Bukan 2, agar langsung to the point)
    jawaban_terbaik = jawaban_terkait.sort_values(by='skor_mirip', ascending=False).head(1)

    # 6. Rangkai Balasan Bot
    reply = f"🤖 [Kategori: {nama_kat} | Yakin: {nilai_yakin*100:.1f}%]\n"

    for index, row in jawaban_terbaik.iterrows():
        # Jika sama sekali tidak ada kata yang cocok (Skor 0)
        if row['skor_mirip'] == 0.0:
            reply += "Saya mendeteksi ini tentang layanan tersebut, tapi mohon maaf saya belum punya jawaban spesifiknya di database."
        else:
            reply += f"💡 {row['jawaban']}\n"

    return reply

# --- 8. LOOPING CHAT INTERAKTIF ---
print("\n" + "="*50)
print("Selamat datang di AEGIS 2.0 (Ketik 'keluar' atau 'exit' untuk berhenti)")
print("="*50)

while True:
    teks_user = input("\nKamu: ")

    if teks_user.lower() in ['keluar', 'exit', 'quit']:
        print("AEGIS 2.0: Terima kasih! Sampai jumpa lagi.")
        break

    balasan_bot = get_response(teks_user)
    print(f"\nAEGIS: \n{balasan_bot}")