"""
debug_overlap.py — Analisis kata overlap antar dua kategori
============================================================
Jalankan dari terminal:
    python debug_overlap.py

Preprocessing yang dipakai IDENTIK dengan app.py:
  lowercase → bersihkan tanda baca → normalisasi → 
  stopword Sastrawi → custom stopword → stemming
"""

import os
import json
import re
import collections

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==========================================================
# Konfigurasi — ubah dua kategori yang ingin dibandingkan
# ==========================================================
KAT_A = 4
KAT_B = 8

DATASET_FILE = os.path.join('file-dataset', 'data_pelatihan.json')
NORM_FILE    = os.path.join('file-dataset', 'normalization.csv')

# ==========================================================
# Inisialisasi NLP — identik dengan app.py
# ==========================================================
stemmer  = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

normalisasi_dict = {}
if os.path.exists(NORM_FILE):
    try:
        df_norm = pd.read_csv(NORM_FILE)
        normalisasi_dict = dict(zip(df_norm['kata_typo'], df_norm['kata_baku']))
        print(f"✅ {len(normalisasi_dict)} kata normalisasi dimuat.")
    except Exception as e:
        print(f"⚠️ Gagal memuat normalization.csv: {e}")
else:
    print("⚠️ normalization.csv tidak ditemukan — normalisasi dilewati.")

# Custom stopword — identik dengan app.py
CUSTOM_STOPWORDS = {
    'min', 'mimin', 'admin', 'kak', 'bang', 'mas', 'mba', 'orang',
    'halo', 'hai', 'dong', 'ya', 'yuk', 'nih',
    'jti',
    'apa', 'bagaimana',
    'cek',
    'terima',
    'baik', 'ada', 'aktif',
}

# ==========================================================
# Fungsi preprocessing — identik dengan app.py
# ==========================================================
def preprocess(text: str) -> set[str]:
    """
    Mengembalikan set kata setelah full preprocessing.
    Menggunakan set agar setiap kata dihitung sekali per kalimat.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    if normalisasi_dict:
        words = text.split()
        text  = " ".join(normalisasi_dict.get(w, w) for w in words)

    text  = stopword.remove(text)
    words = [w for w in text.split() if w not in CUSTOM_STOPWORDS]
    text  = " ".join(words)
    text  = stemmer.stem(text)

    return set(text.split())

# ==========================================================
# Load dataset
# ==========================================================
with open(DATASET_FILE, encoding='utf-8') as f:
    data = json.load(f)

# ==========================================================
# Hitung frekuensi kata per kategori
# ==========================================================
words = {KAT_A: collections.Counter(), KAT_B: collections.Counter()}

for row in data['dataset_nlu']:
    if row['id_kategori'] in (KAT_A, KAT_B):
        for w in preprocess(row['pertanyaan_variasi']):
            words[row['id_kategori']][w] += 1

# ==========================================================
# Tampilkan hasil overlap
# ==========================================================
overlap = set(words[KAT_A]) & set(words[KAT_B])

print(f"\nKata overlap kat-{KAT_A} & kat-{KAT_B}: {len(overlap)} kata")

if overlap:
    # Urutkan berdasarkan total frekuensi gabungan — yang paling sering muncul di atas
    overlap_sorted = sorted(
        overlap,
        key=lambda w: words[KAT_A][w] + words[KAT_B][w],
        reverse=True,
    )

    print(f"\n{'Kata':<20} {'kat-' + str(KAT_A):>8} {'kat-' + str(KAT_B):>8}  {'total':>8}")
    print("─" * 48)
    for w in overlap_sorted:
        total = words[KAT_A][w] + words[KAT_B][w]
        print(f"  {w:<18} {words[KAT_A][w]:>8} {words[KAT_B][w]:>8}  {total:>8}")
else:
    print("✅ Tidak ada kata overlap — kedua kategori sudah terdifferensiasi dengan baik.")