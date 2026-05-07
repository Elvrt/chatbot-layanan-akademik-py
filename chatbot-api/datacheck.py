import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Pastikan path sesuai lokasi file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==========================================================
# 1. INISIALISASI SASTRAWI
# ==========================================================
print("⏳ Memuat Sastrawi (Stopword)...")
stopword = StopWordRemoverFactory().create_stop_word_remover()
# Stemmer sengaja dimatikan agar proses audit berjalan cepat
# Jika ingin lebih akurat (tapi lebih lambat), nyalakan stemmer di bawah ini:
# stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = stopword.remove(text)
    # text = stemmer.stem(text)
    return text

# ==========================================================
# 2. LOAD DATA DARI JSON (Sesuai arsitektur sebelumnya)
# ==========================================================
DATASET_FILE = os.path.join('file-dataset', 'data_pelatihan.json')

if not os.path.exists(DATASET_FILE):
    print(f"❌ File {DATASET_FILE} tidak ditemukan!")
    print("Silakan lakukan Retrain dari Laravel/Postman terlebih dahulu agar file ini terbentuk.")
    exit()

print(f"🔄 Membaca data dari {DATASET_FILE}...")
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    data_json = json.load(f)

df_nlu = pd.DataFrame(data_json['dataset_nlu'])
df_knowledge = pd.DataFrame(data_json['knowledge'])

# ==========================================================
# 3. PROSES AUDIT GAP
# ==========================================================
print("🔍 Memulai proses audit data NLU vs Knowledge Base...")

THRESHOLD_COSINE = 0.30  # Batas minimum skor dianggap "ada jawaban"
hasil_gap = []

for index, row in df_nlu.iterrows():
    pertanyaan = row['pertanyaan_variasi']
    id_kategori = row['id_kategori']
    clean_input = preprocess_text(pertanyaan)

    # Filter knowledge berdasarkan kategori
    kb_kategori = df_knowledge[df_knowledge['id_kategori'] == id_kategori].copy()

    if kb_kategori.empty:
        # Kategori sama sekali tidak punya knowledge
        hasil_gap.append({
            'pertanyaan_nlu' : pertanyaan,
            'id_kategori'    : id_kategori,
            'skor_tertinggi' : 0.0,
            'status'         : '❌ Kategori tidak ada di knowledge'
        })
        continue

    # Preprocess teks di Knowledge Base untuk kategori ini
    kb_kategori['pertanyaan_bersih'] = kb_kategori['pertanyaan'].apply(preprocess_text)
    
    # Hitung cosine similarity
    vec = TfidfVectorizer()
    
    try:
        kb_vec = vec.fit_transform(kb_kategori['pertanyaan_bersih'])
        user_vec = vec.transform([clean_input])
        
        skor = cosine_similarity(user_vec, kb_vec).flatten()
        skor_max = skor.max() if len(skor) > 0 else 0.0

        if skor_max < THRESHOLD_COSINE:
            hasil_gap.append({
                'pertanyaan_nlu' : pertanyaan,
                'id_kategori'    : id_kategori,
                'skor_tertinggi' : round(skor_max, 2),
                'status'         : '⚠️ Tidak ada jawaban yang cocok'
            })
    except ValueError:
        # Menangkap error jika semua kata terkena stopword dan kosong
        hasil_gap.append({
            'pertanyaan_nlu' : pertanyaan,
            'id_kategori'    : id_kategori,
            'skor_tertinggi' : 0.0,
            'status'         : '⚠️ Teks terlalu pendek/hanya stopword'
        })

# ==========================================================
# 4. TAMPILKAN & SIMPAN HASIL
# ==========================================================
df_gap = pd.DataFrame(hasil_gap)

print("\n" + "="*50)
if df_gap.empty:
    print("🎉 SELAMAT! Tidak ada gap sama sekali.")
    print("Semua variasi pertanyaan NLU memiliki jawaban yang cocok di Knowledge Base.")
else:
    print(f"🚨 Total gap ditemukan: {len(df_gap)} dari {len(df_nlu)} data latih")
    print("="*50)
    print(df_gap.to_string(index=False))

    # Simpan ke CSV di dalam folder file-dataset
    output_path = os.path.join('file-dataset', 'hasil_audit_gap.csv')
    df_gap.to_csv(output_path, index=False)
    print("\n✅ Hasil audit disimpan ke:", output_path)