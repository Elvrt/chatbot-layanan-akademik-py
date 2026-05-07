import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import re
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Pastikan path sesuai lokasi file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# ==========================================================
# 🔹 1. VARIABEL GLOBAL & KONFIGURASI FOLDER
# ==========================================================
model = None
df_kategori = None
df_knowledge = None

DATASET_DIR = 'file-dataset'
DATASET_FILE = os.path.join(DATASET_DIR, 'data_pelatihan.json')

# ==========================================================
# 🔹 2. INISIALISASI SASTRAWI & NORMALISASI
# ==========================================================
print("Memuat Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

# --- Memuat Kamus Normalisasi (Typo/Slang) ---
print("Memuat Kamus Normalisasi...")
normalisasi_dict = {}

# MENGARAHKAN PATH KE DALAM FOLDER file-dataset
NORM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_DIR, 'normalization.csv')

if os.path.exists(NORM_FILE):
    try:
        df_norm = pd.read_csv(NORM_FILE)
        # Ubah dataframe menjadi dictionary {kata_typo: kata_baku}
        normalisasi_dict = dict(zip(df_norm['kata_typo'], df_norm['kata_baku']))
        print(f"✅ Berhasil memuat {len(normalisasi_dict)} kata normalisasi dari folder {DATASET_DIR}.")
    except Exception as e:
        print(f"⚠️ Gagal memuat normalization.csv: {e}")
else:
    print(f"⚠️ File normalization.csv tidak ditemukan di {NORM_FILE}. Fitur normalisasi dilewati.")
# --------------------------------------------------------

def preprocess_text(text):
    text = str(text).lower()

    # --- Proses Normalisasi ---
    # Membersihkan tanda baca seperlunya agar pemisahan kata akurat
    text = re.sub(r'[^\w\s]', ' ', text)
    
    if normalisasi_dict:
        words = text.split()
        normalized_words = [normalisasi_dict.get(w, w) for w in words]
        text = " ".join(normalized_words)
    # ------------------------------------

    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# ==========================================================
# 🔹 3. LOAD CACHE (MODEL + DATA)
# ==========================================================
def load_cached_data():
    global model, df_kategori, df_knowledge
    try:
        if (
            os.path.exists('aegis_model.pkl') and 
            os.path.exists('kategori_cache.pkl') and
            os.path.exists('knowledge_cache.pkl')
        ):
            model = joblib.load('aegis_model.pkl')
            df_kategori = joblib.load('kategori_cache.pkl')
            df_knowledge = joblib.load('knowledge_cache.pkl')

            print("✅ Model & cache berhasil dimuat!")
        else:
            print("⚠️ Model belum tersedia. Silakan retrain dulu.")
    except Exception as e:
        print("❌ Error load cache:", e)

# Load saat server start
load_cached_data()

# ==========================================================
# 🔹 4. FUNGSI PELATIHAN (BISA DIPANGGIL DARI API & TERMINAL)
# ==========================================================
def latih_dari_file():
    global model, df_kategori, df_knowledge
    
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"File dataset tidak ditemukan di {DATASET_FILE}. Silakan kirim dari Laravel atau Postman terlebih dahulu.")
        
    print(f"🔄 Membaca data dari {DATASET_FILE}...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    # --- 1. LOAD DATA ---
    df_nlu = pd.DataFrame(data_json['dataset_nlu'])
    df_kategori_baru = pd.DataFrame(data_json['kategori']) 
    df_knowledge_baru = pd.DataFrame(data_json['knowledge'])

    # Fix nama kolom kategori agar sesuai dengan yang diharapkan oleh endpoint chat
    df_kategori_baru.columns = ['id_kategori', 'nama_kategori']

    # --- 2. PREPROCESS ---
    print("🔄 Preprocessing teks...")
    df_nlu['pertanyaan_bersih'] = df_nlu['pertanyaan_variasi'].apply(preprocess_text)
    df_knowledge_baru['pertanyaan_baku_bersih'] = df_knowledge_baru['pertanyaan'].apply(preprocess_text)

    X = df_nlu['pertanyaan_bersih'].tolist()
    Y = df_nlu['id_kategori'].tolist()

    # --- 3. TRAIN MODEL ---
    print("🔄 Training model...")
    new_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_df=0.20, min_df=2),
        MultinomialNB(alpha=0.5)
    )
    new_model.fit(X, Y)

    # --- 4. SIMPAN ---
    joblib.dump(new_model, 'aegis_model.pkl')
    joblib.dump(df_kategori_baru, 'kategori_cache.pkl')
    joblib.dump(df_knowledge_baru, 'knowledge_cache.pkl')

    # --- 5. LOAD ULANG ---
    load_cached_data()
    print("✅ Model berhasil dilatih dan disimpan!")

# ==========================================================
# 🔹 5. ENDPOINT CHATBOT
# ==========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    global model, df_kategori, df_knowledge

    if model is None or df_knowledge is None:
        return jsonify({'balasan': '🤖 Sistem belum dilatih.'}), 503

    data = request.json
    
    # Mencegah error jika input kosong atau hanya berisi spasi
    user_input = data.get('pesan', '').strip()

    if not user_input:
        return jsonify({'error': 'Pesan kosong'}), 400

    # Menghitung jumlah kata sebelum di-preprocess (untuk fallback logika)
    jumlah_kata = len(user_input.split())
    
    clean_input = preprocess_text(user_input)
    probabilitas = model.predict_proba([clean_input])[0]
    nilai_yakin = np.max(probabilitas)
    pred_id = model.classes_[np.argmax(probabilitas)]

    # --- LOGIKA FALLBACK JIKA SKOR YAKIN DI BAWAH 50% ---
    if nilai_yakin < 0.50:
        if jumlah_kata == 1:
            balasan_fallback = "🤖 Maaf, pertanyaanmu terlalu singkat sehingga AEGIS kurang paham. Bisa tolong jelaskan dengan kalimat yang lebih lengkap?"
        else:
            balasan_fallback = "🤖 Maaf, AEGIS belum memiliki informasi terkait pertanyaan tersebut. Coba gunakan kata kunci lain atau ketik 'menu' untuk melihat daftar informasi yang tersedia."
            
        return jsonify({
            'kategori_ditebak': 'TIDAK_DIKETAHUI',
            'tingkat_yakin': float(nilai_yakin),
            'balasan': balasan_fallback
        })

    nama_kat = df_kategori[df_kategori['id_kategori'] == pred_id]['nama_kategori'].values[0]
    jawaban_terkait = df_knowledge[df_knowledge['id_kategori'] == pred_id].copy()

    if jawaban_terkait.empty:
        return jsonify({
            'kategori_ditebak': nama_kat,
            'tingkat_yakin': float(nilai_yakin),
            'balasan': "🤖 Kategori ditemukan, tapi belum ada data jawaban."
        })

    jawaban_terkait['pertanyaan_baku_bersih'] = jawaban_terkait['pertanyaan'].apply(preprocess_text)
    vec_kb = TfidfVectorizer()
    kb_vec = vec_kb.fit_transform(jawaban_terkait['pertanyaan_baku_bersih'])
    user_vec = vec_kb.transform([clean_input])

    skor_mirip = cosine_similarity(user_vec, kb_vec).flatten()
    jawaban_terkait['skor_mirip'] = skor_mirip

    jawaban_terbaik = jawaban_terkait.sort_values(by='skor_mirip', ascending=False).head(1)
    row = jawaban_terbaik.iloc[0]

    # --- LOGIKA FALLBACK JIKA KATEGORI BENAR TAPI COSINE SIMILARITY 0 ---
    if row['skor_mirip'] == 0.0:
        if jumlah_kata == 1:
            balasan = "🤖 AEGIS mengerti topik yang kamu maksud, tapi pertanyaan 1 kata terlalu spesifik. Coba jabarkan lagi ya!"
        else:
            balasan = "🤖 AEGIS mengerti arah pertanyaanmu, tapi belum menemukan kecocokan jawaban spesifik di database."
    else:
        balasan = f"🤖 {row['jawaban']}"

    return jsonify({
        'kategori_ditebak': nama_kat,
        'tingkat_yakin': float(nilai_yakin),
        'balasan': balasan
    })

# ==========================================================
# 🔹 6. ENDPOINT RETRAIN (DARI LARAVEL / POSTMAN)
# ==========================================================
@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        print("🔄 Menerima payload JSON dari Frontend/API...")
        data_json = request.json

        if not data_json:
            return jsonify({'status': 'error', 'pesan': 'Data JSON kosong'}), 400

        # Buat folder jika belum ada
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            print(f"📁 Folder '{DATASET_DIR}' berhasil dibuat.")

        # --- SIMPAN JSON KE FOLDER ---
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, indent=4)
        print(f"💾 File JSON berhasil disimpan di {DATASET_FILE}")

        # --- PANGGIL FUNGSI PELATIHAN ---
        latih_dari_file()

        return jsonify({
            'status': 'success',
            'pesan': '✅ Data berhasil disimpan ke folder dan Model berhasil diretrain!'
        })

    except Exception as e:
        print("❌ Error retrain:", str(e))
        return jsonify({
            'status': 'error',
            'pesan': str(e)
        }), 500

# ==========================================================
# 🔹 7. ENDPOINT STATUS
# ==========================================================
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'online'}), 200

# ==========================================================
# 🔹 8. RUN SERVER ATAU JALANKAN DARI TERMINAL
# ==========================================================
if __name__ == '__main__':
    # Jika dijalankan dengan perintah: python app.py train
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("🚀 Menjalankan pelatihan manual dari terminal...")
        try:
            latih_dari_file()
        except Exception as e:
            print(f"❌ Gagal melatih: {e}")
    else:
        # Jika dijalankan normal: python app.py
        app.run(debug=True, host='127.0.0.1', port=5000)