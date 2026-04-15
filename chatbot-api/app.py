import os
import joblib
import pandas as pd
import numpy as np
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
# 🔹 1. INISIALISASI SASTRAWI
# ==========================================================
print("Memuat Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess_text(text):
    text = str(text).lower()
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# ==========================================================
# 🔹 2. VARIABEL GLOBAL
# ==========================================================
model = None
df_kategori = None
df_knowledge = None

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
# 🔹 4. ENDPOINT CHATBOT
# ==========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    global model, df_kategori, df_knowledge

    if model is None or df_knowledge is None:
        return jsonify({'balasan': '🤖 Sistem belum dilatih.'}), 503

    data = request.json
    user_input = data.get('pesan')

    if not user_input:
        return jsonify({'error': 'Pesan kosong'}), 400

    # --- 1. PREPROCESS INPUT ---
    clean_input = preprocess_text(user_input)

    # --- 2. PREDIKSI INTENT ---
    probabilitas = model.predict_proba([clean_input])[0]
    nilai_yakin = np.max(probabilitas)
    pred_id = model.classes_[np.argmax(probabilitas)]

    # --- 3. THRESHOLD ---
    if nilai_yakin < 0.30:
        return jsonify({
            'kategori_ditebak': 'TIDAK_DIKETAHUI',
            'tingkat_yakin': float(nilai_yakin),
            'balasan': "🤖 Maaf, AEGIS belum mengerti pertanyaan tersebut."
        })

    # --- 4. AMBIL KATEGORI ---
    nama_kat = df_kategori[
        df_kategori['id_kategori'] == pred_id
    ]['nama_kategori'].values[0]

    # --- 5. FILTER KNOWLEDGE ---
    jawaban_terkait = df_knowledge[
        df_knowledge['id_kategori'] == pred_id
    ].copy()

    if jawaban_terkait.empty:
        return jsonify({
            'kategori_ditebak': nama_kat,
            'tingkat_yakin': float(nilai_yakin),
            'balasan': "🤖 Kategori ditemukan, tapi belum ada data jawaban."
        })

    # --- 6. PREPROCESS KNOWLEDGE ---
    jawaban_terkait['pertanyaan_baku_bersih'] = jawaban_terkait['pertanyaan'].apply(preprocess_text)

    # --- 7. TF-IDF + COSINE SIMILARITY ---
    vec_kb = TfidfVectorizer()
    kb_vec = vec_kb.fit_transform(jawaban_terkait['pertanyaan_baku_bersih'])
    user_vec = vec_kb.transform([clean_input])

    skor_mirip = cosine_similarity(user_vec, kb_vec).flatten()
    jawaban_terkait['skor_mirip'] = skor_mirip

    # --- 8. AMBIL JAWABAN TERBAIK ---
    jawaban_terbaik = jawaban_terkait.sort_values(
        by='skor_mirip', ascending=False
    ).head(1)

    row = jawaban_terbaik.iloc[0]

    if row['skor_mirip'] == 0.0:
        balasan = "🤖 Saya mengerti kategorinya, tapi belum ada jawaban spesifik."
    else:
        balasan = f"🤖 {row['jawaban']}"

    return jsonify({
        'kategori_ditebak': nama_kat,
        'tingkat_yakin': float(nilai_yakin),
        'balasan': balasan
    })

# ==========================================================
# 🔹 5. ENDPOINT RETRAIN (DARI LARAVEL)
# ==========================================================
@app.route('/api/retrain', methods=['POST'])
def retrain():
    global model, df_kategori, df_knowledge

    try:
        print("🔄 Menerima data retrain dari Laravel...")

        data_json = request.json

        # --- 1. LOAD DATA ---
        df_nlu = pd.DataFrame(data_json['dataset_nlu'])
        df_kategori_baru = pd.DataFrame(data_json['intents'])
        df_knowledge_baru = pd.DataFrame(data_json['knowledge_bases'])

        # Fix nama kolom kategori
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

        return jsonify({
            'status': 'success',
            'pesan': '✅ Model berhasil diretrain!'
        })

    except Exception as e:
        print("❌ Error retrain:", str(e))
        return jsonify({
            'status': 'error',
            'pesan': str(e)
        }), 500

# ==========================================================
# 🔹 6. RUN SERVER
# ==========================================================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)