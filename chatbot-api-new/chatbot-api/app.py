import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import re
import NBmodel
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

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
# 🔹 2. ENDPOINT CHATBOT
# ==========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    if NBmodel.model is None or NBmodel.df_knowledge is None:
        return jsonify({'balasan': '🤖 Sistem belum dilatih.'}), 503

    data = request.json

    # Mencegah error jika input kosong atau hanya berisi spasi
    user_input = data.get('pesan', '').strip()

    if not user_input:
        return jsonify({'error': 'Pesan kosong'}), 400

    # Menghitung jumlah kata sebelum di-preprocess (untuk fallback logika)
    jumlah_kata = len(user_input.split())
    
    clean_input = NBmodel.preprocess_text(user_input)
    probabilitas = NBmodel.model.predict_proba([clean_input])[0]
    nilai_yakin = np.max(probabilitas)
    pred_id = NBmodel.model.classes_[np.argmax(probabilitas)]

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

    nama_kat = NBmodel.df_kategori[NBmodel.df_kategori['id_kategori'] == pred_id]['nama_kategori'].values[0]
    jawaban_terkait = NBmodel.df_knowledge[NBmodel.df_knowledge['id_kategori'] == pred_id].copy()

    if jawaban_terkait.empty:
        return jsonify({
            'kategori_ditebak': nama_kat,
            'tingkat_yakin': float(nilai_yakin),
            'balasan': "🤖 Kategori ditemukan, tapi belum ada data jawaban."
        })

    jawaban_terkait['pertanyaan_baku_bersih'] = jawaban_terkait['pertanyaan'].apply(NBmodel.preprocess_text)
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
# 🔹 3. ENDPOINT RETRAIN (DARI LARAVEL / POSTMAN)
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
        NBmodel.latih_dari_file()

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
# 🔹 4. ENDPOINT STATUS
# ==========================================================
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'online'}), 200

# ==========================================================
# 🔹 5. RUN SERVER ATAU JALANKAN DARI TERMINAL
# ==========================================================
if __name__ == '__main__':
    # Jika dijalankan dengan perintah: python app.py train
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("🚀 Menjalankan pelatihan manual dari terminal...")
        try:
            NBmodel.latih_dari_file()
        except Exception as e:
            print(f"❌ Gagal melatih: {e}")
    else:
        # Jika dijalankan normal: python app.py
        app.run(debug=True, host='127.0.0.1', port=5000)