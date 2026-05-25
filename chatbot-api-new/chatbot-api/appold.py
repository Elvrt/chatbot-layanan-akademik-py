import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- MEMASTIKAN LOKASI FOLDER BENAR ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# ==========================================================
# 1. PERSIAPAN MODEL & DATABASE (Hanya jalan 1x saat server nyala)
# ==========================================================
print("Memuat model AEGIS 2.0 dan sistem NLU... (Mohon tunggu)")

# Load Sastrawi
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess_text(text):
    text = str(text).lower()
    text = stopword.remove(text)
    return stemmer.stem(text)

# Load Model yang baru saja kamu cetak
model = joblib.load('aegis_model.pkl')

# Load Database Jawaban & Kategori
df_kategori = pd.read_csv('kategori-intent.csv', sep=';')
df_kategori.columns = ['id_kategori', 'nama_kategori']

df_knowledge = pd.read_csv('knowledge.csv', sep=';')
# Bersihkan pertanyaan baku di database sejak awal agar balasan chat super cepat
df_knowledge['pertanyaan_baku_bersih'] = df_knowledge['pertanyaan_baku'].apply(preprocess_text)

print("✅ SERVER API AEGIS 2.0 SIAP DIGUNAKAN!")

# ==========================================================
# 2. ENDPOINT API (URL yang akan dipanggil oleh Laravel)
# ==========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    # Menangkap pesan dari Laravel
    data = request.json
    user_input = data.get('pesan')

    if not user_input:
        return jsonify({'error': 'Pesan tidak boleh kosong'}), 400

    # 1. Bersihkan teks input
    clean_input = preprocess_text(user_input)
    
    # 2. Tebak Kategori dengan Naive Bayes
    probabilitas = model.predict_proba([clean_input])[0]
    nilai_yakin = np.max(probabilitas)
    pred_id = model.classes_[np.argmax(probabilitas)]
    
    # Threshold: Jika bot sangat tidak yakin
    if nilai_yakin < 0.30:
        return jsonify({
            'kategori_ditebak': 'TIDAK_DIKETAHUI',
            'tingkat_yakin': float(nilai_yakin),
            'balasan': "🤖 Maaf, AEGIS belum mengerti pertanyaan tersebut. Coba gunakan kata kunci lain seputar akademik kampus."
        })
    
    # 3. Cari Jawaban Terbaik (Cosine Similarity)
    nama_kat = df_kategori[df_kategori['id_kategori'] == pred_id]['nama_kategori'].values[0]
    jawaban_terkait = df_knowledge[df_knowledge['id_kategori'] == pred_id].copy()
    
    vec_kb = CountVectorizer()
    kb_vec = vec_kb.fit_transform(jawaban_terkait['pertanyaan_baku_bersih'])
    user_vec = vec_kb.transform([clean_input])
    
    skor_mirip = cosine_similarity(user_vec, kb_vec).flatten()
    jawaban_terkait['skor_mirip'] = skor_mirip
    jawaban_terbaik = jawaban_terkait.sort_values(by='skor_mirip', ascending=False).head(1)
    
 # 4. Rangkai Balasan Akhir
    row = jawaban_terbaik.iloc[0]
    balasan = ""
    
    if row['skor_mirip'] == 0.0:
        # Menghapus tag kategori agar terlihat lebih natural
        balasan = "🤖 Mohon maaf, saya mengerti arah pertanyaanmu, tapi saya belum punya jawaban spesifiknya di database saat ini."
    else:
        # Menghapus tag kategori dan persentase yakin, langsung to the point ke jawaban
        balasan = f"🤖 {row['jawaban']}"
        
    # Kembalikan jawaban ke Laravel
    return jsonify({
        'kategori_ditebak': nama_kat,        # Data ini tetap dikirim ke Laravel untuk log/debugging
        'tingkat_yakin': float(nilai_yakin), # Data ini tetap dikirim ke Laravel untuk log/debugging
        'balasan': balasan                     # Teks balasan sekarang sudah bersih!
    })
# Menjalankan server di localhost port 5000
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)