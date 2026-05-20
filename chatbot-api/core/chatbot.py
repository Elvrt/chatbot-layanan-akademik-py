"""
chatbot.py
----------
Tanggung jawab TUNGGAL: orkestrasi alur lengkap chatbot.

Menerima input mentah dari user → kembalikan dict respons siap pakai.
app.py hanya perlu memanggil get_response() dan retrain().

Tidak ada logika HTTP, model, atau preprocessing di sini.
"""

import os
import json

from core.preprocessor import preprocess, reload_normalisasi
from core.classifier   import predict, get_nama_kategori, train, train_dari_file, is_ready
from core              import retriever

# ==========================================================
# 🔹 KONFIGURASI PATH
# ==========================================================
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR  = os.path.join(_BASE_DIR, '..', 'file-dataset')
_DATASET_FILE = os.path.join(_DATASET_DIR, 'data_pelatihan.json')

# Kata pertama kalimat yang mengindikasikan sapaan.
# Jika terdeteksi, custom stopwords dilewati saat preprocessing query
# agar token sapaan ('halo', 'hai', dll.) tidak hilang sebelum prediksi.
_KATA_AWAL_SAPAAN: set = {
    'halo', 'hai', 'hei', 'hey', 'hello', 'hi',
    'permisi', 'punten', 'assalamualaikum', 'waalaikumsalam',
    'salam', 'pagi', 'siang', 'sore', 'malam', 'selamat',
}

# ==========================================================
# 🔹 FUNGSI PUBLIK — CHAT
# ==========================================================

def get_response(user_input: str) -> dict:
    """
    Proses input user dan kembalikan respons chatbot.

    Parameters
    ----------
    user_input : teks mentah dari user (belum di-preprocess)

    Returns
    -------
    dict dengan key:
        - balasan       : str   → teks jawaban untuk ditampilkan ke user
        - kategori      : str   → nama kategori yang diprediksi
        - tingkat_yakin : float → confidence Naive Bayes (0.0 - 1.0)
    """
    if not is_ready():
        return _build_response(
            balasan='🤖 Sistem belum dilatih. Silakan hubungi administrator.',
            kategori='SISTEM_ERROR',
            yakin=0.0,
        )

    jumlah_kata        = len(user_input.strip().split())
    kata_pertama       = user_input.strip().split()[0].lower() if user_input.strip() else ''
    kemungkinan_sapaan = kata_pertama in _KATA_AWAL_SAPAAN

    clean_input = preprocess(
        user_input,
        pakai_fuzzy=True,
        skip_custom_sw=kemungkinan_sapaan,
    )

    # Threshold lebih rendah untuk sapaan pendek —
    # input 1 kata yang terdeteksi sapaan tidak perlu confidence tinggi
    if kemungkinan_sapaan and jumlah_kata <= 2:
        threshold = 0.30
    else:
        threshold = 0.50

    id_kategori, confidence = predict(clean_input, threshold=threshold)

    # Fallback: confidence rendah
    if id_kategori is None:
        if jumlah_kata == 1 and not kemungkinan_sapaan:
            # Hanya tampilkan "terlalu singkat" untuk bukan sapaan
            balasan = (
                "🤖 Maaf, pertanyaanmu terlalu singkat sehingga AEGIS kurang paham. "
                "Bisa tolong jelaskan dengan kalimat yang lebih lengkap?"
            )
        else:
            balasan = (
                "🤖 Maaf, AEGIS belum memiliki informasi terkait pertanyaan tersebut. "
                "Coba gunakan kata kunci lain atau ketik 'menu' untuk melihat "
                "daftar informasi yang tersedia."
            )
        return _build_response(balasan=balasan, kategori='TIDAK_DIKETAHUI', yakin=confidence)

    nama_kat = get_nama_kategori(id_kategori) or id_kategori

    # --- CARI JAWABAN (TF-IDF Cosine via retriever) ---
    best_row, skor = retriever.find_best_answer(clean_input, id_kategori)

    # --- FALLBACK: kategori ditemukan tapi tidak ada data knowledge ---
    if best_row is None:
        return _build_response(
            balasan='🤖 Kategori ditemukan, tapi belum ada data jawaban.',
            kategori=nama_kat,
            yakin=confidence,
        )

    # --- FALLBACK: kategori benar tapi cosine similarity = 0 ---
    if skor == 0.0:
        if jumlah_kata == 1:
            balasan = (
                "🤖 AEGIS mengerti topik yang kamu maksud, tapi pertanyaan 1 kata "
                "terlalu spesifik. Coba jabarkan lagi ya!"
            )
        else:
            balasan = (
                "🤖 AEGIS mengerti arah pertanyaanmu, tapi belum menemukan "
                "kecocokan jawaban spesifik di database."
            )
        return _build_response(balasan=balasan, kategori=nama_kat, yakin=confidence)

    # --- JAWABAN DITEMUKAN ---
    return _build_response(
        balasan=f"🤖 {best_row['jawaban']}",
        kategori=nama_kat,
        yakin=confidence,
    )


# ==========================================================
# 🔹 FUNGSI PUBLIK — RETRAIN
# ==========================================================

def retrain(data_json: dict):
    """
    Simpan payload JSON ke file lalu latih ulang model.
    Dipanggil oleh app.py saat menerima POST /api/retrain.

    Parameters
    ----------
    data_json : dict payload dari Laravel / Postman
    """
    if not data_json:
        raise ValueError("Data JSON kosong.")

    os.makedirs(_DATASET_DIR, exist_ok=True)
    with open(_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)
    print(f"[chatbot] 💾 Dataset disimpan ke {_DATASET_FILE}")

    reload_normalisasi()
    train(data_json, jalankan_evaluasi=True)

    print("[chatbot] ✅  Retrain selesai.")


def retrain_dari_file():
    """Retrain dari file dataset yang sudah ada. Berguna untuk training manual."""
    reload_normalisasi()
    train_dari_file(jalankan_evaluasi=True)
    print("[chatbot] ✅  Retrain dari file selesai.")


# ==========================================================
# 🔹 FUNGSI INTERNAL
# ==========================================================

def _build_response(balasan: str, kategori: str, yakin: float) -> dict:
    return {
        'balasan':       balasan,
        'kategori':      kategori,
        'tingkat_yakin': round(yakin, 4),
    }