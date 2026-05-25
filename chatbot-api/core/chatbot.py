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
# 🔹 FALLBACK PER KATEGORI
# Digunakan saat kategori berhasil diprediksi tapi jawaban
# spesifik tidak ditemukan di knowledge base (cosine rendah).
# Memberikan arahan yang relevan per topik daripada pesan
# generik "tidak ada informasi".
# ==========================================================
_FALLBACK_KATEGORI: dict = {
    'STATUS_AKADEMIK': (
        "🤖 AEGIS mengenali pertanyaanmu tentang status akademik, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"syarat cuti akademik\"\n"
        "• \"cara lapor aktif setelah terminal\"\n"
        "• \"batas maksimal cuti kuliah\"\n\n"
        "Atau hubungi Admin Akademik Jurusan secara langsung."
    ),
    'ADMINISTRASI_SURAT': (
        "🤖 AEGIS mengenali pertanyaanmu tentang persuratan, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"cara minta surat keterangan kuliah\"\n"
        "• \"prosedur cetak KHS digital\"\n"
        "• \"syarat pengajuan surat bebas tanggungan\"\n\n"
        "Atau ajukan pertanyaan ke helpdesk akademik di "
        "https://helpakademik.polinema.ac.id/"
    ),
    'KEUANGAN_UKT': (
        "🤖 AEGIS mengenali pertanyaanmu tentang keuangan/UKT, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"cara mengajukan keringanan UKT mahasiswa\"\n"
        "• \"syarat angsuran UKT\"\n"
        "• \"berkas keringanan UKT\"\n\n"
        "Atau hubungi Bagian Keuangan Polinema."
    ),
    'KELULUSAN_WISUDA': (
        "🤖 AEGIS mengenali pertanyaanmu tentang kelulusan/wisuda, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"syarat daftar wisuda\"\n"
        "• \"prosedur ijazah hilang\"\n"
        "• \"cara legalisir ijazah\"\n\n"
        "Atau hubungi Layanan Akademik Pusat di Gedung AA."
    ),
    'DATA_MAHASISWA': (
        "🤖 AEGIS mengenali pertanyaanmu tentang data mahasiswa, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"cara verifikasi data di SIAKAD\"\n"
        "• \"NIK orang tua meninggal diisi apa\"\n"
        "• \"prosedur perubahan data mahasiswa\"\n\n"
        "Atau hubungi Admin Akademik Jurusan."
    ),
    'KEMAHASISWAAN': (
        "🤖 AEGIS mengenali pertanyaanmu tentang kemahasiswaan, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"syarat daftar beasiswa\"\n"
        "• \"cara input prestasi di SIAKAD\"\n"
        "• \"prosedur klaim asuransi mahasiswa\"\n\n"
        "Atau hubungi Bagian Kemahasiswaan JTI."
    ),
    'AKADEMIK_UMUM': (
        "🤖 AEGIS mengenali pertanyaanmu tentang akademik umum, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"jadwal kuliah semester ini\"\n"
        "• \"cara cetak KHS\"\n"
        "Atau akses informasi di portal akademik JTI."
    ),
    'INFORMASI_DOSEN': (
        "🤖 AEGIS mengenali pertanyaanmu tentang informasi dosen, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan menyebutkan nama dosen, contoh:\n"
        "• \"nomor WA Bu Ana\"\n"
        "• \"kontak Pak Budi\"\n"
        "• \"ruangan dosen 3 ada siapa saja\"\n"
    ),
    'TINGKAT_AKHIR': (
        "🤖 AEGIS mengenali pertanyaanmu tentang skripsi/TA, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"syarat pendaftaran sidang akhir\"\n"
        "• \"minimal bimbingan sebelum sidang\"\n"
        "• \"cara daftar sempro\"\n\n"
        "Atau konsultasikan dengan dosen pembimbing atau Admin Jurusan."
    ),
    'MAGANG/PKL': (
        "🤖 AEGIS mengenali pertanyaanmu tentang magang/PKL, "
        "namun belum menemukan jawaban spesifik di database.\n\n"
        "Coba tanyakan dengan kata kunci seperti:\n"
        "• \"alur PKL dari awal sampai selesai\"\n"
        "• \"cara cek dosen pembimbing magang\"\n"
        "• \"lapor masalah darurat PKL ke siapa\"\n\n"
        "Atau hubungi Admin Akademik Jurusan."
    ),
    'SAPAAN': (
        "🤖 Halo! 👋 AEGIS siap membantu informasi akademik JTI Polinema. "
        "Silakan ketik pertanyaanmu atau ketik 'menu' untuk melihat "
        "topik yang tersedia."
    ),
    'TERIMA_KASIH': (
        "🤖 Sama-sama! 😊 Senang bisa membantu. "
        "Ada yang bisa AEGIS bantu lagi?"
    ),
}

# Pesan fallback global jika kategori tidak dikenal
_FALLBACK_GLOBAL = (
    "🤖 Maaf, AEGIS belum dapat memahami pertanyaanmu atau belum memiliki "
    "informasi terkait topik tersebut.\n\n"
    "💡 Coba salah satu cara berikut:\n"
    "• Gunakan kata kunci yang lebih spesifik, contoh:\n"
    "  - \"jadwal kuliah semester ini\"\n"
    "  - \"cara daftar keringanan UKT\"\n"
    "  - \"syarat pendaftaran sidang akhir\"\n"
    "• Ketik 'menu' untuk melihat daftar topik yang bisa AEGIS bantu\n\n"
    "📋 Topik yang tersedia:\n"
    "  Status Akademik | Persuratan | UKT & Keuangan | Wisuda & Kelulusan\n"
    "  Data Mahasiswa | Kemahasiswaan | Info Akademik | Info Dosen\n"
    "  Skripsi & TA | Magang & PKL\n\n"
    "📞 Butuh bantuan langsung?\n"
    "• Helpdesk Akademik :  <a href='https://helpakademik.polinema.ac.id/' target='_blank' color='blue'>klik disini</a>\n"
    "• Admin Akademik JTI: Gedung Sipil & Teknologi Informasi, Lantai 6\n"
    "• Jam layanan       : Senin–Jumat, 08.00–16.00 WIB"
)

# Skor cosine minimum — di bawah ini dianggap tidak relevan
# dan sistem menggunakan fallback per kategori
_COSINE_MIN_SCORE: float = 0.15


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
        - balasan           : str   → teks jawaban untuk ditampilkan ke user
        - kategori_ditebak  : str   → nama kategori yang diprediksi
        - tingkat_yakin     : float → confidence Naive Bayes (0.0 - 1.0)
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

    # Threshold dinamis — lebih rendah untuk sapaan pendek
    # karena input 1-2 kata secara matematis menghasilkan distribusi
    # probabilitas yang lebih merata (cost-sensitive classification)
    if kemungkinan_sapaan and jumlah_kata <= 2:
        threshold = 0.30
    else:
        threshold = 0.50

    id_kategori, confidence = predict(clean_input, threshold=threshold)

    # --- FALLBACK: confidence rendah — kategori tidak dikenali ---
    if id_kategori is None:
        if jumlah_kata == 1 and not kemungkinan_sapaan:
            balasan = (
                "🤖 Maaf, pertanyaanmu terlalu singkat sehingga AEGIS kurang paham. "
                "Bisa tolong jelaskan dengan kalimat yang lebih lengkap?"
            )
        else:
            balasan = _FALLBACK_GLOBAL
        return _build_response(balasan=balasan, kategori='TIDAK_DIKETAHUI', yakin=confidence)

    nama_kat = get_nama_kategori(id_kategori) or str(id_kategori)

    # --- CARI JAWABAN (TF-IDF Cosine via retriever) ---
    best_row, skor = retriever.find_best_answer(clean_input, id_kategori)

    # --- FALLBACK: tidak ada knowledge untuk kategori ini ---
    if best_row is None:
        balasan = _fallback_kategori(nama_kat)
        return _build_response(balasan=balasan, kategori=nama_kat, yakin=confidence)

    # --- FALLBACK: cosine terlalu rendah — jawaban tidak relevan ---
    # Gunakan fallback per kategori agar tetap informatif
    if skor < _COSINE_MIN_SCORE:
        balasan = _fallback_kategori(nama_kat)
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

def _fallback_kategori(nama_kat: str) -> str:
    """
    Kembalikan pesan fallback yang relevan berdasarkan kategori.
    Lebih informatif dari pesan generik karena menyertakan
    contoh pertanyaan dan arahan ke sumber informasi yang tepat.
    """
    return _FALLBACK_KATEGORI.get(nama_kat, _FALLBACK_GLOBAL)


def _build_response(balasan: str, kategori: str, yakin: float) -> dict:
    return {
        'balasan':          balasan,
        'kategori_ditebak': kategori,
        'tingkat_yakin':    round(yakin, 4),
    }