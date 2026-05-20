"""
preprocessor.py
---------------
Tanggung jawab TUNGGAL: membersihkan dan menormalisasi teks input.

Tidak boleh ada logika model, HTTP, atau database di sini.
"""

import os
import re
import pandas as pd
from difflib import get_close_matches
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================================
# 🔹 KONFIGURASI PATH
# ==========================================================
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_NORM_FILE      = os.path.join(_BASE_DIR, '..', 'file-dataset', 'normalization.csv')
_STOPWORDS_FILE = os.path.join(_BASE_DIR, '..', 'file-dataset', 'custom_stopwords.txt')

# ==========================================================
# 🔹 INISIALISASI SASTRAWI (dilakukan sekali saat import)
# ==========================================================
print("[preprocessor] Memuat Sastrawi...")
_stemmer  = StemmerFactory().create_stemmer()
_stopword = StopWordRemoverFactory().create_stop_word_remover()

# ==========================================================
# 🔹 STATE INTERNAL
# ==========================================================
_normalisasi_dict: dict = {}
_kosakata_norm:    set  = set()
_custom_stopwords: set  = set()

# Kata-kata khas sapaan yang wajib diselamatkan dari custom_stopwords
# dan dari Sastrawi stopword removal — berlaku untuk training maupun inferensi.
# Tanpa ini: 'hai', 'halo', 'salam', 'permisi' hilang → data SAPAAN jadi
# kosong → model bocor ke AKADEMIK_UMUM.
_WHITELIST_SAPAAN: set = {
    'halo', 'hai', 'hei', 'hey', 'hello', 'hi',
    'permisi', 'punten', 'assalamualaikum', 'waalaikumsalam',
    'salam', 'pagi', 'siang', 'sore', 'malam',
    'selamat', 'kabar', 'maaf', 'izin', 'nanya',
    'bertanya', 'tanya', 'bantuan', 'aktif', 'online',
}

# ==========================================================
# 🔹 FUNGSI LOAD INTERNAL
# ==========================================================

def _load_normalisasi():
    """Muat normalization.csv ke dictionary. Dipanggil saat modul diimport."""
    global _normalisasi_dict, _kosakata_norm

    norm_path = os.path.normpath(_NORM_FILE)
    if not os.path.exists(norm_path):
        print(f"[preprocessor] ⚠️  normalization.csv tidak ditemukan di {norm_path}. "
              "Fitur normalisasi dilewati.")
        return

    try:
        df                = pd.read_csv(norm_path)
        _normalisasi_dict = dict(zip(df['kata_typo'], df['kata_baku']))
        _kosakata_norm    = set(df['kata_baku'].tolist())
        print(f"[preprocessor] ✅  {len(_normalisasi_dict)} kata normalisasi dimuat.")
    except Exception as e:
        print(f"[preprocessor] ❌  Gagal memuat normalization.csv: {e}")


def _load_custom_stopwords():
    """Muat custom_stopwords.txt ke set. Dipanggil saat modul diimport."""
    global _custom_stopwords

    sw_path = os.path.normpath(_STOPWORDS_FILE)
    if not os.path.exists(sw_path):
        print(f"[preprocessor] ⚠️  custom_stopwords.txt tidak ditemukan di {sw_path}. "
              "Fitur ini dilewati.")
        return

    try:
        with open(sw_path, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        _custom_stopwords = set(words)
        print(f"[preprocessor] ✅  {len(_custom_stopwords)} custom stopwords dimuat.")
    except Exception as e:
        print(f"[preprocessor] ❌  Gagal memuat custom_stopwords.txt: {e}")


# Auto-load saat import
_load_normalisasi()
_load_custom_stopwords()

# ==========================================================
# 🔹 FUNGSI PUBLIK
# ==========================================================

def reload_normalisasi():
    """
    Refresh normalization.csv dan custom_stopwords.txt tanpa restart server.
    Dipanggil oleh chatbot.py setelah retrain.
    """
    _load_normalisasi()
    _load_custom_stopwords()


def update_kosakata(kosakata_tambahan: set):
    """
    Terima kosakata dari knowledge base (dipanggil oleh retriever.py setelah
    index dibangun) agar fuzzy fallback makin kaya referensi.
    """
    global _kosakata_norm
    _kosakata_norm.update(kosakata_tambahan)


def preprocess(text: str, pakai_fuzzy: bool = True, skip_custom_sw: bool = False) -> str:
    """
    Pipeline preprocessing lengkap.

    Parameters
    ----------
    text           : teks mentah dari user atau knowledge base
    pakai_fuzzy    : True  → query user (aktifkan fuzzy koreksi typo)
                     False → data training (tidak ubah data asli)
    skip_custom_sw : True  → lewati penghapusan custom stopwords.
                     Wajib True untuk NLU kategori SAPAAN agar token
                     khas sapaan tidak hilang saat training.

    Alur
    ----
    1. Lowercase + hapus tanda baca
    2. Normalisasi typo (normalization.csv)
    3. Fuzzy fallback           — hanya jika pakai_fuzzy=True
    4. Simpan whitelist sapaan  — SELALU aktif (training & inferensi)
    5. Hapus custom stopwords   — dilewati jika skip_custom_sw=True
    6. Stopword removal Sastrawi
    7. Stemming Sastrawi
    8. Kembalikan whitelist sapaan yang hilang di langkah 5/6/7
    9. Guard kosong             — kembalikan teks lowercase minimal
    """

    # 1. Lowercase + hapus tanda baca
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    # 2. Normalisasi typo dari CSV
    if _normalisasi_dict:
        words = [_normalisasi_dict.get(w, w) for w in text.split()]
        text  = " ".join(words)

    # 3. Fuzzy fallback (hanya untuk query user)
    if pakai_fuzzy:
        text = _koreksi_typo_fuzzy(text)

    # 4. Simpan whitelist sapaan SEBELUM apapun dihapus.
    #    Sengaja tidak dibatasi pakai_fuzzy agar data training SAPAAN
    #    juga terlindungi — 'hai', 'halo', 'salam' tidak hilang saat training.
    whitelist_saved = [w for w in text.split() if w in _WHITELIST_SAPAAN]

    # 5. Hapus custom stopwords
    #    Di-skip untuk NLU SAPAAN agar token sapaan tetap ada
    if _custom_stopwords and not skip_custom_sw:
        words = [w for w in text.split() if w not in _custom_stopwords]
        text  = " ".join(words)

    # 6. Stopword removal Sastrawi
    text = _stopword.remove(text)

    # 7. Stemming Sastrawi
    text = _stemmer.stem(text)

    # 8. Kembalikan whitelist yang dihapus di langkah 5, 6, atau 7.
    #    Ini yang menyelamatkan 'hai', 'halo', 'salam', 'permisi'
    #    yang dihapus Sastrawi meski tidak ada di custom_stopwords.
    if whitelist_saved:
        existing = set(text.split())
        tambahan = [w for w in whitelist_saved if w not in existing]
        if tambahan:
            text = (text + ' ' + ' '.join(tambahan)).strip()

    # 9. Guard kosong — kembalikan teks lowercase minimal agar model
    #    tetap punya token untuk diproses
    if not text.strip():
        fallback = str(text).lower()
        return re.sub(r'[^\w\s]', ' ', fallback).strip()

    return text


# ==========================================================
# 🔹 FUNGSI INTERNAL
# ==========================================================

def _koreksi_typo_fuzzy(text: str) -> str:
    """
    Jaring pengaman terakhir: kata yang tidak dikenal dicari padanannya
    dari kosakata baku menggunakan difflib.
    Hanya aktif jika kosakata referensi tersedia.
    """
    if not _kosakata_norm:
        return text

    hasil = []
    for kata in text.split():
        if kata not in _kosakata_norm and len(kata) > 3:
            saran = get_close_matches(kata, _kosakata_norm, n=1, cutoff=0.82)
            hasil.append(saran[0] if saran else kata)
        else:
            hasil.append(kata)
    return " ".join(hasil)