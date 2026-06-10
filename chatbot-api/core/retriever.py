"""
retriever.py
------------
Tanggung jawab TUNGGAL: mencari jawaban terbaik dari knowledge base
menggunakan TF-IDF + Cosine Similarity.

Vectorizer di-fit SEKALI saat training dari SELURUH knowledge base,
lalu disimpan ke cache dan dimuat ulang saat server start.
Ini yang memperbaiki bug skor cosine = 0.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix

# ==========================================================
# 🔹 KONFIGURASI PATH
# ==========================================================
_BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR          = os.path.join(_BASE_DIR, '..', 'model-cache')
_VECTORIZER_FILE    = os.path.join(_MODEL_DIR, 'cosine_vectorizer.pkl')
_KNOWLEDGE_VEC_FILE = os.path.join(_MODEL_DIR, 'knowledge_vectors.pkl')

# ==========================================================
# 🔹 STATE INTERNAL (dimuat dari cache saat import)
# ==========================================================
_vectorizer:       TfidfVectorizer | None = None
_knowledge_matrix: spmatrix | None        = None
_df_knowledge:     pd.DataFrame | None    = None   # referensi dari classifier

# ==========================================================
# 🔹 FUNGSI PUBLIK
# ==========================================================

def build_index(df_knowledge: pd.DataFrame):
    """
    Di-fit dari SELURUH knowledge base saat training.
    Dipanggil oleh classifier.py di dalam latih_dari_file().

    Parameters
    ----------
    df_knowledge : DataFrame dengan kolom 'pertanyaan_baku_bersih'
                   (sudah di-preprocess tanpa fuzzy).
    """
    global _vectorizer, _knowledge_matrix, _df_knowledge

    print("[retriever] Membangun TF-IDF index dari seluruh knowledge base...")

    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=1
    )
    _knowledge_matrix = _vectorizer.fit_transform(
        df_knowledge['pertanyaan_baku_bersih']
    )
    _df_knowledge = df_knowledge.reset_index(drop=True)

    # Simpan ke cache
    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(_vectorizer,       _VECTORIZER_FILE)
    joblib.dump(_knowledge_matrix, _KNOWLEDGE_VEC_FILE)

    # Bagikan kosakata baku ke preprocessor untuk fuzzy fallback
    _share_kosakata()

    print(f"[retriever] ✅  Index berhasil dibangun "
          f"({_knowledge_matrix.shape[0]} dokumen, "
          f"{_knowledge_matrix.shape[1]} fitur).")


def load_index(df_knowledge: pd.DataFrame):
    """
    Muat vectorizer & matrix dari cache saat server start.
    Dipanggil oleh classifier.py di dalam load_cached_data().

    Parameters
    ----------
    df_knowledge : DataFrame knowledge base yang sudah dimuat dari cache.
    """
    global _vectorizer, _knowledge_matrix, _df_knowledge

    vec_exists = os.path.exists(_VECTORIZER_FILE)
    mat_exists = os.path.exists(_KNOWLEDGE_VEC_FILE)

    if not (vec_exists and mat_exists):
        print("[retriever] ⚠️  Cache cosine belum ada. Jalankan retrain dulu.")
        return

    try:
        _vectorizer       = joblib.load(_VECTORIZER_FILE)
        _knowledge_matrix = joblib.load(_KNOWLEDGE_VEC_FILE)
        _df_knowledge     = df_knowledge.reset_index(drop=True)

        _share_kosakata()

        print(f"[retriever] ✅  Index dimuat dari cache "
              f"({_knowledge_matrix.shape[0]} dokumen).")
    except Exception as e:
        print(f"[retriever] ❌  Gagal memuat cache cosine: {e}")


def find_best_answer(clean_input: str, id_kategori) -> tuple[dict | None, float]:
    """
    Cari jawaban terbaik untuk query yang sudah di-preprocess,
    terbatas pada id_kategori yang diprediksi Naive Bayes.

    Parameters
    ----------
    clean_input  : teks query sudah di-preprocess
    id_kategori  : id kategori hasil prediksi Naive Bayes

    Returns
    -------
    (row_dict, skor) : row jawaban terbaik sebagai dict, dan skor cosine-nya.
                       row_dict = None jika tidak ada data atau index belum siap.
    """
    if _vectorizer is None or _knowledge_matrix is None or _df_knowledge is None:
        print("[retriever] ⚠️  Index belum siap.")
        return None, 0.0

    # Filter baris knowledge sesuai kategori
    mask = _df_knowledge['id_kategori'] == id_kategori
    df_kat = _df_knowledge[mask]

    if df_kat.empty:
        return None, 0.0

    # Ambil baris matrix yang sesuai kategori (tanpa fit ulang!)
    idx = df_kat.index.tolist()
    kb_vectors = _knowledge_matrix[idx]

    # Transform query dengan vectorizer yang sama
    user_vec = _vectorizer.transform([clean_input])

    # Hitung cosine similarity
    skor_mirip = cosine_similarity(user_vec, kb_vectors).flatten()

    best_pos = int(np.argmax(skor_mirip))
    best_skor = float(skor_mirip[best_pos])
    best_row  = df_kat.iloc[best_pos].to_dict()

    return best_row, best_skor


def get_kosakata() -> set:
    """Kembalikan kosakata dari vectorizer untuk keperluan luar."""
    if _vectorizer is None:
        return set()
    return set(_vectorizer.vocabulary_.keys())


# ==========================================================
# 🔹 FUNGSI INTERNAL
# ==========================================================

def _share_kosakata():
    """Kirim kosakata ke preprocessor agar fuzzy fallback makin akurat."""
    try:
        from core.preprocessor import update_kosakata
        update_kosakata(get_kosakata())
        print("[retriever] ✅  Kosakata dibagikan ke preprocessor.")
    except Exception as e:
        print(f"[retriever] ⚠️  Gagal berbagi kosakata ke preprocessor: {e}")