"""
retriever.py
------------
Tanggung jawab TUNGGAL: mencari jawaban terbaik dari knowledge base
menggunakan TF-IDF + Cosine Similarity.

Vectorizer di-fit SEKALI saat training dari SELURUH knowledge base,
lalu disimpan ke cache dan dimuat ulang saat server start.
Ini yang memperbaiki bug skor cosine = 0.

[PATCH BENCHMARK]
Ditambahkan pengukuran waktu internal di find_best_answer() dan
find_best_answer_global() untuk keperluan pengujian efisiensi.
Logika pencarian tidak berubah sama sekali.
Aktifkan dengan: retriever.BENCHMARK_MODE = True
"""

import os
import time
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
# 🔹 STATE INTERNAL
# ==========================================================
_vectorizer:       TfidfVectorizer | None = None
_knowledge_matrix: spmatrix | None        = None
_df_knowledge:     pd.DataFrame | None    = None

# ==========================================================
# 🔹 KONTROL BENCHMARK
# Ubah ke True dari luar untuk mengaktifkan pencatatan waktu.
# Contoh: import core.retriever as retriever; retriever.BENCHMARK_MODE = True
# ==========================================================
BENCHMARK_MODE: bool = False

# Log waktu yang terkumpul selama BENCHMARK_MODE aktif.
# Format setiap entri:
#   {
#     'mode'       : 'hybrid' | 'global',
#     'id_kategori': int | None,
#     'n_docs'     : int,   ← jumlah dokumen yang diproses cosine
#     'transform_ms': float, ← waktu vectorizer.transform()
#     'cosine_ms'  : float, ← waktu cosine_similarity()
#     'total_ms'   : float, ← transform + cosine (tanpa filter overhead)
#   }
benchmark_log: list[dict] = []


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

    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(_vectorizer,       _VECTORIZER_FILE)
    joblib.dump(_knowledge_matrix, _KNOWLEDGE_VEC_FILE)

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
    [Skenario Hybrid] Cari jawaban terbaik, dibatasi pada dokumen
    dalam kategori hasil prediksi Naïve Bayes.

    Ini adalah fungsi yang dipakai sistem normal.
    Saat BENCHMARK_MODE=True, waktu transform & cosine dicatat ke benchmark_log.

    Parameters
    ----------
    clean_input  : teks query sudah di-preprocess
    id_kategori  : id kategori hasil prediksi Naive Bayes

    Returns
    -------
    (row_dict, skor) — tidak berubah dari versi asli.
    """
    if _vectorizer is None or _knowledge_matrix is None or _df_knowledge is None:
        print("[retriever] ⚠️  Index belum siap.")
        return None, 0.0

    # Filter baris knowledge sesuai kategori
    mask   = _df_knowledge['id_kategori'] == id_kategori
    df_kat = _df_knowledge[mask]

    if df_kat.empty:
        return None, 0.0

    idx        = df_kat.index.tolist()
    kb_vectors = _knowledge_matrix[idx]

    # ── PENGUKURAN WAKTU (hanya aktif saat BENCHMARK_MODE=True) ──
    if BENCHMARK_MODE:
        t0           = time.perf_counter()
        user_vec     = _vectorizer.transform([clean_input])
        t_transform  = (time.perf_counter() - t0) * 1000

        t1           = time.perf_counter()
        skor_mirip   = cosine_similarity(user_vec, kb_vectors).flatten()
        t_cosine     = (time.perf_counter() - t1) * 1000

        benchmark_log.append({
            'mode'        : 'hybrid',
            'id_kategori' : id_kategori,
            'n_docs'      : len(idx),
            'transform_ms': round(t_transform, 6),
            'cosine_ms'   : round(t_cosine, 6),
            'total_ms'    : round(t_transform + t_cosine, 6),
        })
    else:
        # Jalur normal — tidak ada overhead pengukuran
        user_vec   = _vectorizer.transform([clean_input])
        skor_mirip = cosine_similarity(user_vec, kb_vectors).flatten()
    # ─────────────────────────────────────────────────────────────

    best_pos  = int(np.argmax(skor_mirip))
    best_skor = float(skor_mirip[best_pos])
    best_row  = df_kat.iloc[best_pos].to_dict()

    return best_row, best_skor


def find_best_answer_global(clean_input: str) -> tuple[dict | None, float]:
    """
    [Skenario Global Search] Cari jawaban terbaik dari SELURUH knowledge base
    tanpa filter kategori. Hanya dipakai untuk keperluan benchmark.

    Tidak dipanggil oleh sistem normal — chatbot.py tetap memakai
    find_best_answer() dengan filter intent.

    Parameters
    ----------
    clean_input : teks query sudah di-preprocess

    Returns
    -------
    (row_dict, skor) — baris jawaban terbaik dari seluruh KB.
    """
    if _vectorizer is None or _knowledge_matrix is None or _df_knowledge is None:
        print("[retriever] ⚠️  Index belum siap.")
        return None, 0.0

    # ── PENGUKURAN WAKTU ──
    if BENCHMARK_MODE:
        t0          = time.perf_counter()
        user_vec    = _vectorizer.transform([clean_input])
        t_transform = (time.perf_counter() - t0) * 1000

        t1          = time.perf_counter()
        skor_mirip  = cosine_similarity(user_vec, _knowledge_matrix).flatten()
        t_cosine    = (time.perf_counter() - t1) * 1000

        benchmark_log.append({
            'mode'        : 'global',
            'id_kategori' : None,
            'n_docs'      : _knowledge_matrix.shape[0],
            'transform_ms': round(t_transform, 6),
            'cosine_ms'   : round(t_cosine, 6),
            'total_ms'    : round(t_transform + t_cosine, 6),
        })
    else:
        user_vec   = _vectorizer.transform([clean_input])
        skor_mirip = cosine_similarity(user_vec, _knowledge_matrix).flatten()
    # ──────────────────────

    best_pos  = int(np.argmax(skor_mirip))
    best_skor = float(skor_mirip[best_pos])
    best_row  = _df_knowledge.iloc[best_pos].to_dict()

    return best_row, best_skor


def reset_benchmark_log():
    """Kosongkan log benchmark. Panggil sebelum sesi pengujian baru."""
    benchmark_log.clear()


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