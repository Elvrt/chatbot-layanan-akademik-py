"""
classifier.py
-------------
Tanggung jawab TUNGGAL: Naive Bayes — training, prediksi, load/save cache.

Mengintegrasikan preprocessor.py dan retriever.py:
- Saat training : panggil preprocessor, lalu build index di retriever
- Saat prediksi : hanya Naive Bayes, cosine diserahkan ke retriever
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from core.preprocessor import preprocess
from core import retriever

# ==========================================================
# 🔹 KONFIGURASI PATH
# ==========================================================
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR    = os.path.join(_BASE_DIR, '..', 'file-dataset')
_DATASET_FILE   = os.path.join(_DATASET_DIR, 'data_pelatihan.json')
_MODEL_DIR      = os.path.join(_BASE_DIR, '..', 'model-cache')
_MODEL_FILE     = os.path.join(_MODEL_DIR, 'aegis_model.pkl')
_KATEGORI_FILE  = os.path.join(_MODEL_DIR, 'kategori_cache.pkl')
_KNOWLEDGE_FILE = os.path.join(_MODEL_DIR, 'knowledge_cache.pkl')

# ID kategori SAPAAN di dataset — custom stopwords dilewati untuk kategori ini
# agar token khas sapaan ('halo', 'hai', 'permisi', dll.) tetap ada saat training
_ID_KATEGORI_SAPAAN = 8

# ==========================================================
# 🔹 STATE INTERNAL
# ==========================================================
_model:        object       | None = None
_df_kategori:  pd.DataFrame | None = None
_df_knowledge: pd.DataFrame | None = None

# ==========================================================
# 🔹 FUNGSI PUBLIK — AKSES STATE
# ==========================================================

def is_ready() -> bool:
    """Cek apakah model sudah dimuat dan siap dipakai."""
    return _model is not None and _df_kategori is not None and _df_knowledge is not None


def get_nama_kategori(id_kategori) -> str | None:
    """Ambil nama kategori dari id-nya."""
    if _df_kategori is None:
        return None
    row = _df_kategori[_df_kategori['id_kategori'] == id_kategori]
    if row.empty:
        return None
    return row['nama_kategori'].values[0]


# ==========================================================
# 🔹 FUNGSI PUBLIK — PREDIKSI
# ==========================================================

def predict(clean_input: str, threshold: float = 0.50) -> tuple[str | None, float]:
    """
    Prediksi kategori dari teks yang sudah di-preprocess.

    Parameters
    ----------
    clean_input : teks sudah di-preprocess via preprocessor.preprocess()
    threshold   : batas minimum confidence (default 0.50)

    Returns
    -------
    (id_kategori, confidence) : id_kategori = None jika di bawah threshold
    """
    if _model is None:
        return None, 0.0

    probabilitas = _model.predict_proba([clean_input])[0]
    confidence   = float(np.max(probabilitas))
    id_kategori  = _model.classes_[np.argmax(probabilitas)]

    if confidence < threshold:
        return None, confidence

    return id_kategori, confidence


# ==========================================================
# 🔹 FUNGSI PUBLIK — TRAINING
# ==========================================================

def train(data_json: dict, jalankan_evaluasi: bool = True):
    """
    Latih model dari dict JSON (bisa dari file atau payload HTTP).

    Parameters
    ----------
    data_json         : dict dengan key 'dataset_nlu', 'kategori', 'knowledge'
    jalankan_evaluasi : jalankan cross validation sebelum training final
    """
    global _model, _df_kategori, _df_knowledge

    print("[classifier] Memproses data training...")

    df_nlu  = pd.DataFrame(data_json['dataset_nlu'])
    df_kat  = pd.DataFrame(data_json['kategori'])
    df_know = pd.DataFrame(data_json['knowledge'])

    # Pastikan nama kolom konsisten
    df_kat.columns = ['id_kategori', 'nama_kategori']

    # ----------------------------------------------------------
    # PREPROCESSING NLU
    # Kategori SAPAAN mendapat skip_custom_sw=True agar token
    # khas sapaan ('halo', 'hai', 'permisi', dll.) tidak hilang.
    # Tanpa ini, 34 dari 147 data SAPAAN menjadi string kosong
    # dan 72 lainnya hanya punya 1 token ambigu → bocor ke
    # AKADEMIK_UMUM (30 kesalahan di confusion matrix).
    # ----------------------------------------------------------
    print("[classifier] Preprocessing NLU dataset...")

    def _preprocess_nlu(row) -> str:
        is_sapaan = int(row['id_kategori']) == _ID_KATEGORI_SAPAAN
        return preprocess(
            row['pertanyaan_variasi'],
            pakai_fuzzy=False,
            skip_custom_sw=is_sapaan,   # <-- perbedaan utama
        )

    df_nlu['pertanyaan_bersih'] = df_nlu.apply(_preprocess_nlu, axis=1)

    # ----------------------------------------------------------
    # PREPROCESSING KNOWLEDGE BASE
    # pakai_fuzzy=False agar data asli tidak berubah.
    # Knowledge tidak butuh skip_custom_sw karena pertanyaan
    # di knowledge sudah ditulis lengkap dan formal.
    # ----------------------------------------------------------
    print("[classifier] Preprocessing knowledge base...")
    df_know['pertanyaan_baku_bersih'] = df_know['pertanyaan'].apply(
        lambda t: preprocess(t, pakai_fuzzy=False)
    )

    X = df_nlu['pertanyaan_bersih'].tolist()
    Y = df_nlu['id_kategori'].tolist()

    # --- EVALUASI (opsional) ---
    if jalankan_evaluasi:
        _evaluasi(X, Y)

    # --- TRAINING NAIVE BAYES ---
    print("[classifier] Melatih Naive Bayes...")
    new_model = _buat_pipeline()
    new_model.fit(X, Y)

    # --- SIMPAN CACHE ---
    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(new_model, _MODEL_FILE)
    joblib.dump(df_kat,    _KATEGORI_FILE)
    joblib.dump(df_know,   _KNOWLEDGE_FILE)

    # --- BUILD INDEX COSINE ---
    retriever.build_index(df_know)

    # --- UPDATE STATE ---
    _model        = new_model
    _df_kategori  = df_kat
    _df_knowledge = df_know

    print("[classifier] ✅  Training selesai!")


def train_dari_file(jalankan_evaluasi: bool = True):
    """Baca dataset dari file JSON lalu panggil train()."""
    if not os.path.exists(_DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {_DATASET_FILE}. "
            "Kirim data via retrain endpoint terlebih dahulu."
        )

    print(f"[classifier] Membaca dataset dari {_DATASET_FILE}...")
    with open(_DATASET_FILE, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    train(data_json, jalankan_evaluasi)


# ==========================================================
# 🔹 FUNGSI PUBLIK — LOAD CACHE
# ==========================================================

def load_cache():
    """
    Muat model & data dari cache saat server start.
    Otomatis dipanggil saat modul diimport (di bawah).
    """
    global _model, _df_kategori, _df_knowledge

    files_exist = all(os.path.exists(f) for f in [
        _MODEL_FILE, _KATEGORI_FILE, _KNOWLEDGE_FILE
    ])

    if not files_exist:
        print("[classifier] ⚠️  Cache belum ada. Jalankan retrain dulu.")
        return

    try:
        _model        = joblib.load(_MODEL_FILE)
        _df_kategori  = joblib.load(_KATEGORI_FILE)
        _df_knowledge = joblib.load(_KNOWLEDGE_FILE)

        retriever.load_index(_df_knowledge)

        print("[classifier] ✅  Model & cache berhasil dimuat.")
    except Exception as e:
        print(f"[classifier] ❌  Gagal memuat cache: {e}")


# ==========================================================
# 🔹 FUNGSI INTERNAL
# ==========================================================

def _buat_pipeline() -> object:
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1,
            sublinear_tf=True,
        ),
        MultinomialNB(alpha=0.5)
    )


def _evaluasi(X: list, Y: list, n_splits: int = 10):
    counts    = Counter(Y)
    min_count = min(counts.values())

    if min_count < n_splits:
        n_splits = min_count
        print(f"[classifier] Sampel terlalu sedikit, n_splits → {n_splits}")

    if n_splits < 2:
        print("[classifier] Data terlalu sedikit untuk cross validation. Dilewati.")
        return

    pipeline = _buat_pipeline()
    cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores   = cross_val_score(pipeline, X, Y, cv=cv, scoring='accuracy')

    print(f"\n[classifier] Cross Validation ({n_splits}-Fold):")
    print(f"  Akurasi tiap fold : {[round(s, 2) for s in scores]}")
    print(f"  Rata-rata akurasi : {scores.mean():.2f} ± {scores.std():.2f}\n")


# ==========================================================
# 🔹 AUTO LOAD SAAT IMPORT
# ==========================================================
load_cache()