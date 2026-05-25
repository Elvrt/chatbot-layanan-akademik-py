"""
evaluate.py — Evaluasi model Naive Bayes (aegis_model.pkl)
Menghasilkan:
  - Confusion matrix (gambar PNG)
  - Classification report (teks)
Menggunakan cross_val_predict agar hasil lebih jujur (tidak overfit ke training data).
"""

import os
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------------------------
# PATH — sesuaikan dengan struktur project kamu
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "file-dataset", "data_pelatihan.json")
NORM_FILE    = os.path.join(BASE_DIR, "file-dataset", "normalization.csv")
MODEL_FILE   = os.path.join(BASE_DIR, "model", "aegis_model.pkl")
OUTPUT_DIR   = os.path.join(BASE_DIR, "evaluasi")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# INISIALISASI SASTRAWI
# ---------------------------------------------------------------------------
print("Memuat Sastrawi...")
stemmer   = StemmerFactory().create_stemmer()
stopword  = StopWordRemoverFactory().create_stop_word_remover()

# ---------------------------------------------------------------------------
# NORMALISASI (opsional — dilewati kalau file tidak ada)
# ---------------------------------------------------------------------------
normalisasi_dict: dict = {}

if os.path.exists(NORM_FILE):
    try:
        df_norm = pd.read_csv(NORM_FILE)
        normalisasi_dict = dict(zip(df_norm["kata_typo"], df_norm["kata_baku"]))
        print(f"Normalisasi dimuat: {len(normalisasi_dict)} kata.")
    except Exception as e:
        print(f"Gagal memuat normalization.csv: {e}")
else:
    print("normalization.csv tidak ditemukan — fitur normalisasi dilewati.")


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    if normalisasi_dict:
        words = text.split()
        text  = " ".join(normalisasi_dict.get(w, w) for w in words)

    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text


# ---------------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[list, list, list]:
    """
    Mengembalikan:
      X      — list teks yang sudah di-preprocess
      Y      — list id_kategori (label)
      labels — list nama_kategori (untuk label di confusion matrix)
    """
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_FILE}")

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_nlu      = pd.DataFrame(data["dataset_nlu"])
    df_kategori = pd.DataFrame(data["kategori"])
    df_kategori.columns = ["id_kategori", "nama_kategori"]

    print("Preprocessing teks dataset...")
    df_nlu["teks_bersih"] = df_nlu["pertanyaan_variasi"].apply(preprocess_text)

    X = df_nlu["teks_bersih"].tolist()
    Y = df_nlu["id_kategori"].tolist()

    # Mapping id → nama untuk label confusion matrix
    id_to_nama = dict(zip(df_kategori["id_kategori"], df_kategori["nama_kategori"]))
    labels     = [id_to_nama.get(i, str(i)) for i in sorted(set(Y))]

    return X, Y, labels


# ---------------------------------------------------------------------------
# LOAD MODEL DARI FILE .pkl
# ---------------------------------------------------------------------------
def load_model():
    """Load model pipeline yang sudah di-train dari aegis_model.pkl."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_FILE}\n"
            "Jalankan train.py / NBmodel.py terlebih dahulu."
        )
    model = joblib.load(MODEL_FILE)
    print(f"Model berhasil dimuat dari: {MODEL_FILE}")
    return model


# ---------------------------------------------------------------------------
# CONFUSION MATRIX — menggunakan cross_val_predict (lebih jujur dari .predict)
# ---------------------------------------------------------------------------
def buat_confusion_matrix(model, X: list, Y: list, labels: list, n_splits: int = 10):
    """
    Hasilkan confusion matrix dengan cross-validation prediction.
    cross_val_predict lebih jujur karena setiap sampel diprediksi
    saat menjadi data uji, bukan data latih.
    """
    from collections import Counter

    # Pastikan n_splits tidak melebihi jumlah sampel per kelas
    min_count = min(Counter(Y).values())
    if min_count < n_splits:
        n_splits = min_count
        print(f"n_splits disesuaikan ke {n_splits} karena sampel minimum = {min_count}")

    if n_splits < 2:
        raise ValueError("Data terlalu sedikit untuk cross-validation (min 2 sampel per kelas).")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Menjalankan {n_splits}-Fold cross-validation untuk confusion matrix...")
    Y_pred = cross_val_predict(model, X, Y, cv=cv)

    # --- Hitung confusion matrix ---
    # Ambil urutan kelas sesuai model agar baris/kolom konsisten
    class_order  = list(model.classes_)
    id_to_nama   = dict(zip(sorted(set(Y)), labels))
    display_labels = [id_to_nama.get(c, str(c)) for c in class_order]

    cm = confusion_matrix(Y, Y_pred, labels=class_order)

    # --- Plot ---
    fig_size = max(8, len(class_order))  # ukuran otomatis berdasar jumlah kelas
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")

    ax.set_title(
        f"Confusion Matrix — Naive Bayes\n({n_splits}-Fold Cross-Validation)",
        fontsize=14,
        pad=16,
    )
    plt.tight_layout()

    # --- Simpan gambar ---
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix disimpan: {output_path}")

    return Y_pred, class_order


# ---------------------------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------------------------
def cetak_dan_simpan_report(Y_true, Y_pred, labels: list):
    report = classification_report(Y_true, Y_pred, target_names=labels, zero_division=0)

    print("\n=== Classification Report ===")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report — Naive Bayes\n")
        f.write("=" * 40 + "\n")
        f.write(report)

    print(f"Classification report disimpan: {report_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load data & model
    X, Y, labels = load_dataset()
    model        = load_model()

    # 2. Buat confusion matrix + prediksi via cross-validation
    Y_pred, class_order = buat_confusion_matrix(model, X, Y, labels, n_splits=10)

    # 3. Classification report
    #    Gunakan label sesuai urutan kelas model agar konsisten dengan matrix
    id_to_nama     = dict(zip(sorted(set(Y)), labels))
    ordered_labels = [id_to_nama.get(c, str(c)) for c in class_order]
    cetak_dan_simpan_report(Y, Y_pred, ordered_labels)

    print("\nEvaluasi selesai. Cek folder 'evaluasi/' untuk hasilnya.")