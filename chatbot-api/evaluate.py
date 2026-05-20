"""
evaluate.py
-----------
Evaluasi model AEGIS secara independen.

Output (disimpan di folder 'evaluasi/'):
  - akurasi_kfold.png     : grafik akurasi tiap fold + rata-rata
  - confusion_matrix.png  : confusion matrix heatmap

Jalankan dari terminal:
  python evaluate.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, aman untuk server/terminal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from collections import Counter

from core.preprocessor import preprocess

# ==========================================================
# 🔹 KONFIGURASI
# ==========================================================
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_DATASET_FILE = os.path.join(_BASE_DIR, 'file-dataset', 'data_pelatihan.json')
_OUTPUT_DIR   = os.path.join(_BASE_DIR, 'evaluasi')
N_SPLITS      = 10

os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 🔹 LOAD DATASET
# ==========================================================
def load_dataset():
    if not os.path.exists(_DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {_DATASET_FILE}. "
            "Jalankan retrain dulu."
        )

    print(f"[evaluate] Membaca dataset dari {_DATASET_FILE}...")
    with open(_DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    import pandas as pd
    df_nlu = pd.DataFrame(data['dataset_nlu'])
    df_kat = pd.DataFrame(data['kategori'])
    df_kat.columns = ['id_kategori', 'nama_kategori']

    print(f"[evaluate] Total sampel NLU : {len(df_nlu)}")
    print(f"[evaluate] Total kategori   : {len(df_kat)}")

    print("[evaluate] Preprocessing teks...")
    df_nlu['bersih'] = df_nlu['pertanyaan_variasi'].apply(
        lambda t: preprocess(t, pakai_fuzzy=False)
    )

    X = df_nlu['bersih'].tolist()
    Y = df_nlu['id_kategori'].tolist()

    # Map id → nama kategori untuk label confusion matrix
    id_to_nama = dict(zip(df_kat['id_kategori'], df_kat['nama_kategori']))

    return X, Y, id_to_nama


# ==========================================================
# 🔹 BUAT PIPELINE
# ==========================================================
def buat_pipeline():
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1,
            sublinear_tf=True
        ),
        MultinomialNB(alpha=0.5)
    )


# ==========================================================
# 🔹 K-FOLD CROSS VALIDATION
# ==========================================================
def jalankan_kfold(X, Y, id_to_nama):
    counts    = Counter(Y)
    min_count = min(counts.values())
    n_splits  = min(N_SPLITS, min_count)

    if n_splits < 2:
        raise ValueError(
            f"Data terlalu sedikit untuk cross validation "
            f"(min sampel per kelas: {min_count})."
        )

    if n_splits < N_SPLITS:
        print(f"[evaluate] ⚠️  n_splits disesuaikan: {N_SPLITS} → {n_splits} "
              f"(min sampel per kelas: {min_count})")

    print(f"[evaluate] Menjalankan {n_splits}-Fold Cross Validation...")

    pipeline = buat_pipeline()
    cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Akurasi tiap fold
    scores = cross_val_score(pipeline, X, Y, cv=cv, scoring='accuracy')

    # Prediksi untuk confusion matrix
    y_pred = cross_val_predict(pipeline, X, Y, cv=cv)

    print(f"\n{'='*50}")
    print(f"  Hasil {n_splits}-Fold Cross Validation")
    print(f"{'='*50}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i:2d} : {s:.4f} ({s*100:.2f}%)")
    print(f"{'─'*50}")
    print(f"  Rata-rata : {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"  Std Dev   : {scores.std():.4f}")
    print(f"{'='*50}\n")

    # Classification report
    label_ids   = sorted(set(Y))
    label_names = [id_to_nama.get(i, str(i)) for i in label_ids]
    print("[evaluate] Classification Report:")
    print(classification_report(Y, y_pred, labels=label_ids, target_names=label_names))

    return scores, y_pred, n_splits


# ==========================================================
# 🔹 PLOT AKURASI K-FOLD
# ==========================================================
def plot_akurasi(scores, n_splits):
    fig, ax = plt.subplots(figsize=(10, 5))

    fold_labels = [f"Fold {i}" for i in range(1, n_splits + 1)]
    bar_colors  = [
        '#4CAF50' if s >= scores.mean() else '#FF7043'
        for s in scores
    ]

    bars = ax.bar(fold_labels, scores * 100, color=bar_colors,
                  edgecolor='white', linewidth=0.8, zorder=3)

    # Garis rata-rata
    ax.axhline(scores.mean() * 100, color='#1565C0', linewidth=2,
               linestyle='--', label=f'Rata-rata: {scores.mean()*100:.2f}%', zorder=4)

    # Label nilai di atas tiap bar
    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{s*100:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    ax.set_title(f'Akurasi {n_splits}-Fold Cross Validation — Naive Bayes AEGIS',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('Akurasi (%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')

    # Kotak ringkasan
    summary = (
        f"Mean  : {scores.mean()*100:.2f}%\n"
        f"Std   : {scores.std()*100:.2f}%\n"
        f"Max   : {scores.max()*100:.2f}%\n"
        f"Min   : {scores.min()*100:.2f}%"
    )
    ax.text(0.98, 0.97, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(_OUTPUT_DIR, 'akurasi_kfold.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[evaluate] ✅  Grafik akurasi disimpan → {out_path}")


# ==========================================================
# 🔹 PLOT CONFUSION MATRIX
# ==========================================================
def plot_confusion_matrix(Y, y_pred, id_to_nama):
    label_ids   = sorted(set(Y))
    label_names = [id_to_nama.get(i, str(i)) for i in label_ids]

    cm = confusion_matrix(Y, y_pred, labels=label_ids)

    # Ukuran figure menyesuaikan jumlah kelas
    n      = len(label_ids)
    size   = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        linecolor='#E0E0E0',
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title('Confusion Matrix — Naive Bayes AEGIS (10-Fold CV)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Prediksi', fontsize=11, labelpad=10)
    ax.set_ylabel('Aktual', fontsize=11, labelpad=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

    plt.tight_layout()
    out_path = os.path.join(_OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[evaluate] ✅  Confusion matrix disimpan → {out_path}")


# ==========================================================
# 🔹 MAIN
# ==========================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  EVALUASI MODEL AEGIS")
    print("="*50 + "\n")

    try:
        X, Y, id_to_nama = load_dataset()
        scores, y_pred, n_splits = jalankan_kfold(X, Y, id_to_nama)
        plot_akurasi(scores, n_splits)
        plot_confusion_matrix(Y, y_pred, id_to_nama)

        print(f"\n[evaluate] ✅  Selesai! Hasil disimpan di folder → {_OUTPUT_DIR}/")

    except FileNotFoundError as e:
        print(f"\n[evaluate] ❌  {e}")
    except ValueError as e:
        print(f"\n[evaluate] ❌  {e}")
    except Exception as e:
        print(f"\n[evaluate] ❌  Error tidak terduga: {e}")
        raise