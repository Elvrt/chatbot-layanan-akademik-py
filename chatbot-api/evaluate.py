"""
evaluate.py
-----------
Evaluasi model AEGIS secara independen.

Output (disimpan di folder 'evaluasi/'):
  - akurasi_kfold.png        : grafik akurasi tiap fold + rata-rata
  - confusion_matrix.png     : confusion matrix heatmap
  - classification_report.png: tabel precision/recall/f1 per kategori

Jalankan dari terminal:
  python evaluate.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
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

    scores = cross_val_score(pipeline, X, Y, cv=cv, scoring='accuracy')
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

    ax.axhline(scores.mean() * 100, color='#1565C0', linewidth=2,
               linestyle='--', label=f'Rata-rata: {scores.mean()*100:.2f}%', zorder=4)

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

    cm     = confusion_matrix(Y, y_pred, labels=label_ids)
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
# 🔹 PLOT CLASSIFICATION REPORT  ← FUNGSI BARU
# ==========================================================
def plot_classification_report(Y, y_pred, id_to_nama):
    """
    Menyimpan classification report sebagai tabel PNG.

    Setiap baris = satu kategori.
    Kolom: Precision, Recall, F1-Score, Support.
    Sel diwarnai berdasarkan nilai: hijau (≥0.90), kuning (≥0.80), merah (<0.80).
    Baris accuracy ditampilkan terpisah di bawah tabel.
    """
    label_ids   = sorted(set(Y))
    label_names = [id_to_nama.get(i, str(i)) for i in label_ids]

    # Hitung metrik per kelas
    precision, recall, f1, support = precision_recall_fscore_support(
        Y, y_pred, labels=label_ids, zero_division=0
    )

    # Hitung accuracy keseluruhan
    accuracy = sum(p == t for p, t in zip(y_pred, Y)) / len(Y)

    # Susun data tabel
    col_headers = ['Kategori', 'Precision', 'Recall', 'F1-Score', 'Support']
    rows = []
    for name, p, r, f, s in zip(label_names, precision, recall, f1, support):
        rows.append([name, f'{p:.2f}', f'{r:.2f}', f'{f:.2f}', str(int(s))])

    n_rows = len(rows)
    n_cols = len(col_headers)

    # Ukuran figure menyesuaikan jumlah baris
    fig_h = max(4, n_rows * 0.45 + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis('off')

    ax.set_title(
        'Classification Report — Naive Bayes AEGIS (10-Fold CV)',
        fontsize=13, fontweight='bold', pad=16
    )

    # Warna sel berdasarkan nilai metrik
    def warna(val_str, col_idx):
        """Kembalikan warna background untuk sel metrik (bukan kolom nama/support)."""
        if col_idx not in (1, 2, 3):
            return '#FFFFFF'
        try:
            v = float(val_str)
        except ValueError:
            return '#FFFFFF'
        if v >= 0.90:
            return '#C8E6C9'   # hijau muda
        if v >= 0.80:
            return '#FFF9C4'   # kuning muda
        return '#FFCDD2'       # merah muda

    # Warna header
    header_colors = [['#1565C0'] * n_cols]
    # Warna tiap sel data
    cell_colors = [
        [warna(cell, c) for c, cell in enumerate(row)]
        for row in rows
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=['#1565C0'] * n_cols,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.5)

    # Header putih tebal
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_text_props(color='white', fontweight='bold')

    # Kolom 'Kategori' rata kiri, lebar lebih besar
    tbl.auto_set_column_width(col=list(range(n_cols)))
    for r in range(n_rows):
        tbl[r + 1, 0].set_text_props(ha='left')

    # Baris alternating agar lebih mudah dibaca (hanya kolom nama & support)
    for r in range(n_rows):
        if r % 2 == 1:
            for c in (0, 4):
                tbl[r + 1, c].set_facecolor('#F5F5F5')

    # Ringkasan accuracy di bawah tabel
    fig.text(
        0.5, 0.01,
        f'Overall Accuracy: {accuracy*100:.2f}%   |   '
        f'Total Sampel: {len(Y)}   |   '
        f'Jumlah Kategori: {n_rows}',
        ha='center', va='bottom', fontsize=9.5,
        color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.8)
    )

    # Legenda warna
    legend_items = [
        (plt.Rectangle((0, 0), 1, 1, fc='#C8E6C9'), '≥ 0.90  Baik'),
        (plt.Rectangle((0, 0), 1, 1, fc='#FFF9C4'), '≥ 0.80  Cukup'),
        (plt.Rectangle((0, 0), 1, 1, fc='#FFCDD2'), '< 0.80  Perlu perbaikan'),
    ]
    ax.legend(
        [item[0] for item in legend_items],
        [item[1] for item in legend_items],
        loc='upper right',
        fontsize=8,
        framealpha=0.9,
        title='Keterangan warna',
        title_fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(_OUTPUT_DIR, 'classification_report.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[evaluate] ✅  Classification report disimpan → {out_path}")


# ==========================================================
# 🔹 MAIN
# ==========================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  EVALUASI MODEL AEGIS")
    print("="*50 + "\n")

    try:
        X, Y, id_to_nama          = load_dataset()
        scores, y_pred, n_splits  = jalankan_kfold(X, Y, id_to_nama)
        plot_akurasi(scores, n_splits)
        plot_confusion_matrix(Y, y_pred, id_to_nama)
        plot_classification_report(Y, y_pred, id_to_nama)   # ← baris baru

        print(f"\n[evaluate] ✅  Selesai! Hasil disimpan di folder → {_OUTPUT_DIR}/")

    except FileNotFoundError as e:
        print(f"\n[evaluate] ❌  {e}")
    except ValueError as e:
        print(f"\n[evaluate] ❌  {e}")
    except Exception as e:
        print(f"\n[evaluate] ❌  Error tidak terduga: {e}")
        raise