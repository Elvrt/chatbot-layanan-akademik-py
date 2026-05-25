import pandas as pd
import numpy as np
import os
import re
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from collections import Counter

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = 'file-dataset'
DATASET_FILE = os.path.join(DATASET_DIR, 'data_pelatihan.json')
NORM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_DIR, 'normalization.csv')

MODEL_DIR = 'model'
MODEL_FILE     = os.path.join(MODEL_DIR, 'aegis_model.pkl')
KATEGORI_FILE  = os.path.join(MODEL_DIR, 'kategori_cache.pkl')
KNOWLEDGE_FILE = os.path.join(MODEL_DIR, 'knowledge_cache.pkl')

# --- INISIALISASI SASTRAWI ---
print("Memuat Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

model = None
df_kategori = None
df_knowledge = None

# --- PREPROCESSING DATA ---
if os.path.exists(NORM_FILE):
    try:
        df_norm = pd.read_csv(NORM_FILE)
        # Ubah dataframe menjadi dictionary {kata_typo: kata_baku}
        normalisasi_dict = dict(zip(df_norm['kata_typo'], df_norm['kata_baku']))
        print(f"Berhasil memuat {len(normalisasi_dict)} kata normalisasi dari folder {DATASET_DIR}.")
    except Exception as e:
        print(f"Gagal memuat normalization.csv: {e}")
else:
    print(f"File normalization.csv tidak ditemukan di {NORM_FILE}. Fitur normalisasi dilewati.")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    if normalisasi_dict:
        words = text.split()
        normalized_words = [normalisasi_dict.get(w, w) for w in words]
        text = " ".join(normalized_words)

    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

def load_cached_data():
    global model, df_kategori, df_knowledge
    try:
        if (
            os.path.exists(MODEL_FILE) and 
            os.path.exists(KATEGORI_FILE) and
            os.path.exists(KNOWLEDGE_FILE)
        ):
            model = joblib.load(MODEL_FILE)
            df_kategori = joblib.load(KATEGORI_FILE)
            df_knowledge = joblib.load(KNOWLEDGE_FILE)

            print("Model & cache berhasil dimuat!")
        else:
            print("Model belum tersedia. Silakan retrain dulu.")
    except Exception as e:
        print("Error load cache:", e)

def buat_model() -> object:
    pipeline = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), 
            max_df=0.85, 
            min_df=1, 
            sublinear_tf=True
        ),
        MultinomialNB(alpha=0.5)
    )
    return pipeline

def evaluasi_model(X: list, Y: list, n_splits: int = 10) -> float:
    counts = Counter(Y)
    min_count = min(counts.values())

    if min_count < n_splits:
        n_splits = min_count
        print(f"Sampel terlalu sedikit, n_splits disesuaikan ke {n_splits}")

    if n_splits < 2:
        print("Data terlalu sedikit untuk cross validation. Evaluasi dilewati.")
        return 0.0
    
    pipeline = buat_model()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, Y, cv=cv, scoring='accuracy')

    print(f"\nHasil Cross Validation ({n_splits}-Fold):")
    print(f"Akurasi tiap fold : {[round(s, 2) for s in scores]}")
    print(f"Rata-rata akurasi : {scores.mean():.2f} ± {scores.std():.2f}\n")

    return scores.mean()

def latih_dari_file(jalankan_evaluasi: bool = True):
    global model, df_kategori, df_knowledge
    
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"File dataset tidak ditemukan di {DATASET_FILE}. Silakan kirim dari Laravel atau Postman terlebih dahulu.")
        
    print(f"Membaca data dari {DATASET_FILE}...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    df_nlu = pd.DataFrame(data_json['dataset_nlu'])
    df_kategori_baru = pd.DataFrame(data_json['kategori']) 
    df_knowledge_baru = pd.DataFrame(data_json['knowledge'])

    # Fix nama kolom kategori agar sesuai dengan yang diharapkan oleh endpoint chat
    df_kategori_baru.columns = ['id_kategori', 'nama_kategori']

    print("Preprocessing teks...")
    df_nlu['pertanyaan_bersih'] = df_nlu['pertanyaan_variasi'].apply(preprocess_text)
    df_knowledge_baru['pertanyaan_baku_bersih'] = df_knowledge_baru['pertanyaan'].apply(preprocess_text)

    X = df_nlu['pertanyaan_bersih'].tolist()
    Y = df_nlu['id_kategori'].tolist()

    if jalankan_evaluasi:
        evaluasi_model(X, Y)
    
    print("Melatih model...")
    new_model = buat_model()
    new_model.fit(X, Y)

    joblib.dump(new_model, MODEL_FILE)
    joblib.dump(df_kategori_baru, KATEGORI_FILE)
    joblib.dump(df_knowledge_baru, KNOWLEDGE_FILE)

    load_cached_data()
    print("Model berhasil dilatih dan disimpan!")

def predict_with_threshold(pertanyaan_bersih, threshold=0.3):
    if model is None:
        return None, 0.0
    
    probs = model.predict_proba([pertanyaan_bersih])[0]
    max_prob = probs.max()
    predicted_class = model.classes_[probs.argmax()]
    
    if max_prob < threshold:
        return None, max_prob
    
    return predicted_class, max_prob

load_cached_data()