import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 1. INISIALISASI SASTRAWI ---
print("Memuat Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess_text(text):
    text = str(text).lower()
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# --- 2. MEMBACA FILE CSV ---
print("Membaca dataset NLU...")
df_nlu = pd.read_csv('dataset-nlu-clean.csv', sep=';')

# --- 3. PRE-PROCESSING DATA ---
print("Membersihkan teks dengan Sastrawi (Mohon tunggu)...")
df_nlu['pertanyaan_bersih'] = df_nlu['pertanyaan_variasi'].apply(preprocess_text)

X = df_nlu['pertanyaan_bersih'].tolist()
Y = df_nlu['id_kategori'].tolist()

# --- 4. PERSIAPAN MODEL ---
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.20, min_df=2),
    MultinomialNB(alpha=0.5)
)

# --- 5. PELATIHAN MODEL ---
print("Melatih model AEGIS 2.0...")
model.fit(X, Y)

# --- 6. MENYIMPAN MODEL KE DALAM FILE .PKL ---
print("Menyimpan model ke file .pkl...")
joblib.dump(model, 'aegis_model.pkl')

print("\nSUKSES! File 'aegis_model.pkl' berhasil dibuat dan siap digunakan untuk API Flask.")