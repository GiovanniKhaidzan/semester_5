"""
app.py  --  Tugas Pemrosesan Data & Klasifikasi Gaussian Naive Bayes
Mata Kuliah : Pembelajaran Mesin - Nelly Indriani Widiastuti, M.T.
Dataset     : Student Exam Performance & Success Dataset (Kaggle)

Cara menjalankan:
    streamlit run app.py
"""

import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
)

# =================================================================
# KONSTANTA KONFIGURASI
# =================================================================

KOLOM_TARGET = "pass_status"

# Kolom yang dibuang: ID tanpa daya prediktif dan kolom turunan
# nilai ujian (data leakage karena menentukan pass/fail secara langsung).
KOLOM_DIBUANG = [
    "student_id", "exam_score", "performance_grade", "performance_level",
]

# Pemetaan encoding ordinal -- urutan dari terendah (0) ke tertinggi.
ORDINAL_MAP = {
    "education_level":     ["High School", "Undergraduate"],
    "family_income":       ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    "parent_education":    ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "class_participation": ["Low", "Medium", "High"],
    "study_consistency":   ["Low", "Medium", "High"],
    "motivation_level":    ["Low", "Medium", "High"],
    "educational_app_usage": ["Low", "Moderate", "High"],
    "study_environment":   ["Quiet", "Moderate", "Noisy"],
    "revision_frequency":  ["Rarely", "Weekly", "Daily"],
    "notes_quality":       ["Poor", "Average", "Excellent"],
    "sleep_quality":       ["Poor", "Fair", "Good", "Excellent"],
    "break_frequency":     ["Rarely", "Occasionally", "Frequently"],
    "exam_difficulty":     ["Easy", "Medium", "Hard"],
}

# Kolom yang di-one-hot encode (nominal tanpa urutan).
KOLOM_ONEHOT = ["gender", "school_type", "urban_rural", "study_method"]

# Kolom device_availability diubah jadi biner: Shared = 1, selainnya = 0.
KOLOM_DEVICE = "device_availability"

# Skala pengukuran tiap kolom (untuk Tab 3).
SKALA_KOLOM = {
    # --- Nominal (9) ---
    "student_id":          ("Nominal", "Hanya label pengenal unik tiap siswa, tanpa urutan."),
    "gender":              ("Nominal", "Kategori jenis kelamin (Female, Male, Other) tanpa tingkatan."),
    "school_type":         ("Nominal", "Jenis sekolah (Public, Private, Charter) tanpa urutan bermakna."),
    "urban_rural":         ("Nominal", "Lokasi tempat tinggal (Urban, Suburban, Rural) tanpa hierarki."),
    "study_method":        ("Nominal", "Metode belajar (Flashcards, Practice Tests, dll.) tanpa urutan."),
    "device_availability": ("Nominal", "Jenis kepemilikan perangkat (Shared, Dedicated) tanpa tingkatan."),
    "private_tuition":     ("Nominal", "Biner 0/1: mengikuti bimbingan privat atau tidak, tanpa urutan."),
    "internet_access":     ("Nominal", "Biner 0/1: ada akses internet di rumah atau tidak, tanpa urutan."),
    "pass_status":         ("Nominal", "Target klasifikasi biner (Pass/Fail), hanya label tanpa tingkatan."),
    # --- Ordinal (15) ---
    "education_level":     ("Ordinal", "Jenjang pendidikan berurutan: High School < Undergraduate."),
    "family_income":       ("Ordinal", "Tingkat pendapatan keluarga dari Low sampai High (5 tingkat)."),
    "parent_education":    ("Ordinal", "Pendidikan tertinggi orang tua: High School sampai Doctorate."),
    "class_participation": ("Ordinal", "Tingkat keaktifan di kelas: Low < Medium < High."),
    "study_consistency":   ("Ordinal", "Konsistensi belajar: Low < Medium < High."),
    "study_environment":   ("Ordinal", "Tingkat kebisingan lingkungan belajar: Quiet < Moderate < Noisy."),
    "revision_frequency":  ("Ordinal", "Frekuensi mengulang materi: Rarely < Weekly < Daily."),
    "notes_quality":       ("Ordinal", "Kualitas catatan: Poor < Average < Excellent."),
    "sleep_quality":       ("Ordinal", "Kualitas tidur: Poor < Fair < Good < Excellent."),
    "break_frequency":     ("Ordinal", "Frekuensi istirahat: Rarely < Occasionally < Frequently."),
    "motivation_level":    ("Ordinal", "Tingkat motivasi: Low < Medium < High."),
    "educational_app_usage": ("Ordinal", "Intensitas penggunaan aplikasi edukasi: Low < Moderate < High."),
    "exam_difficulty":     ("Ordinal", "Tingkat kesulitan ujian: Easy < Medium < Hard."),
    "performance_grade":   ("Ordinal", "Nilai huruf berurutan: A > B > C > D > F (dari exam_score)."),
    "performance_level":   ("Ordinal", "Tingkat performa: Low < Medium < High (dari exam_score)."),
    # --- Interval (4) ---
    "previous_gpa":        ("Interval", "IPK (1.0-4.0): selisih bermakna, tapi nol bukan berarti 'tidak ada'."),
    "stress_level":        ("Interval", "Skala stres (5-10 di data ini): selisih bermakna, titik nol arbitrer."),
    "exam_anxiety_level":  ("Interval", "Skala kecemasan (1.0-10.0): selisih bermakna, titik nol arbitrer."),
    "time_management_score": ("Interval", "Skor manajemen waktu: selisih bermakna, titik nol arbitrer."),
    # --- Rasio (16) ---
    "age":                       ("Rasio", "Usia dalam tahun: nol absolut ada dan rasio bermakna."),
    "previous_exam_score":       ("Rasio", "Nilai ujian sebelumnya (0-100): nol berarti tidak ada skor."),
    "attendance_percentage":     ("Rasio", "Persentase kehadiran (0-100%): nol berarti tidak pernah hadir."),
    "assignment_completion_rate":("Rasio", "Persentase tugas selesai: nol berarti tidak ada tugas selesai."),
    "study_hours_per_day":       ("Rasio", "Jam belajar per hari: nol berarti tidak belajar sama sekali."),
    "self_study_hours":          ("Rasio", "Jam belajar mandiri: nol berarti tidak belajar mandiri."),
    "online_learning_hours":     ("Rasio", "Jam belajar daring: nol berarti tidak belajar daring."),
    "practice_tests_completed":  ("Rasio", "Jumlah latihan soal: nol berarti tidak ada latihan."),
    "sleep_hours":               ("Rasio", "Durasi tidur harian dalam jam: nol berarti tidak tidur."),
    "daily_screen_time":         ("Rasio", "Waktu layar harian dalam jam: nol berarti tidak pakai layar."),
    "physical_activity_hours":   ("Rasio", "Jam olahraga harian: nol berarti tidak ada aktivitas fisik."),
    "online_course_hours":       ("Rasio", "Jam kursus online: nol berarti tidak ikut kursus."),
    "exam_preparation_days":     ("Rasio", "Hari persiapan ujian: nol berarti tidak ada persiapan."),
    "questions_attempted":       ("Rasio", "Jumlah soal dikerjakan: nol berarti tidak mengerjakan soal."),
    "questions_correct":         ("Rasio", "Jumlah soal benar: nol berarti tidak ada yang benar."),
    "exam_score":                ("Rasio", "Nilai akhir ujian (0-100): nol berarti tidak mendapat nilai."),
}


# =================================================================
# FUNGSI PEMROSESAN DATA
# =================================================================

def muat_data(path: str) -> pd.DataFrame:
    """Membaca file CSV dari path lokal."""
    return pd.read_csv(path)


def info_tipe_kolom(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan tipe data, jumlah terisi, kosong, dan contoh nilai."""
    baris = []
    for col in df.columns:
        terisi = int(df[col].notnull().sum())
        kosong = int(df[col].isnull().sum())
        valid = df[col].dropna()
        contoh = str(valid.iloc[0]) if len(valid) > 0 else "-"
        baris.append({
            "Kolom": col, "Tipe": str(df[col].dtype),
            "Terisi": f"{terisi:,}", "Kosong": kosong, "Contoh Nilai": contoh,
        })
    return pd.DataFrame(baris)


def hitung_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom yang punya nilai kosong, diurutkan terbanyak duluan."""
    m = df.isnull().sum()
    m = m[m > 0].sort_values(ascending=False)
    return pd.DataFrame({
        "Kolom": m.index, "Jumlah Kosong": m.values,
        "Persen (%)": (m.values / len(df) * 100).round(2),
    })


def persen_baris_ada_kosong(df: pd.DataFrame) -> float:
    """Persentase baris yang punya minimal satu nilai kosong."""
    return round(df.isnull().any(axis=1).sum() / len(df) * 100, 2)


def distribusi_target(df: pd.DataFrame) -> pd.DataFrame:
    """Jumlah dan persentase tiap kelas target."""
    c = df[KOLOM_TARGET].value_counts()
    p = df[KOLOM_TARGET].value_counts(normalize=True) * 100
    return pd.DataFrame({
        "Kelas": c.index, "Jumlah": c.values, "Persen (%)": p.values.round(2),
    })


def bukti_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Min/max exam_score per kelas (membuktikan batas skor 50)."""
    g = df.groupby(KOLOM_TARGET)["exam_score"].agg(["min", "max", "mean"])
    g.columns = ["Minimum", "Maksimum", "Rata-rata"]
    return g.round(2)


def tabel_skala() -> pd.DataFrame:
    """Tabel skala pengukuran seluruh kolom."""
    return pd.DataFrame([
        {"Kolom": col, "Skala": skala, "Alasan": alasan}
        for col, (skala, alasan) in SKALA_KOLOM.items()
    ])


def ringkasan_skala() -> pd.DataFrame:
    """Jumlah kolom per skala pengukuran."""
    counter = {}
    for _, (skala, _) in SKALA_KOLOM.items():
        counter[skala] = counter.get(skala, 0) + 1
    return pd.DataFrame([{"Skala": k, "Jumlah": v} for k, v in counter.items()])


# ---------- Langkah-langkah Pemrosesan ----------

def langkah_a(df: pd.DataFrame):
    """A: pisahkan target dan buang 4 kolom leakage/ID."""
    y = df[KOLOM_TARGET].copy()
    X = df.drop(columns=[KOLOM_TARGET] + KOLOM_DIBUANG, errors="ignore")
    return X, y


def langkah_b(X: pd.DataFrame) -> pd.DataFrame:
    """B: imputasi -- numerik -> median, kategorikal -> modus."""
    X = X.copy()
    for col in X.columns:
        if X[col].isnull().sum() == 0:
            continue
        if is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            modus = X[col].mode()
            if len(modus) > 0:
                X[col] = X[col].fillna(modus.iloc[0])
    return X


def langkah_c(X: pd.DataFrame) -> pd.DataFrame:
    """C: encoding ordinal (urutan dijaga, mulai dari 0)."""
    X = X.copy()
    for col, urutan in ORDINAL_MAP.items():
        if col in X.columns:
            mapping = {val: idx for idx, val in enumerate(urutan)}
            X[col] = X[col].map(mapping)
    return X


def langkah_d(X: pd.DataFrame) -> pd.DataFrame:
    """D: one-hot nominal + device_availability jadi biner."""
    X = X.copy()
    if KOLOM_DEVICE in X.columns:
        X[KOLOM_DEVICE] = (X[KOLOM_DEVICE] == "Shared").astype(int)
    X = pd.get_dummies(X, columns=KOLOM_ONEHOT, drop_first=True, dtype=int)
    return X


def langkah_e(y: pd.Series) -> pd.Series:
    """E: encoding target -- Pass = 1, Fail = 0."""
    return y.map({"Pass": 1, "Fail": 0})


def hitung_outlier_iqr(X: pd.DataFrame) -> pd.DataFrame:
    """F: deteksi outlier IQR (hanya lapor, tidak hapus baris)."""
    baris = []
    for col in X.columns:
        if not is_numeric_dtype(X[col]) or X[col].nunique() <= 2:
            continue
        s = X[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        if n > 0:
            baris.append({
                "Kolom": col,
                "Batas Bawah": round(lo, 2), "Batas Atas": round(hi, 2),
                "Jumlah Outlier": n,
                "Persen (%)": round(n / len(s) * 100, 2),
            })
    return pd.DataFrame(baris)


def proses_lengkap(df: pd.DataFrame):
    """Pipeline penuh: data mentah -> data siap model (X, y)."""
    X, y = langkah_a(df)
    X = langkah_b(X)
    X = langkah_c(X)
    X = langkah_d(X)
    y = langkah_e(y)
    return X, y


def latih_dan_evaluasi(df: pd.DataFrame) -> dict:
    """
    Bagi data 80/20 (stratify, random_state=42), latih GaussianNB.
    Imputasi di-fit hanya dari data latih agar tidak ada data leakage.
    """
    X_raw, y_raw = langkah_a(df)
    y_all = langkah_e(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_all, test_size=0.2, stratify=y_all, random_state=42,
    )

    med_vals, mod_vals = {}, {}
    for col in X_train.columns:
        if X_train[col].isnull().sum() == 0:
            continue
        if is_numeric_dtype(X_train[col]):
            med_vals[col] = X_train[col].median()
        else:
            m = X_train[col].mode()
            if len(m) > 0:
                mod_vals[col] = m.iloc[0]

    for col, v in med_vals.items():
        X_train[col] = X_train[col].fillna(v)
        X_test[col]  = X_test[col].fillna(v)
    for col, v in mod_vals.items():
        X_train[col] = X_train[col].fillna(v)
        X_test[col]  = X_test[col].fillna(v)

    X_train = langkah_d(langkah_c(X_train))
    X_test  = langkah_d(langkah_c(X_test))
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(
        y_test, y_pred,
        target_names=["Fail (0)", "Pass (1)"],
        output_dict=True,
    )
    return {
        "jumlah_latih": len(X_train),
        "jumlah_uji":   len(X_test),
        "akurasi":      akurasi,
        "confusion_matrix": cm,
        "classification_report": cr,
    }


# =================================================================
# FUNGSI HELPER TAMPILAN
# =================================================================

def st_tampilkan_dataframe(data, **kwargs):
    """Menampilkan dataframe responsif tanpa pesan deprecation."""
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)


# =================================================================
# KONFIGURASI HALAMAN & TAMPILAN STREAMLIT
# =================================================================

st.set_page_config(
    page_title="Tugas Pemrosesan Data & Gaussian Naive Bayes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk merapikan antarmuka dan tab
st.markdown("""
<style>
    /* Styling font dan kontainer utama */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.8rem;
    }
    
    /* Header aplikasi */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .app-header h1 {
        color: white !important;
        margin: 0 0 0.4rem 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #e0f2fe;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Badge ringkasan cepat */
    .badge-bar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .badge-item {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(4px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* Styling Tabs utama */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 8px 18px;
        border: none;
        background-color: transparent;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f9ff !important;
        border-bottom: 3px solid #0284c7 !important;
        color: #0369a1 !important;
    }

    /* Kotak penjelasan bersih */
    .penjelasan-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        color: #334155;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    /* Kartu metrik evaluasi */
    .eval-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .eval-card h4 {
        margin-top: 0;
        color: #0f172a;
        font-size: 1.05rem;
    }

    /* Metrik rapi */
    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================
# SIDEBAR: SUMBER DATA & PENGATURAN
# =================================================================

st.sidebar.title("Konfigurasi Data")
st.sidebar.caption("Tugas Pemrograman Pembelajaran Mesin")

# Path default relatif terhadap lokasi app.py
default_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    "student_exam_performance.csv",
)
path_input = st.sidebar.text_input("Path File CSV:", value=default_path)
uploaded = st.sidebar.file_uploader("Atau Unggah CSV Alternatif:", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Informasi Tugas**")
st.sidebar.write("- **Mata Kuliah**: Pembelajaran Mesin")
st.sidebar.write("- **Dosen Pengampu**: Nelly Indriani Widiastuti, M.T.")
st.sidebar.write("- **Metode Klasifikasi**: Gaussian Naive Bayes")
st.sidebar.write("- **Target Klasifikasi**: pass_status (Pass/Fail)")
st.sidebar.write("- **Ukuran Data**: 100.000 baris x 44 kolom")


# =================================================================
# CACHE DATA & PIPELINE
# =================================================================

@st.cache_data(show_spinner="Memuat dataset...")
def _muat(path):
    return muat_data(path)

@st.cache_data(show_spinner="Membaca file unggahan...")
def _muat_upload(data_bytes):
    return pd.read_csv(io.BytesIO(data_bytes))

@st.cache_data(show_spinner="Menjalankan pemrosesan data...")
def _proses(df):
    return proses_lengkap(df)

@st.cache_data(show_spinner="Melatih model Gaussian Naive Bayes...")
def _evaluasi(df):
    return latih_dan_evaluasi(df)


# Pemuatan data
df = None
if uploaded is not None:
    df = _muat_upload(uploaded.getvalue())
elif os.path.exists(path_input):
    df = _muat(path_input)

if df is None:
    st.error(
        "File dataset tidak ditemukan. Silakan periksa path file di sidebar "
        "atau unggah file CSV melalui formulir di sidebar."
    )
    st.stop()


# =================================================================
# BANNER HEADER APLIKASI
# =================================================================

st.markdown("""
<div class="app-header">
    <h1>Aplikasi Pemrosesan Data & Klasifikasi Gaussian Naive Bayes</h1>
    <p>Alur Komprehensif Data Mining: Dari Data Mentah, Eksplorasi, Skala Pengukuran, Pemrosesan, hingga Evaluasi Model</p>
    <div class="badge-bar">
        <span class="badge-item">Dataset: Student Exam Performance</span>
        <span class="badge-item">Baris: 100.000</span>
        <span class="badge-item">Kolom Mentah: 44</span>
        <span class="badge-item">Target: pass_status</span>
        <span class="badge-item">Algoritma: Gaussian Naive Bayes</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =================================================================
# STRUKTUR TAB TUNGGAL
# =================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Data Mentah",
    "2. Eksplorasi Data",
    "3. Skala Pengukuran",
    "4. Pemrosesan Data",
    "5. Hasil Pemrosesan",
    "6. Klasifikasi Gaussian NB",
])


# -----------------------------------------------------------------
# TAB 1: DATA MENTAH
# -----------------------------------------------------------------
with tab1:
    st.header("1. Data Mentah")
    st.caption("Pemeriksaan awal file dataset tabular sebelum dilakukan pengolahan.")

    c_pros, c_hasil = st.columns([1, 1], gap="medium")
    with c_pros:
        st.subheader("Proses")
        st.write(
            "Memuat file CSV sumber dataset dan memeriksa keutuhan data awal, "
            "jumlah dimensi baris dan kolom yang tersedia. Langkah ini memastikan data "
            "terbaca dengan benar sebelum dianalisis lebih lanjut."
        )
        st.markdown(
            "**Sumber Dataset**:\n"
            "[Student Exam Performance & Success Dataset - Kaggle]"
            "(https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-and-success-dataset)"
        )
    with c_hasil:
        st.subheader("Hasil")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Jumlah Total Baris", f"{df.shape[0]:,}")
        col_m2.metric("Jumlah Total Kolom", f"{df.shape[1]}")
        st.write(
            f"Dataset berhasil dimuat ke dalam sistem dengan total **{df.shape[0]:,} baris** "
            f"dan **{df.shape[1]} kolom atribut**."
        )

    st.markdown("---")
    st.subheader("Pratinjau Data Mentah")
    
    st.markdown("**5 Data Teratas (Data Mentah)**")
    st_tampilkan_dataframe(df.head())
    
    st.markdown("**5 Data Terbawah (Data Mentah)**")
    st_tampilkan_dataframe(df.tail())


# -----------------------------------------------------------------
# TAB 2: EKSPLORASI DATA
# -----------------------------------------------------------------
with tab2:
    st.header("2. Eksplorasi Data")
    st.caption("Analisis profil dataset: tipe data, nilai kosong, duplikasi, dan distribusi kelas target.")

    st.subheader("Proses")
    st.write(
        "Memeriksa kondisi aktual data sebelum dibersihkan: tipe tiap kolom, sebaran nilai kosong (missing values), "
        "keberadaan baris duplikat, dan keseimbangan kelas target (pass_status). Eksplorasi ini menjadi dasar untuk "
        "merumuskan teknik pembersihan dan transformasi data yang tepat."
    )

    st.markdown("---")
    st.subheader("Hasil")

    # Bagian Ringkasan Missing Values & Duplikat
    missing_df = hitung_missing(df)
    persen_kosong = persen_baris_ada_kosong(df)
    jml_duplikat = int(df.duplicated().sum())

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Kolom dengan Nilai Kosong", f"{len(missing_df)} dari {df.shape[1]}")
    col_e2.metric("Baris Terdampak Kosong", f"{persen_kosong}%")
    col_e3.metric("Jumlah Baris Duplikat", jml_duplikat)

    st.markdown("#### Distribusi Nilai Kosong per Kolom")
    if len(missing_df) > 0:
        c_ch1, c_ch2 = st.columns([1, 1], gap="medium")
        with c_ch1:
            st.bar_chart(missing_df.set_index("Kolom")["Jumlah Kosong"])
        with c_ch2:
            st_tampilkan_dataframe(missing_df)
            st.write(
                f"Terdapat 7 kolom yang memiliki nilai kosong. Sebanyak **{persen_kosong}% baris** "
                "memiliki minimal 1 nilai kosong. Karena hampir separuh data memiliki nilai kosong di salah satu kolomnya, "
                "maka baris data tidak boleh dihapus, melainkan harus diisi melalui teknik imputasi."
            )

    st.markdown("---")
    st.markdown("#### Distribusi Target Klasifikasi (pass_status)")
    dist_df = distribusi_target(df)
    c_tgt1, c_tgt2 = st.columns([1, 1], gap="medium")
    with c_tgt1:
        st_tampilkan_dataframe(dist_df)
    with c_tgt2:
        st.bar_chart(dist_df.set_index("Kelas")["Jumlah"])
    st.write(
        "Distribusi kelas target menunjukkan ketidakseimbangan (imbalance): "
        "kelas **Pass berjumlah 77.330 (77,33%)** dan kelas **Fail berjumlah 22.670 (22,67%)**. "
        "Kondisi ini mewajibkan pembagian data latih dan data uji menggunakan metode stratified split "
        "agar proporsi kelas tetap terjaga seimbang pada kedua set."
    )

    st.markdown("---")
    st.markdown("#### Pembuktian Batas Kelulusan (Nilai 50)")
    c_th1, c_th2 = st.columns([1, 1], gap="medium")
    with c_th1:
        st_tampilkan_dataframe(bukti_threshold(df))
    with c_th2:
        st.write(
            "Tabel di samping membuktikan secara empiris bahwa penentuan status kelulusan (pass_status) "
            "sepenuhnya didasarkan pada ambang batas skor ujian 50:\n\n"
            "- Siswa dengan status **Pass** memiliki nilai ujian mulai dari 50,00 hingga 100,00.\n"
            "- Siswa dengan status **Fail** memiliki nilai ujian di bawah 50 (antara 18,00 hingga 49,00).\n\n"
            "Oleh karena itu, fitur `exam_score` beserta turunannya (`performance_grade` dan `performance_level`) "
            "merupakan **data leakage** (kebocoran data) dan wajib dibuang dari fitur prediktor."
        )

    st.markdown("---")
    with st.expander("Lihat Rincian Tipe Data Seluruh Kolom & Contoh Nilai"):
        st_tampilkan_dataframe(info_tipe_kolom(df), height=400)


# -----------------------------------------------------------------
# TAB 3: SKALA PENGUKURAN
# -----------------------------------------------------------------
with tab3:
    st.header("3. Jenis Data (Skala Pengukuran)")
    st.caption("Identifikasi 4 skala pengukuran (Nominal, Ordinal, Interval, Rasio) pada seluruh 44 atribut.")

    st.subheader("Proses")
    st.write(
        "Mengelompokkan setiap atribut ke dalam salah satu dari empat skala pengukuran data statistik: "
        "Nominal, Ordinal, Interval, atau Rasio. Penentuan skala pengukuran ini krusial karena menentukan "
        "operasi matematika dan teknik encoding yang sah secara statistik (misalnya nominal memerlukan One-Hot Encoding, "
        "sedangkan ordinal memerlukan integer mapping berurutan)."
    )

    st.markdown("---")
    st.subheader("Hasil")

    ringkas_skala_df = ringkasan_skala()
    cols_skala = st.columns(4)
    for idx, row in ringkas_skala_df.iterrows():
        cols_skala[idx % 4].metric(f"Skala {row['Skala']}", f"{row['Jumlah']} Kolom")

    st.markdown("#### Tabel Klasifikasi Skala Pengukuran Seluruh Atribut")
    pilihan_filter = st.multiselect(
        "Saring berdasarkan jenis skala:",
        ["Nominal", "Ordinal", "Interval", "Rasio"],
        default=["Nominal", "Ordinal", "Interval", "Rasio"],
    )
    
    tabel_lengkap = tabel_skala()
    data_terfilter = tabel_lengkap[tabel_lengkap["Skala"].isin(pilihan_filter)]
    st_tampilkan_dataframe(data_terfilter, height=450)


# -----------------------------------------------------------------
# TAB 4: PEMROSESAN DATA
# -----------------------------------------------------------------
with tab4:
    st.header("4. Pemrosesan Data")
    st.caption("Transformasi data mentah menjadi data siap klasifikasi melalui 7 langkah berurutan.")

    # Sub-tab untuk memudahkan navigasi langkah-langkah pemrosesan
    sub_a, sub_b, sub_c, sub_d, sub_e, sub_f, sub_g = st.tabs([
        "Langkah A: Pisahkan Target",
        "Langkah B: Imputasi",
        "Langkah C: Ordinal Encoding",
        "Langkah D: One-Hot Encoding",
        "Langkah E: Target Encoding",
        "Langkah F: Duplikat & Outlier",
        "Langkah G: Analisis Scaling",
    ])

    # Eksekusi pipeline data lokal untuk pratinjau tiap langkah
    X_a, y_a = langkah_a(df)
    X_b = langkah_b(X_a)
    X_c = langkah_c(X_b)
    X_d = langkah_d(X_c)
    y_e = langkah_e(y_a)

    # --- Langkah A ---
    with sub_a:
        st.subheader("Langkah A: Pemisahan Target dan Pembuangan Kolom Leakage/ID")
        st.write(
            "Memisahkan kolom **pass_status** sebagai target (y), serta membuang 4 kolom dari fitur (X): "
            "`student_id`, `exam_score`, `performance_grade`, dan `performance_level`. "
            "Pemisahan dilakukan agar model tidak melihat kunci jawaban, kolom ID dibuang karena tidak "
            "memiliki nilai prediksi, dan 3 kolom turunan nilai ujian dibuang untuk mencegah kebocoran data (data leakage)."
        )
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Kolom Awal", df.shape[1])
        ca2.metric("Kolom Fitur (X)", X_a.shape[1])
        ca3.metric("Kolom Dibuang + Target", df.shape[1] - X_a.shape[1])
        st.write("Daftar kolom yang dibuang:", ", ".join(KOLOM_DIBUANG))
        with st.expander("Pratinjau 3 Baris Pertama Fitur (X) Setelah Langkah A"):
            st_tampilkan_dataframe(X_a.head(3))

    # --- Langkah B ---
    with sub_b:
        st.subheader("Langkah B: Imputasi Nilai Kosong (Missing Values)")
        persen_k = persen_baris_ada_kosong(X_a)
        st.write(
            f"Mengisi nilai kosong pada fitur numerik menggunakan nilai **median**, "
            f"dan pada fitur kategorikal menggunakan nilai **modus** (kategori tersering). "
            f"Penghapusan baris dihindari karena terdapat {persen_k}% baris yang memiliki nilai kosong. "
            f"Median dipilih untuk data numerik karena tahan terhadap pengaruh pencilan (outlier), "
            f"sedangkan modus mewakili kategori yang paling dominan."
        )
        cb1, cb2 = st.columns(2)
        cb1.metric("Total Kosong Sebelum Imputasi", f"{int(X_a.isnull().sum().sum()):,}")
        cb2.metric("Total Kosong Sesudah Imputasi", int(X_b.isnull().sum().sum()))

        missing_cols = hitung_missing(X_a)
        if len(missing_cols) > 0:
            st.markdown("#### Nilai Pengisi Imputasi per Kolom")
            detail_imp = []
            for _, row in missing_cols.iterrows():
                cn = row["Kolom"]
                if is_numeric_dtype(X_a[cn]):
                    val_str = f"Median = {X_a[cn].median():.2f}"
                else:
                    val_str = f"Modus = '{X_a[cn].mode().iloc[0]}'"
                detail_imp.append({
                    "Kolom": cn,
                    "Jumlah Kosong": int(row["Jumlah Kosong"]),
                    "Diisi Menggunakan": val_str,
                })
            st_tampilkan_dataframe(pd.DataFrame(detail_imp))

    # --- Langkah C ---
    with sub_c:
        st.subheader("Langkah C: Ordinal Encoding")
        st.write(
            f"Mengubah {len(ORDINAL_MAP)} kolom kategorikal yang memiliki urutan tingkatan bertingkat "
            "menjadi angka bulat terurut mulai dari 0. Teknik ini menjaga informasi hierarki alami data "
            "sehingga algoritma memahami bahwa nilai yang lebih tinggi merepresentasikan tingkatan yang lebih tinggi."
        )
        st.markdown("#### Kamus Pemetaan Ordinal Encoding")
        kamus_list = []
        for col, urutan in ORDINAL_MAP.items():
            pemetaan_str = " < ".join([f"{v} ({i})" for i, v in enumerate(urutan)])
            kamus_list.append({"Kolom": col, "Urutan Tingkatan": pemetaan_str})
        st_tampilkan_dataframe(pd.DataFrame(kamus_list), height=350)
        with st.expander("Pratinjau 3 Baris Pertama Fitur (X) Setelah Ordinal Encoding"):
            st_tampilkan_dataframe(X_c.head(3))

    # --- Langkah D ---
    with sub_d:
        st.subheader("Langkah D: One-Hot Encoding & Konversi Biner")
        st.write(
            "Menerapkan One-Hot Encoding dengan opsi `drop_first=True` pada 4 kolom nominal: "
            "`gender`, `school_type`, `urban_rural`, dan `study_method`. "
            "Kolom `device_availability` diubah menjadi biner (Shared = 1, Dedicated = 0). "
            "Kolom `private_tuition` dan `internet_access` dipertahankan karena sudah berbentuk biner 0/1. "
            "One-Hot Encoding digunakan untuk mencegah asumsi urutan palsu pada data tanpa tingkatan, "
            "dan drop_first=True diterapkan untuk mencegah multikolinearitas (dummy variable trap)."
        )
        cd1, cd2 = st.columns(2)
        cd1.metric("Jumlah Kolom Sebelum One-Hot", X_c.shape[1])
        cd2.metric("Jumlah Kolom Sesudah One-Hot", X_d.shape[1])
        with st.expander("Pratinjau 3 Baris Pertama Fitur (X) Setelah One-Hot Encoding"):
            st_tampilkan_dataframe(X_d.head(3))

    # --- Langkah E ---
    with sub_e:
        st.subheader("Langkah E: Encoding Target (pass_status)")
        st.write(
            "Mengonversi label teks target menjadi numerik biner: **Pass diubah menjadi 1** "
            "dan **Fail diubah menjadi 0**. Pengubahan ini dibutuhkan oleh model klasifikasi "
            "untuk perhitungan probabilitas dan evaluasi performa."
        )
        ce1, ce2 = st.columns(2)
        with ce1:
            st.markdown("**Sebelum (Teks)**")
            st_tampilkan_dataframe(y_a.value_counts().reset_index())
        with ce2:
            st.markdown("**Sesudah (Angka)**")
            st_tampilkan_dataframe(y_e.value_counts().reset_index())

    # --- Langkah F ---
    with sub_f:
        st.subheader("Langkah F: Pengecekan Baris Duplikat & Deteksi Outlier")
        st.write(
            "Memeriksa keberadaan baris duplikat dan mengidentifikasi pencilan (outlier) "
            "pada fitur numerik menggunakan metode Interquartile Range (IQR). "
            "Dataset terbukti tidak memiliki duplikat sama sekali. Nilai outlier yang terdeteksi "
            "tetap dipertahankan karena merefleksikan variasi alami data manusia (seperti jam belajar atau skor) "
            "dan agar seluruh 100.000 sampel data tetap utuh."
        )
        st.metric("Jumlah Baris Duplikat", int(df.duplicated().sum()))
        outlier_tab = hitung_outlier_iqr(X_d)
        if len(outlier_tab) > 0:
            st.markdown("#### Ringkasan Deteksi Outlier Metode IQR")
            st_tampilkan_dataframe(outlier_tab)

    # --- Langkah G ---
    with sub_g:
        st.subheader("Langkah G: Analisis Kebutuhan Scaling Fitur")
        st.write(
            "Standarisasi atau normalisasi skala fitur **secara sengaja tidak dilakukan** pada alur kerja ini. "
            "Alasan teknisnya adalah algoritma Gaussian Naive Bayes menghitung estimasi parameter rata-rata (mean) "
            "dan varians dari setiap fitur secara independen per kelas. Karena probabilitas dihitung terpisah untuk masing-masing "
            "dimensi fitur, perbedaan rentang nilai antar fitur tidak memengaruhi hasil perhitungan probabilitas posterior. "
            "Hal ini berbeda dengan model berbasis jarak (seperti KNN atau SVM) yang sensitif terhadap skala."
        )


# -----------------------------------------------------------------
# TAB 5: HASIL PEMROSESAN
# -----------------------------------------------------------------
with tab5:
    st.header("5. Hasil Pemrosesan")
    st.caption("Verifikasi data akhir yang telah bersih, terenkode penuh, dan siap dimasukkan ke model.")

    st.subheader("Proses")
    st.write(
        "Menggabungkan seluruh hasil transformasi data ke dalam satu struktur matriks akhir: "
        "memvalidasi tidak adanya nilai kosong yang tersisa, memastikan seluruh fitur telah bertipe numerik, "
        "dan meletakkan kolom target di bagian akhir tabel."
    )

    st.markdown("---")
    st.subheader("Hasil")

    X_final, y_final = _proses(df)
    sisa_kosong = int(X_final.isnull().sum().sum())
    semua_num = all(is_numeric_dtype(X_final[c]) for c in X_final.columns)

    ch_col1, ch_col2, ch_col3 = st.columns(3)
    ch_col1.metric("Total Baris Akhir", f"{X_final.shape[0]:,}")
    ch_col2.metric("Total Kolom Fitur (X)", X_final.shape[1])
    ch_col3.metric("Sisa Nilai Kosong (NaN)", sisa_kosong)

    # Data Gabungan Akhir
    data_hasil = X_final.copy()
    data_hasil["pass_status"] = y_final.values

    st.markdown("---")
    st.markdown("### Penjelasan Detail Hasil Pemrosesan Data")
    st.write(
        "Berikut adalah penjelasan mengenai kondisi data hasil pemrosesan sebelum dilatih menggunakan algoritma pembelajaran mesin:\n\n"
        f"1. **Dimensi Akhir Data**: Dataset mempertahankan seluruh **{X_final.shape[0]:,} baris**, dengan total **{X_final.shape[1]} kolom fitur independen** "
        f"dan 1 kolom target (`pass_status`).\n"
        "2. **Pembersihan Missing Value**: Tidak ada satupun nilai kosong tersisa (**0 NaN**). Seluruh nilai kosong berhasil diimputasi "
        "dengan median (untuk atribut numerik) dan modus (untuk atribut kategorikal).\n"
        "3. **Transformasi Menjadi Numerik Penuh**: Seluruh 45 kolom fitur kini berformat numerik (integer atau float), "
        "sehingga siap dikomputasi oleh model matematika Gaussian Naive Bayes tanpa kendala tipe data objek/string.\n"
        "4. **Pencegahan Data Leakage**: Empat kolom (`student_id`, `exam_score`, `performance_grade`, dan `performance_level`) "
        "telah berhasil dieliminasi, sehingga model tidak akan mengalami bias atau kecurangan dari fitur yang diturunkan dari nilai kelulusan.\n"
        "5. **Encoding Bebas Multikolinearitas**: Kolom nominal telah dipecah menjadi representasi biner dengan `drop_first=True`, "
        "sedangkan kolom ordinal mempertahankan tingkatan nilai terurut mulai dari 0."
    )

    st.markdown("---")
    st.subheader("Tampilan Data Hasil Pemrosesan")
    
    st.markdown("**5 Data Teratas (Data Hasil Pemrosesan)**")
    st_tampilkan_dataframe(data_hasil.head())

    st.markdown("**5 Data Terbawah (Data Hasil Pemrosesan)**")
    st_tampilkan_dataframe(data_hasil.tail())

    st.markdown("---")
    csv_unduh = data_hasil.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh File CSV Data Hasil Pemrosesan",
        data=csv_unduh,
        file_name="data_hasil_pemrosesan.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------
# TAB 6: KLASIFIKASI GAUSSIAN NAIVE BAYES
# -----------------------------------------------------------------
with tab6:
    st.header("6. Klasifikasi Gaussian Naive Bayes")
    st.caption("Pelatihan model, pengujian bebas data leakage, dan evaluasi matriks performa.")

    st.subheader("Proses")
    st.write(
        "Membagi dataset menjadi 80% data latih (80.000 sampel) dan 20% data uji (20.000 sampel) menggunakan "
        "metode stratified split (random_state=42) agar rasio kelas tetap terjaga. "
        "Untuk mencegah data leakage, estimasi imputasi dihitung murni dari data latih saja, lalu diaplikasikan ke data uji. "
        "Model Gaussian Naive Bayes dilatih pada data latih dan diuji pada data uji untuk mengukur kemampuan generalisasi model."
    )

    st.markdown("---")
    st.subheader("Hasil Evaluasi Model")

    hasil_eval = _evaluasi(df)
    cm = hasil_eval["confusion_matrix"]
    cr = hasil_eval["classification_report"]

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    total_uji = hasil_eval["jumlah_uji"]
    total_benar = tn + tp
    total_salah = fp + fn

    cm_col1, cm_col2, cm_col3 = st.columns(3)
    cm_col1.metric("Sampel Data Latih (80%)", f"{hasil_eval['jumlah_latih']:,}")
    cm_col2.metric("Sampel Data Uji (20%)", f"{hasil_eval['jumlah_uji']:,}")
    cm_col3.metric("Akurasi Model", f"{hasil_eval['akurasi'] * 100:.2f}%")

    st.markdown("---")
    c_eval1, c_eval2 = st.columns([1, 1], gap="large")

    with c_eval1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Fail (0)", "Pass (1)"], fontsize=10)
        ax.set_yticklabels(["Fail (0)", "Pass (1)"], fontsize=10)
        ax.set_xlabel("Prediksi Model", fontsize=11, fontweight="bold")
        ax.set_ylabel("Kelas Aktual", fontsize=11, fontweight="bold")
        ax.set_title("Confusion Matrix", fontsize=12, pad=12, fontweight="bold")
        for i in range(2):
            for j in range(2):
                warna = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13, color=warna, fontweight="bold"
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)

    with c_eval2:
        st.markdown("#### Classification Report")
        cr_df = pd.DataFrame(cr).T
        st_tampilkan_dataframe(cr_df.round(4))

    st.markdown("---")
    st.markdown("### Penjelasan Mendalam Hasil Klasifikasi")

    st.write(
        f"Berdasarkan evaluasi terhadap **{total_uji:,} sampel data uji**, model Gaussian Naive Bayes berhasil "
        f"memprediksi dengan akurasi sebesar **{hasil_eval['akurasi'] * 100:.2f}%** "
        f"({total_benar:,} sampel berhasil ditebak dengan benar, dan hanya {total_salah:,} sampel yang salah diprediksi). "
        "Berikut adalah rincian pembacaan hasil evaluasinya:"
    )

    col_p1, col_p2 = st.columns(2, gap="medium")

    with col_p1:
        st.markdown("#### 1. Rincian Confusion Matrix")
        st.write(
            f"- **True Negative (TN) = {tn:,}**:\n"
            "  Siswa yang sebenarnya **Fail** dan berhasil diprediksi dengan tepat sebagai **Fail** oleh model.\n\n"
            f"- **False Positive (FP) = {fp:,} (Galat Tipe I)**:\n"
            "  Siswa yang sebenarnya **Fail**, namun model keliru memprediksi mereka sebagai **Pass**.\n\n"
            f"- **False Negative (FN) = {fn:,} (Galat Tipe II)**:\n"
            "  Siswa yang sebenarnya **Pass**, namun model keliru memprediksi mereka sebagai **Fail**.\n\n"
            f"- **True Positive (TP) = {tp:,}**:\n"
            "  Siswa yang sebenarnya **Pass** dan berhasil diprediksi dengan tepat sebagai **Pass** oleh model."
        )

    with col_p2:
        st.markdown("#### 2. Rincian Metrik Performa")
        prec_fail = cr["Fail (0)"]["precision"] * 100
        rec_fail = cr["Fail (0)"]["recall"] * 100
        f1_fail = cr["Fail (0)"]["f1-score"] * 100

        prec_pass = cr["Pass (1)"]["precision"] * 100
        rec_pass = cr["Pass (1)"]["recall"] * 100
        f1_pass = cr["Pass (1)"]["f1-score"] * 100

        st.write(
            f"- **Kelas Fail (Minoritas ~23%)**:\n"
            f"  - **Presisi: {prec_fail:.2f}%**: Ketika model memprediksi seorang siswa Fail, probabilitas kebenarannya adalah {prec_fail:.2f}%.\n"
            f"  - **Recall: {rec_fail:.2f}%**: Model berhasil menjaring {rec_fail:.2f}% dari seluruh siswa yang berisiko tidak lulus.\n"
            f"  - **F1-Score: {f1_fail:.2f}%**: Keseimbangan harmonis antara presisi dan recall untuk kelas Fail.\n\n"
            f"- **Kelas Pass (Mayoritas ~77%)**:\n"
            f"  - **Presisi: {prec_pass:.2f}%**: Sangat tinggi, hampir seluruh prediksi Pass tepat sasaran.\n"
            f"  - **Recall: {rec_pass:.2f}%**: Model berhasil mendeteksi {rec_pass:.2f}% dari seluruh siswa yang lulus.\n"
            f"  - **F1-Score: {f1_pass:.2f}%**: Menunjukkan performa yang sangat kokoh pada kelas dominan."
        )

    st.markdown("---")
    st.markdown("#### 3. Kesimpulan & Analisis Asumsi Gaussian Naive Bayes")
    st.write(
        "Meskipun Gaussian Naive Bayes mengasumsikan bahwa seluruh fitur berdistribusi normal (Gaussian) "
        "dan saling bebas satu sama lain (conditional independence), asumsi tersebut pada kenyataannya jarang "
        "terpenuhi secara sempurna pada data tabular di dunia nyata. Variabel seperti tingkat stres atau jam belajar "
        "tentu memiliki korelasi tertentu dengan variabel lainnya.\n\n"
        f"Namun, performa model terbukti tetap sangat efektif dengan akurasi mencapai **{hasil_eval['akurasi'] * 100:.2f}%**. "
        "Hal ini terjadi karena dalam klasifikasi Naive Bayes, penentuan kelas pemenang didasarkan pada peringkat probabilitas "
        "terbesar (argmax probabilitas relatif), bukan nilai probabilitas mutlaknya. Dengan demikian, meskipun asumsi independensi "
        "dilanggar, ranking probabilitas antar kelas sering kali tetap tepat, menjadikan algoritma ini tangguh dan efisien "
        "untuk dataset berskala besar seperti 100.000 sampel ini."
    )
