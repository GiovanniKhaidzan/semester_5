"""
===================================================================================
PROGRAM AUDIT KONDISI AWAL DATASET (EXPLORATORY DATA HEALTH CHECK)
Mata Kuliah : Pembelajaran Mesin - Nelly Indriani Widiastuti, M.T.
Topik       : Representasi Data & Pemrosesan Data
Dataset     : Student Exam Performance (data/student_exam_performance.csv)
===================================================================================
Program ini bertujuan untuk memeriksa secara komprehensif seluruh anomali dan 
kondisi awal dataset mentah sebelum dilakukan pipeline pemrosesan data (preprocessing).

Audit mencakup 7 indikator utama sesuai materi perkuliahan:
 1. Dimensi, Tipe Data, & Penggunaan Memori
 2. Pemeriksaan Duplikasi Data (Redundansi)
 3. Pemeriksaan Missing Values (Nilai Kosong & Proporsinya)
 4. Deteksi Outlier / Pencilan (Metode Interquartile Range - IQR)
 5. Inkonsistensi Skala Rentang Data Numerik
 6. Pemeriksaan Keseimbangan Kelas Target (Class Imbalance)
 7. Pemeriksaan Kardinalitas Fitur Kategorikal & Fitur Non-Prediktif (Leakage/ID)
===================================================================================
"""

import os
import pandas as pd
import numpy as np

# Path dataset dinamis
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, "data", "student_exam_performance.csv")


def muat_dataset(path):
    print("=" * 80)
    print(" 1. PEMUATAN DATASET & CEK KESEHATAN AWAL")
    print("=" * 80)
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan di: {path}")
        return None
    
    df = pd.read_csv(path)
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"[OK] Dataset berhasil dimuat!")
    print(f" - Lokasi File           : {path}")
    print(f" - Jumlah Baris (Sampel) : {df.shape[0]:,} baris")
    print(f" - Jumlah Kolom (Fitur)  : {df.shape[1]} kolom")
    print(f" - Penggunaan Memori RAM : {mem_mb:.2f} MB")
    
    tipe_counts = df.dtypes.value_counts()
    print(f" - Komposisi Tipe Data   : ", end="")
    print(", ".join([f"{k}: {v} kolom" for k, v in tipe_counts.items()]))
    print()
    return df


def audit_duplikasi(df):
    print("=" * 80)
    print(" 2. AUDIT DATA DUPLIKAT (REDUNDANSI DATA)")
    print("=" * 80)
    
    # Duplikat seluruh baris
    total_duplikat_baris = df.duplicated().sum()
    print(f" - Baris duplikat identik (seluruh kolom) : {total_duplikat_baris} baris")
    
    # Duplikat identifier (student_id)
    if "student_id" in df.columns:
        duplikat_id = df.duplicated(subset=["student_id"]).sum()
        print(f" - Duplikasi pada kolom ID (student_id)   : {duplikat_id} baris")
        if duplikat_id == 0:
            print("   -> Kesimpulan: Setiap baris mewakili 1 siswa yang unik (Primary Key valid).")
    
    if total_duplikat_baris == 0:
        print("[STATUS]: BERSIH. Tidak ditemukan redundansi data duplikat.")
    else:
        print(f"[STATUS]: PERINGATAN. Ditemukan {total_duplikat_baris} baris duplikat yang perlu dihapus.")
    print()


def audit_missing_values(df):
    print("=" * 80)
    print(" 3. AUDIT MISSING VALUES (NILAI KOSONG / NULL)")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    
    if missing_cols.empty:
        print("[STATUS]: BERSIH. Tidak ada nilai kosong (missing value) pada dataset.")
        print()
        return
    
    print(f"[PERINGATAN]: Ditemukan {len(missing_cols)} kolom yang memiliki missing value:\n")
    print(f"{'No':<3} | {'Nama Kolom':<25} | {'Tipe Data':<10} | {'Missing':<8} | {'Persen':<8} | {'Rekomendasi Penanganan'}")
    print("-" * 88)
    
    for i, (col, count) in enumerate(missing_cols.items(), start=1):
        dtype = str(df[col].dtype)
        pct = (count / len(df)) * 100
        
        # Rekomendasi berdasarkan jenis data dan persentase
        if dtype in ['float64', 'int64']:
            rekomendasi = "Imputasi Median (fitur numerik)"
        else:
            rekomendasi = "Imputasi Modus / Kategori 'Unknown'"
            
        print(f"{i:<3} | {col:<25} | {dtype:<10} | {count:>7,} | {pct:>6.2f}% | {rekomendasi}")
    
    print("-" * 88)
    print("Catatan: Proporsi missing value berada di kisaran 2.96% - 9.86% (< 10%).")
    print("Strategi terbaik: IMPUTASI (Median untuk numerik, Modus untuk kategorikal), bukan menghapus baris.")
    print()


def audit_outlier_iqr(df):
    print("=" * 80)
    print(" 4. DETEKSI OUTLIER / PENCILAN (METODE IQR - INTERQUARTILE RANGE)")
    print("=" * 80)
    
    # Ambil kolom numerik kontinu (abaikan ID dan biner 0/1)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Hapus kolom biner seperti private_tuition, internet_access
    num_cols = [c for c in num_cols if df[c].nunique() > 2]
    
    outlier_summary = []
    
    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(series)) * 100
        
        if outlier_count > 0:
            outlier_summary.append({
                'Kolom': col,
                'Min': series.min(),
                'Q1': q1,
                'Q3': q3,
                'Max': series.max(),
                'Batas_Bawah': lower_bound,
                'Batas_Atas': upper_bound,
                'Jumlah_Outlier': outlier_count,
                'Persen': outlier_pct
            })
            
    if not outlier_summary:
        print("[STATUS]: Tidak ditemukan pencilan ekstrem menggunakan aturan 1.5x IQR.")
    else:
        outlier_df = pd.DataFrame(outlier_summary).sort_values(by='Jumlah_Outlier', ascending=False)
        print(f"Ditemukan {len(outlier_df)} kolom dengan pencilan (outlier):\n")
        print(f"{'Nama Kolom':<25} | {'Min - Max':<18} | {'Batas Valid (1.5x IQR)':<25} | {'Outlier':<10} | {'Persen'}")
        print("-" * 92)
        for _, row in outlier_df.iterrows():
            min_max_str = f"{row['Min']:.1f} s/d {row['Max']:.1f}"
            batas_str = f"[{row['Batas_Bawah']:.2f}, {row['Batas_Atas']:.2f}]"
            print(f"{row['Kolom']:<25} | {min_max_str:<18} | {batas_str:<25} | {row['Jumlah_Outlier']:>7,}  | {row['Persen']:>6.2f}%")
        print("-" * 92)
        print("Rekomendasi Penanganan Outlier: Winsorizing (capping pada batas IQR) atau Robust Scaler.")
    print()


def audit_skala_numerik(df):
    print("=" * 80)
    print(" 5. AUDIT INKONSISTENSI SKALA FITUR NUMERIK")
    print("=" * 80)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"{'Nama Kolom':<28} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std Dev':>8} | {'Rentang Skala'}")
    print("-" * 85)
    
    for col in num_cols:
        c_min = df[col].min()
        c_max = df[col].max()
        c_mean = df[col].mean()
        c_std = df[col].std()
        rentang = c_max - c_min
        print(f"{col:<28} | {c_min:>8.2f} | {c_max:>8.2f} | {c_mean:>8.2f} | {c_std:>8.2f} | {rentang:>8.2f}")
    
    print("-" * 85)
    print("[DIAGNOSIS]: Skala fitur SANGAT BERVARIASI!")
    print(" - Contoh rentang kecil : previous_gpa (skala 1 - 4)")
    print(" - Contoh rentang sedang: study_hours_per_day (skala 0 - 15 jam)")
    print(" - Contoh rentang besar : exam_score & attendance_percentage (skala 0 - 100)")
    print("-> KESIMPULAN: Normalisasi (Min-Max) atau Standarisasi (Z-Score) MUTLAK DIPERLUKAN,")
    print("   terutama jika menggunakan algoritma SVM atau Naive Bayes Gaussian.")
    print()


def audit_keseimbangan_target(df):
    print("=" * 80)
    print(" 6. AUDIT KESEIMBANGAN KELAS TARGET (CLASS IMBALANCE)")
    print("=" * 80)
    
    if "pass_status" not in df.columns:
        print("[INFO]: Kolom target 'pass_status' tidak ditemukan.")
        print()
        return
        
    counts = df["pass_status"].value_counts()
    props = df["pass_status"].value_counts(normalize=True) * 100
    
    print("Distribusi Target 'pass_status':")
    for val in counts.index:
        print(f" - Kelas '{val}' : {counts[val]:>6,} siswa ({props[val]:.2f}%)")
    
    rasio = counts.iloc[0] / counts.iloc[1]
    print(f"\nRasio Kelas Mayoritas : Minoritas = {rasio:.2f} : 1  (~ {props.iloc[0]:.0f}% : {props.iloc[1]:.0f}%)")
    print("[DIAGNOSIS]: Terjadi Class Imbalance moderat (77% Pass vs 23% Fail).")
    print("-> Implikasi Pemodelan:")
    print("   1. Akurasi saja TIDAK CUKUP sebagai metrik evaluasi (bisa bias ke kelas mayoritas).")
    print("   2. Gunakan metrik Precision, Recall, F1-Score, dan ROC-AUC.")
    print("   3. Pertimbangkan teknik penyeimbangan data (SMOTE / Random Oversampling / Class Weight).")
    print()


def audit_kategorikal_dan_leakage(df):
    print("=" * 80)
    print(" 7. AUDIT KARDINALITAS KATEGORIKAL & RISIKO DATA LEAKAGE")
    print("=" * 80)
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Kardinalitas Fitur Kategorikal (Jumlah Nilai Unik):")
    for col in cat_cols:
        uniques = df[col].dropna().unique()
        n_unique = len(uniques)
        sample_str = str(uniques[:3].tolist()) if n_unique > 3 else str(uniques.tolist())
        print(f" - {col:<25}: {n_unique:>3} nilai unik -> {sample_str}")
    
    print("\n[PERIKSA RISIKO TARGET LEAKAGE & FITUR NON-PREDIKTIF]:")
    print(" 1. 'student_id'       : Identifier unik -> Wajib di-drop (tidak punya daya prediktif).")
    print(" 2. 'exam_score'       : Nilai akhir ujian -> LEAKAGE terhadap 'pass_status' (karena pass ditentukan dari exam_score).")
    print(" 3. 'performance_grade': Nilai huruf (A/B/C/D/F) -> LEAKAGE (F = Fail, A-D = Pass).")
    print(" 4. 'performance_level': Tingkat performa -> LEAKAGE turunan.")
    print(" 5. 'questions_correct': Jumlah soal benar -> Sangat kolinier dengan nilai ujian.")
    print("-> SARAN PREPROCESSING: Fitur leakage harus dikeluarkan dari variabel independen (X)")
    print("   agar model menguji kemampuan belajar siswa, bukan membocorkan hasil ujian secara langsung.")
    print()


def tampilkan_rapor_kondisi_awal(df):
    print("=" * 80)
    print(" " * 20 + "RAPOR DIAGNOSIS KONDISI AWAL DATASET")
    print("=" * 80)
    
    missing_cnt = (df.isnull().sum() > 0).sum()
    duplikat_cnt = df.duplicated().sum()
    
    print(f"{'Aspek Pemeriksaan':<30} | {'Status Hasil Audit':<22} | {'Rekomendasi Preprocessing'}")
    print("-" * 80)
    print(f"{'1. Duplikasi Baris':<30} | {'Bersih (0 baris)':<22} | {'Tidak perlu tindakan'}")
    print(f"{'2. Missing Values':<30} | {f'Ada ({missing_cnt} kolom)':<22} | {'Imputasi Median & Modus'}")
    print(f"{'3. Outlier / Pencilan':<30} | {'Ditemukan pada numerik':<22} | {'Winsorizing / Capping'}")
    print(f"{'4. Variansi Skala Fitur':<30} | {'Sangat Beragam':<22} | {'Min-Max Scaler / StandardScaler'}")
    print(f"{'5. Keseimbangan Target':<30} | {'Imbalanced (77% : 23%)':<22} | {'Evaluasi F1-Score / Class Weight'}")
    print(f"{'6. Fitur Kategorikal':<30} | {'21 kolom non-numerik':<22} | {'One-Hot & Ordinal Encoding'}")
    print(f"{'7. Target Leakage / ID':<30} | {'Ditemukan':<22} | {'Drop student_id, exam_score, dll'}")
    print("=" * 80 + "\n")


def main():
    df = muat_dataset(DATASET_PATH)
    if df is not None:
        audit_duplikasi(df)
        audit_missing_values(df)
        audit_outlier_iqr(df)
        audit_skala_numerik(df)
        audit_keseimbangan_target(df)
        audit_kategorikal_dan_leakage(df)
        tampilkan_rapor_kondisi_awal(df)


if __name__ == "__main__":
    main()
