import os
import pandas as pd

# Menentukan path file dataset secara dinamis agar aman dijalankan dari folder mana pun
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "data", "student_exam_performance.csv")

def baca_dataset(file_path):
    print("=" * 60)
    print("MEMBACA DATASET PEMBELAJARAN MESIN")
    print(f"Lokasi File: {file_path}")
    print("=" * 60)

    # 1. Cek keberadaan file
    if not os.path.exists(file_path):
        print(f"[ERROR] File dataset tidak ditemukan di: {file_path}")
        return None

    # 2. Membaca dataset menggunakan pandas
    try:
        df = pd.read_csv(file_path)
        print("[SUKSES] Dataset berhasil dimuat ke dalam DataFrame!\n")
    except Exception as e:
        print(f"[ERROR] Gagal membaca dataset: {e}")
        return None

    # 3. Menampilkan informasi dimensi/ukuran data
    print("--- 1. Dimensi Data ---")
    print(f"Jumlah Baris : {df.shape[0]:,}")
    print(f"Jumlah Kolom : {df.shape[1]}")
    print()

    # 4. Menampilkan 5 baris pertama data
    print("--- 2. Pratinjau 5 Baris Pertama (head) ---")
    print(df.head())
    print()

    # 5. Menampilkan informasi kolom, tipe data, dan memory usage
    print("--- 3. Informasi Kolom & Tipe Data (info) ---")
    df.info()
    print()

    # 6. Menampilkan statistik deskriptif untuk fitur numerik
    print("--- 4. Ringkasan Statistik Deskriptif (describe) ---")
    print(df.describe())
    print()

    # 7. Memeriksa missing values (nilai kosong / NaN)
    print("--- 5. Pemeriksaan Missing Values (Nilai Kosong) ---")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print("Kolom dengan nilai kosong:")
        for col, count in missing_cols.items():
            persentase = (count / len(df)) * 100
            print(f" - {col:<30}: {count:>6} baris ({persentase:.2f}%)")
    else:
        print("Tidak ditemukan missing value pada dataset.")
    print()

    # 8. Memeriksa duplikasi data
    print("--- 6. Pemeriksaan Data Duplikat ---")
    total_duplikat = df.duplicated().sum()
    print(f"Jumlah baris duplikat: {total_duplikat}")
    print("=" * 60)

    return df

if __name__ == "__main__":
    df = baca_dataset(dataset_path)
