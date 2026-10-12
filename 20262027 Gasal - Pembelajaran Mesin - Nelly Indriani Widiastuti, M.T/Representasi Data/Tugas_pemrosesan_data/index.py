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

def jawab_pertanyaan_siapa_yang_diukur(df):
    if df is None:
        return
    
    print("\n" + "=" * 60)
    print("CERITA DATA: PERTANYAAN 'SIAPA YANG DIUKUR?'")
    print("=" * 60)
    
    # 1. Rentang usia
    min_usia = int(df['age'].min())
    max_usia = int(df['age'].max())
    print(f"1. Rentang Usia:")
    print(f"   - Usia Minimum : {min_usia} tahun")
    print(f"   - Usia Maksimum : {max_usia} tahun")
    daftar_usia = [int(u) for u in sorted(df['age'].unique())]
    print(f"   - Nilai Usia Unik: {daftar_usia}")
    print()

    # 2. Sebaran Jenjang Pendidikan
    print("2. Sebaran Jenjang Pendidikan (education_level):")
    jenjang_counts = df['education_level'].value_counts()
    for jenjang, count in jenjang_counts.items():
        persen = (count / len(df)) * 100
        print(f"   - {jenjang:<15}: {count:>8,} siswa ({persen:.2f}%)")
    print()

    # 3. Distribusi Usia per Jenjang Pendidikan
    print("3. Rincian Usia per Jenjang Pendidikan:")
    rincian = df.groupby('education_level')['age'].agg(
        Usia_Min='min',
        Usia_Max='max',
        Rata_rata='mean',
        Total_Siswa='count'
    )
    print(rincian)
    print()

    # 4. Kesimpulan untuk laporan
    print("-" * 60)
    jenjang_str = ", ".join(df['education_level'].unique())
    print(f"[KESIMPULAN JAWABAN TUGAS]")
    print(f"Pertanyaan : Siapa yang diukur?")
    print(f"Jawaban    : Siswa usia {min_usia}-{max_usia} tahun, jenjang {jenjang_str}")
    print("=" * 60 + "\n")

def cek_jenis_kelamin(df):
    if df is None or 'gender' not in df.columns:
        return

    print("=" * 60)
    print("ANALISIS FITUR: JENIS KELAMIN (GENDER)")
    print("=" * 60)

    # 1. Kategori unik
    kategori_gender = df['gender'].unique().tolist()
    print(f"1. Kategori Jenis Kelamin : {kategori_gender}")
    print(f"   Missing Values (Kosong): {df['gender'].isnull().sum()}")
    print()

    # 2. Jumlah dan Persentase
    print("2. Distribusi Jumlah & Persentase:")
    gender_counts = df['gender'].value_counts()
    for gender, count in gender_counts.items():
        persen = (count / len(df)) * 100
        print(f"   - {gender:<10}: {count:>8,} siswa ({persen:.2f}%)")
    print()

    # 3. Sebaran Gender per Jenjang Pendidikan (Cross-tabulation)
    print("3. Sebaran Gender per Jenjang Pendidikan:")
    ct = pd.crosstab(df['education_level'], df['gender'], margins=True, margins_name="Total")
    print(ct)
    print()

    # 4. Rata-rata Nilai Ujian per Gender (jika ada kolom exam_score)
    if 'exam_score' in df.columns:
        print("4. Rata-rata Nilai Ujian (exam_score) per Gender:")
        perf = df.groupby('gender')['exam_score'].agg(
            Rata_rata_Nilai='mean',
            Nilai_Min='min',
            Nilai_Max='max'
        )
        print(perf)
        print()

    # 5. Informasi Skala Pengukuran untuk Laporan Tugas
    print("-" * 60)
    print("[INFORMASI SKALA PENGUKURAN UNTUK LAPORAN]")
    print("Nama Fitur        : gender")
    print("Jenis Data        : Kualitatif / Kategorikal")
    print("Skala Pengukuran  : NOMINAL (karena hanya melabelkan kategori tanpa tingkatan/ranking)")
    print("=" * 60 + "\n")

def cek_target_pass_status(df):
    if df is None or 'pass_status' not in df.columns:
        return

    print("=" * 60)
    print("ANALISIS TARGET KLASIFIKASI: STATUS KELULUSAN (pass_status)")
    print("=" * 60)

    # 1. Menghitung proporsi dengan normalize=True
    proporsi = df["pass_status"].value_counts(normalize=True)
    jumlah = df["pass_status"].value_counts()

    print("1. Hasil df['pass_status'].value_counts(normalize=True):")
    print(proporsi)
    print()

    # 2. Rincian detail jumlah & persentase
    print("2. Rincian Jumlah dan Persentase:")
    for status, p in proporsi.items():
        cnt = jumlah[status]
        persen = p * 100
        print(f"   - {status:<6}: {cnt:>8,} siswa ({persen:.2f}% ~ dibulatkan {round(persen)}%)")
    print()

    # 3. Keterangan untuk Laporan Tugas
    print("-" * 60)
    print("[INFORMASI UNTUK LAPORAN TUGAS]")
    print(f"Target Variabel   : pass_status ({', '.join(proporsi.index)})")
    print(f"Proporsi Kelas    : Pass = {proporsi.get('Pass', 0)*100:.2f}% (77%), Fail = {proporsi.get('Fail', 0)*100:.2f}% (23%)")
    print("Kondisi Data      : Data Tidak Seimbang (Imbalanced Data ~ 77% : 23%)")
    print("Skala Pengukuran  : NOMINAL / BINER (karena merupakan klasifikasi dua kelas)")
    print("=" * 60 + "\n")

def cek_performance_grade(df):
    if df is None or 'performance_grade' not in df.columns:
        return

    print("=" * 60)
    print("ANALISIS NILAI HURUF: GRADE PRESTASI (performance_grade)")
    print("=" * 60)

    # 1. Proporsi normalisasi
    proporsi = df["performance_grade"].value_counts(normalize=True)
    jumlah = df["performance_grade"].value_counts()

    print("1. Hasil df['performance_grade'].value_counts(normalize=True):")
    print(proporsi)
    print()

    # 2. Rincian berurutan (A -> B -> C -> D -> F)
    urutan_grade = ['A', 'B', 'C', 'D', 'F']
    print("2. Rincian Distribusi Grade (Urutan A s.d. F):")
    for grade in urutan_grade:
        if grade in jumlah:
            cnt = jumlah[grade]
            p = proporsi[grade]
            persen = p * 100
            print(f"   - Grade {grade} : {cnt:>8,} siswa ({persen:>6.2f}% ~ {round(persen):>2}%) [proporsi: {p:.5f}]")
    print()

    # 3. Keterangan untuk Laporan Tugas
    print("-" * 60)
    print("[INFORMASI SKALA PENGUKURAN UNTUK LAPORAN]")
    print("Nama Fitur        : performance_grade")
    print("Jenis Data        : Kualitatif / Kategorikal")
    print("Skala Pengukuran  : ORDINAL (karena memiliki tingkatan berurutan: A > B > C > D > F)")
    print("=" * 60 + "\n")

def tampilkan_info_detail_kolom(df):
    if df is None:
        return

    print("=" * 110)
    print("TABEL INFORMASI DETAIL SEMUA KOLOM (INFO + CONTOH NILAI)")
    print("=" * 110)
    print(f"{'No':<3} | {'Nama Kolom':<28} | {'Tipe Data':<10} | {'Non-Null':<10} | {'Missing':<8} | {'Contoh 1 Nilai':<25}")
    print("-" * 110)

    for i, col in enumerate(df.columns, start=1):
        dtype = str(df[col].dtype)
        non_null = f"{df[col].notnull().sum():,}"
        missing = f"{df[col].isnull().sum():,}"
        # Ambil contoh 1 nilai yang tidak kosong jika memungkinkan
        valid_vals = df[col].dropna()
        contoh_val = str(valid_vals.iloc[0]) if not valid_vals.empty else "NaN"
        # Potong jika terlalu panjang
        if len(contoh_val) > 24:
            contoh_val = contoh_val[:21] + "..."

        print(f"{i:<3} | {col:<28} | {dtype:<10} | {non_null:<10} | {missing:<8} | {contoh_val:<25}")

    print("=" * 110 + "\n")

if __name__ == "__main__":
    df = baca_dataset(dataset_path)
    jawab_pertanyaan_siapa_yang_diukur(df)
    cek_jenis_kelamin(df)
    cek_target_pass_status(df)
    cek_performance_grade(df)
    tampilkan_info_detail_kolom(df)
