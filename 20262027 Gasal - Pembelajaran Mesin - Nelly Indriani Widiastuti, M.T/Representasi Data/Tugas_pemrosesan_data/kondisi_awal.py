import pandas as pd

# 1. Load Dataset
file_path = "data/student_exam_performance.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("INFORMASI AWAL DATASET")
print("=" * 60)

# 2. Cek Jumlah Baris dan Kolom
jumlah_baris, jumlah_kolom = df.shape
print(f"Jumlah Baris : {jumlah_baris:,}")
print(f"Jumlah Kolom : {jumlah_kolom}")
print("-" * 60)

# 3. Cek Missing Value (Nilai Kosong / NaN)
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print("INFORMASI MISSING VALUE:")
if total_missing == 0:
    print("Tidak ditemukan missing value pada dataset.")
else:
    print(f"Total missing values: {total_missing}")
    kolom_missing = missing_values[missing_values > 0]
    for col, val in kolom_missing.items():
        persen = (val / jumlah_baris) * 100
        print(f"- {col}: {val} ({persen:.2f}%)")
print("-" * 60)

# 4. Cek Data Duplikat
jumlah_duplikat = df.duplicated().sum()
print("INFORMASI DATA DUPLIKAT:")
if jumlah_duplikat == 0:
    print("Tidak ada baris duplikat.")
else:
    print(f"Ditemukan {jumlah_duplikat} baris duplikat.")
print("-" * 60)

# 5. Cek Distribusi Kelas Target
# Kolom target potensial dalam dataset: 'pass_status', 'performance_grade', 'performance_level'
target_columns = ["pass_status", "performance_grade", "performance_level"]

print("DISTRIBUSI KELAS TARGET:")
for target in target_columns:
    if target in df.columns:
        print(f"\n--- Distribusi Target: '{target}' ---")
        counts = df[target].value_counts()
        percentages = df[target].value_counts(normalize=True) * 100
        summary_target = pd.DataFrame({"Jumlah": counts, "Persentase (%)": percentages.round(2)})
        print(summary_target)
print("=" * 60)
