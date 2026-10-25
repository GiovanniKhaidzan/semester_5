# ==============================================================================
# Tugas Pembelajaran Mesin: Klasifikasi dengan Regresi Logistik dari NOL (Bagian 2)
# File: bagian2_logistic_regression.py
# Deskripsi: Mengimplementasikan regresi logistik tanpa pustaka sklearn (hanya pandas & numpy/math)
#            menggunakan pembagian dataset penuh: 80% data latih dan 20% data uji.
#            Tampilan terminal dibatasi menampilkan sampel 5 data agar rapi dan mudah dibaca.
# ==============================================================================

import os
import math
import numpy as np
import pandas as pd

# Path dataset (disesuaikan agar mendukung eksekusi dari root direktori maupun folder saat ini)
CSV_PATH = "data/student_exam_performance.csv"
if not os.path.exists(CSV_PATH) and os.path.exists("student_exam_performance.csv"):
    CSV_PATH = "student_exam_performance.csv"

# Pemetaan kategori ke angka (sama persis dengan Bagian 1)
DIFFICULTY_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}
STATUS_MAP = {"Pass": 1, "Fail": 0}
REVERSE_STATUS_MAP = {1: "Pass", 0: "Fail"}

# Jumlah baris sampel yang ditampilkan di terminal tabel agar rapi
JUMLAH_TAMPIL = 5

# Seed acak untuk pembagian data yang konsisten (reproducible)
RANDOM_STATE = 42


def muat_dan_praproses_data(csv_path, rasio_latih=0.8):
    """
    Memuat dataset penuh dari CSV, membagi menjadi rasio_latih (80% latih, 20% uji),
    serta melakukan praproses fitur numerik (f1, f2, dan y).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {csv_path}")

    # Membaca kolom yang diperlukan
    kolom_digunakan = ["student_id", "previous_exam_score", "exam_difficulty", "pass_status"]
    df = pd.read_csv(csv_path, usecols=kolom_digunakan)

    # Praproses:
    # f1 = previous_exam_score / 100
    # f2 = exam_difficulty (Easy=0, Medium=1, Hard=2)
    # y = pass_status (Pass=1, Fail=0)
    df["f1"] = df["previous_exam_score"] / 100.0
    df["f2"] = df["exam_difficulty"].map(DIFFICULTY_MAP)
    df["y"] = df["pass_status"].map(STATUS_MAP)

    # Pembagian 80% data latih dan 20% data uji secara acak yang konsisten
    df_acak = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = len(df_acak)
    n_latih = int(n_total * rasio_latih)

    df_train = df_acak.iloc[:n_latih].copy().reset_index(drop=True)
    df_test = df_acak.iloc[n_latih:].copy().reset_index(drop=True)

    # Tampilkan tabel awal sebelum training (5 sampel data latih & 5 sampel data uji)
    tampilkan_tabel_awal(df_train, df_test)

    return df_train, df_test


def tampilkan_tabel_awal(df_train, df_test):
    """
    Menampilkan tabel sampel data sebelum training.
    Sesuai instruksi: hanya 5 data pertama yang ditampilkan di terminal,
    dan pass_status pada data uji disembunyikan/dirahasiakan.
    """
    print("\n" + "=" * 84)
    print(f"TABEL DATA SEBELUM TRAINING (Total: {len(df_train)} Latih [80%], {len(df_test)} Uji [20%])")
    print(f"Menampilkan {JUMLAH_TAMPIL} Sampel Teratas untuk Setiap Kategori")
    print("=" * 84)
    header = f"{'Jenis Data':<12} | {'student_id':<12} | {'previous_exam_score':<20} | {'exam_difficulty':<16} | {'pass_status':<14}"
    print(header)
    print("-" * len(header))

    # 5 Sampel data latih
    for _, row in df_train.head(JUMLAH_TAMPIL).iterrows():
        print(f"{'Latih':<12} | {row['student_id']:<12} | {row['previous_exam_score']:<20.2f} | {row['exam_difficulty']:<16} | {row['pass_status']:<14}")
    sisa_latih = len(df_train) - JUMLAH_TAMPIL
    print(f"... ({sisa_latih} data latih lainnya tidak ditampilkan) ...")
    print("-" * len(header))

    # 5 Sampel data uji (status kelulusan dirahasiakan)
    for _, row in df_test.head(JUMLAH_TAMPIL).iterrows():
        print(f"{'Uji':<12} | {row['student_id']:<12} | {row['previous_exam_score']:<20.2f} | {row['exam_difficulty']:<16} | {'(Dirahasiakan)':<14}")
    sisa_uji = len(df_test) - JUMLAH_TAMPIL
    print(f"... ({sisa_uji} data uji lainnya tidak ditampilkan) ...")
    print("=" * 84 + "\n")


def sigmoid(z):
    """
    Fungsi aktivasi sigmoid yang aman dari overflow numerik.
    sigma(z) = 1 / (1 + e^(-z))
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def hitung_epoch(epoch_num, df_train, w1, w2, b, eta):
    """
    Menjalankan alur 1 epoch regresi logistik pada seluruh data latih (80.000 data):
    1. z = w1*f1 + w2*f2 + b
    2. sigma = 1 / (1 + e^(-z))
    3. Loss: L = -[ y*ln(sigma) + (1-y)*ln(1-sigma) ]
    4. error = sigma - y
    5. w1_grad = error * f1 ; w2_grad = error * f2

    Di terminal, hanya 5 data pertama yang dicetak sebagai representasi tabel,
    diikuti baris TOTAL dan kalkulasi rata-rata gradien (dibagi N).
    """
    N = len(df_train)
    print("=" * 105)
    print(f"EPOCH {epoch_num} (Total Data Latih: {N})")
    print(f"Bobot Awal: w1 = {w1:.5f}, w2 = {w2:.5f}, b = {b:.5f}, eta = {eta}")
    print("=" * 105)

    headers = ["No", "student_id", "f1", "f2", "z", "sigma", "y", "L", "(sigma-y)", "w1", "w2"]
    header_str = f"{headers[0]:<3} | {headers[1]:<10} | {headers[2]:<8} | {headers[3]:<4} | {headers[4]:<9} | {headers[5]:<9} | {headers[6]:<3} | {headers[7]:<9} | {headers[8]:<10} | {headers[9]:<10} | {headers[10]:<10}"
    print(header_str)
    print("-" * len(header_str))

    # Ekstraksi array fitur untuk perhitungan vektorisasi numpy yang cepat dan presisi penuh
    f1_arr = df_train["f1"].to_numpy(dtype=np.float64)
    f2_arr = df_train["f2"].to_numpy(dtype=np.float64)
    y_arr = df_train["y"].to_numpy(dtype=np.float64)

    # 1. z = w1*f1 + w2*f2 + b
    z_arr = w1 * f1_arr + w2 * f2_arr + b

    # 2. sigma = 1 / (1 + e^(-z))
    sig_arr = sigmoid(z_arr)

    # 3. Loss L = -[ y*ln(sigma) + (1-y)*ln(1-sigma) ] (menggunakan batas epsilon untuk stabilitas log)
    eps = 1e-15
    sig_clipped = np.clip(sig_arr, eps, 1.0 - eps)
    loss_arr = -(y_arr * np.log(sig_clipped) + (1.0 - y_arr) * np.log(1.0 - sig_clipped))

    # 4. error = sigma - y
    err_arr = sig_arr - y_arr

    # 5. kolom w1 dan w2
    kw1_arr = err_arr * f1_arr
    kw2_arr = err_arr * f2_arr

    # Cetak 5 sampel data pertama ke tabel
    for i in range(JUMLAH_TAMPIL):
        sid = df_train.at[i, "student_id"]
        row_str = (
            f"{i+1:<3} | {sid:<10} | {f1_arr[i]:<8.5f} | {int(f2_arr[i]):<4} | "
            f"{z_arr[i]:<9.5f} | {sig_arr[i]:<9.5f} | {int(y_arr[i]):<3} | "
            f"{loss_arr[i]:<9.5f} | {err_arr[i]:<10.5f} | "
            f"{kw1_arr[i]:<10.5f} | {kw2_arr[i]:<10.5f}"
        )
        print(row_str)

    sisa = N - JUMLAH_TAMPIL
    print(f"... [{sisa} baris data latih lainnya dihitung secara penuh] ...")
    print("-" * len(header_str))

    # Hitung Total pada SELURUH N data latih
    total_L = float(np.sum(loss_arr))
    total_error = float(np.sum(err_arr))
    total_w1 = float(np.sum(kw1_arr))
    total_w2 = float(np.sum(kw2_arr))

    # Baris Total
    total_label = f"TOTAL ({N} data)"
    total_str = (
        f"{total_label:<30} | {'':<9} | {'':<9} | {'':<3} | "
        f"{total_L:<9.5f} | {total_error:<10.5f} | {total_w1:<10.5f} | {total_w2:<10.5f}"
    )
    print(total_str)
    print("-" * len(header_str))

    # Hitung rata-rata gradien dan Cost J (dibagi N)
    J = total_L / N
    dw1 = total_w1 / N
    dw2 = total_w2 / N
    db = total_error / N

    # Update bobot (gradient descent)
    w1_baru = w1 - eta * dw1
    w2_baru = w2 - eta * dw2
    b_baru = b - eta * db

    print(f"J (Cost / Rata-rata Loss) = Total L / {N}         = {J:.5f}")
    print(f"dw1 (Rata-rata gradien w1) = Total w1 / {N}      = {dw1:.5f}")
    print(f"dw2 (Rata-rata gradien w2) = Total w2 / {N}      = {dw2:.5f}")
    print(f"db  (Rata-rata error b)    = Total(sigma-y) / {N} = {db:.5f}")
    print(f"w1 baru = w1 - eta * dw1  = {w1:.5f} - ({eta} * {dw1:.5f}) = {w1_baru:.5f}")
    print(f"w2 baru = w2 - eta * dw2  = {w2:.5f} - ({eta} * {dw2:.5f}) = {w2_baru:.5f}")
    print(f"b baru  = b  - eta * db   = {b:.5f} - ({eta} * {db:.5f})  = {b_baru:.5f}")
    print("=" * 105 + "\n")

    return w1_baru, w2_baru, b_baru


def uji(df_test, w1, w2, b):
    """
    Menguji model dengan bobot akhir pada seluruh data uji (20.000 data):
    - z = w1*f1 + w2*f2 + b
    - P(y=1|x) = sigma(z)
    - P(y=0|x) = 1 - sigma(z)
    - Keputusan: jika P(y=1|x) >= 0.5 maka Pass (1), selain itu Fail (0)
    - Menampilkan tabel 5 sampel data uji dan ringkasan metrik akurasi keseluruhan.
    """
    N_test = len(df_test)
    print("=" * 90)
    print(f"PENGUJIAN MODEL (DATA UJI - TOTAL: {N_test} DATA)")
    print(f"Bobot yang Digunakan: w1 = {w1:.5f}, w2 = {w2:.5f}, b = {b:.5f}")
    print("=" * 90)

    # Ekstraksi array data uji
    f1_arr = df_test["f1"].to_numpy(dtype=np.float64)
    f2_arr = df_test["f2"].to_numpy(dtype=np.float64)
    y_asli_arr = df_test["y"].to_numpy(dtype=np.int64)

    # Hitung z dan probabilitas
    z_arr = w1 * f1_arr + w2 * f2_arr + b
    p_pass_arr = sigmoid(z_arr)
    p_fail_arr = 1.0 - p_pass_arr

    # Keputusan klasifikasi
    pred_arr = (p_pass_arr >= 0.5).astype(np.int64)
    evaluasi_arr = (pred_arr == y_asli_arr)

    # Tampilkan tabel pengujian untuk 5 sampel pertama
    header = f"{'student_id':<12} | {'z':<9} | {'P(y=1|x)':<10} | {'P(y=0|x)':<10} | {'Prediksi':<10} | {'Asli':<10} | {'Evaluasi':<10}"
    print(header)
    print("-" * len(header))
    for i in range(JUMLAH_TAMPIL):
        sid = df_test.at[i, "student_id"]
        pred_str = REVERSE_STATUS_MAP[pred_arr[i]]
        asli_str = REVERSE_STATUS_MAP[y_asli_arr[i]]
        eval_str = "Benar" if evaluasi_arr[i] else "Salah"

        row_str = (
            f"{sid:<12} | {z_arr[i]:<9.5f} | {p_pass_arr[i]:<10.5f} | "
            f"{p_fail_arr[i]:<10.5f} | {pred_str:<10} | {asli_str:<10} | {eval_str:<10}"
        )
        print(row_str)

    sisa_test = N_test - JUMLAH_TAMPIL
    print(f"... [{sisa_test} baris data uji lainnya dihitung secara penuh] ...")
    print("-" * len(header))

    # Ringkasan Evaluasi Keseluruhan Data Uji
    jumlah_benar = int(np.sum(evaluasi_arr))
    jumlah_salah = N_test - jumlah_benar
    akurasi = (jumlah_benar / N_test) * 100.0

    print("\nRINGKASAN EVALUASI KESELURUHAN DATA UJI:")
    print(f"- Total Data Uji       : {N_test}")
    print(f"- Prediksi Benar       : {jumlah_benar} data")
    print(f"- Prediksi Salah       : {jumlah_salah} data")
    print(f"- Akurasi Model        : {akurasi:.2f}%")
    print("=" * 90 + "\n")


def main():
    # 1. Memuat dan Mempraproses Data (80% Latih, 20% Uji)
    df_train, df_test = muat_dan_praproses_data(CSV_PATH, rasio_latih=0.8)

    # 2. Parameter Awal (sama dengan Bagian 1)
    w1 = 0.9
    w2 = -0.1
    b = 0.1
    eta = 0.1

    # 3. Training Per Epoch (2 Epoch)
    # Epoch 1
    w1_ep1, w2_ep1, b_ep1 = hitung_epoch(1, df_train, w1, w2, b, eta)

    # Epoch 2 (menggunakan bobot hasil epoch 1)
    w1_ep2, w2_ep2, b_ep2 = hitung_epoch(2, df_train, w1_ep1, w2_ep1, b_ep1, eta)

    # 4. Pengujian dengan bobot akhir epoch 2
    uji(df_test, w1_ep2, w2_ep2, b_ep2)


if __name__ == "__main__":
    main()
