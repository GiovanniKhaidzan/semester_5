# ==============================================================================
# Tugas Pembelajaran Mesin: Klasifikasi dengan Regresi Logistik dari NOL
# File: bagian1_logistic_regression.py
# Deskripsi: Mengimplementasikan regresi logistik tanpa pustaka sklearn (hanya pandas & math/numpy).
# ==============================================================================

import os
import math
import pandas as pd

# Path dataset (disesuaikan agar mendukung eksekusi dari root direktori maupun folder saat ini)
CSV_PATH = "data/student_exam_performance.csv"
if not os.path.exists(CSV_PATH) and os.path.exists("student_exam_performance.csv"):
    CSV_PATH = "student_exam_performance.csv"

# Daftar student_id yang digunakan
TRAIN_STUDENT_IDS = [
    "STU_033737",
    "STU_028639",
    "STU_058955",
    "STU_035620",
    "STU_073090"
]
TEST_STUDENT_ID = "STU_087159"

# Pemetaan kategori ke angka
DIFFICULTY_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}
STATUS_MAP = {"Pass": 1, "Fail": 0}
REVERSE_STATUS_MAP = {1: "Pass", 0: "Fail"}


def muat_dan_praproses_data(csv_path):
    """
    Memuat dataset dari CSV, memfilter 5 data latih dan 1 data uji,
    memvalidasi nilai yang diharapkan dengan assert, serta melakukan praproses fitur.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter data latih dan data uji
    all_target_ids = TRAIN_STUDENT_IDS + [TEST_STUDENT_ID]
    df_filtered = df[df["student_id"].isin(all_target_ids)].copy()

    # Urutkan data latih sesuai urutan yang ditentukan pada instruksi
    df_train = df_filtered[df_filtered["student_id"].isin(TRAIN_STUDENT_IDS)].copy()
    df_train["student_id"] = pd.Categorical(df_train["student_id"], categories=TRAIN_STUDENT_IDS, ordered=True)
    df_train = df_train.sort_values("student_id").reset_index(drop=True)

    # Data uji
    df_test = df_filtered[df_filtered["student_id"] == TEST_STUDENT_ID].reset_index(drop=True)

    # Nilai yang diharapkan dari dataset untuk pengecekan assert
    expected_values = {
        "STU_033737": (76.63, "Easy", "Pass"),
        "STU_028639": (65.93, "Medium", "Pass"),
        "STU_058955": (78.16, "Hard", "Pass"),
        "STU_035620": (59.65, "Hard", "Fail"),
        "STU_073090": (72.63, "Medium", "Fail"),
        "STU_087159": (50.63, "Hard", "Fail"),
    }

    # Pengecekan kebenaran data dengan assert
    for _, row in pd.concat([df_train, df_test]).iterrows():
        sid = row["student_id"]
        exp_score, exp_diff, exp_status = expected_values[sid]
        assert abs(row["previous_exam_score"] - exp_score) < 1e-4, f"Data score {sid} tidak sesuai!"
        assert row["exam_difficulty"] == exp_diff, f"Data difficulty {sid} tidak sesuai!"
        assert row["pass_status"] == exp_status, f"Data pass_status {sid} tidak sesuai!"

    # Tampilkan tabel awal sebelum praproses numerik
    tampilkan_tabel_awal(df_train, df_test)

    # Praproses:
    # f1 = previous_exam_score / 100
    # f2 = exam_difficulty (Easy=0, Medium=1, Hard=2)
    # y = pass_status (Pass=1, Fail=0)
    for d in [df_train, df_test]:
        d["f1"] = d["previous_exam_score"] / 100.0
        d["f2"] = d["exam_difficulty"].map(DIFFICULTY_MAP)
        d["y"] = d["pass_status"].map(STATUS_MAP)

    return df_train, df_test


def tampilkan_tabel_awal(df_train, df_test):
    """
    Menampilkan tabel 5 data latih dan 1 data uji sebelum training.
    Catatan: pass_status pada data uji disembunyikan/tidak ditampilkan.
    """
    print("\n" + "=" * 80)
    print("TABEL DATA SEBELUM TRAINING (5 Data Latih & 1 Data Uji)")
    print("=" * 80)
    header = f"{'Jenis Data':<12} | {'student_id':<12} | {'previous_exam_score':<20} | {'exam_difficulty':<16} | {'pass_status':<12}"
    print(header)
    print("-" * len(header))

    for _, row in df_train.iterrows():
        print(f"{'Latih':<12} | {row['student_id']:<12} | {row['previous_exam_score']:<20.2f} | {row['exam_difficulty']:<16} | {row['pass_status']:<12}")

    for _, row in df_test.iterrows():
        print(f"{'Uji':<12} | {row['student_id']:<12} | {row['previous_exam_score']:<20.2f} | {row['exam_difficulty']:<16} | {'(Dirahasiakan)':<12}")
    print("=" * 80 + "\n")


def sigmoid(z):
    """
    Fungsi aktivasi sigmoid: sigma(z) = 1 / (1 + e^(-z))
    """
    return 1.0 / (1.0 + math.exp(-z))


def hitung_epoch(epoch_num, df_train, w1, w2, b, eta):
    """
    Menjalankan alur 1 epoch regresi logistik:
    1. z = w1*f1 + w2*f2 + b
    2. sigma = 1 / (1 + e^(-z))
    3. Loss: L = -[ y*ln(sigma) + (1-y)*ln(1-sigma) ]
    4. error = sigma - y
    5. w1_grad = error * f1 ; w2_grad = error * f2

    Mengembalikan: w1_baru, w2_baru, b_baru
    """
    print("=" * 105)
    print(f"EPOCH {epoch_num}")
    print(f"Bobot Awal: w1 = {w1:.5f}, w2 = {w2:.5f}, b = {b:.5f}, eta = {eta}")
    print("=" * 105)

    headers = ["No", "student_id", "f1", "f2", "z", "sigma", "y", "L", "(sigma-y)", "w1", "w2"]
    header_str = f"{headers[0]:<3} | {headers[1]:<10} | {headers[2]:<8} | {headers[3]:<4} | {headers[4]:<9} | {headers[5]:<9} | {headers[6]:<3} | {headers[7]:<9} | {headers[8]:<10} | {headers[9]:<10} | {headers[10]:<10}"
    print(header_str)
    print("-" * len(header_str))

    baris_hasil = []
    total_L = 0.0
    total_error = 0.0
    total_w1 = 0.0
    total_w2 = 0.0

    z_data_1 = None
    sigma_data_1 = None

    for idx, row in df_train.iterrows():
        f1 = float(row["f1"])
        f2 = float(row["f2"])
        y = int(row["y"])

        # 1. z = w1*f1 + w2*f2 + b
        z = w1 * f1 + w2 * f2 + b

        # 2. sigma = 1 / (1 + e^(-z))
        sig = sigmoid(z)

        # 3. Loss L = -[ y*ln(sigma) + (1-y)*ln(1-sigma) ]
        loss = -(y * math.log(sig) + (1.0 - y) * math.log(1.0 - sig))

        # 4. error = sigma - y
        err = sig - y

        # 5. kolom w1 dan w2
        kolom_w1 = err * f1
        kolom_w2 = err * f2

        if idx == 0:
            z_data_1 = z
            sigma_data_1 = sig

        total_L += loss
        total_error += err
        total_w1 += kolom_w1
        total_w2 += kolom_w2

        baris_hasil.append({
            "idx": idx + 1,
            "student_id": row["student_id"],
            "f1": f1,
            "f2": f2,
            "z": z,
            "sigma": sig,
            "y": y,
            "L": loss,
            "error": err,
            "kolom_w1": kolom_w1,
            "kolom_w2": kolom_w2
        })

        row_str = (
            f"{idx+1:<3} | {row['student_id']:<10} | {f1:<8.5f} | {int(f2):<4} | "
            f"{z:<9.5f} | {sig:<9.5f} | {y:<3} | {loss:<9.5f} | {err:<10.5f} | "
            f"{kolom_w1:<10.5f} | {kolom_w2:<10.5f}"
        )
        print(row_str)

    print("-" * len(header_str))

    # Baris Total
    total_str = (
        f"{'TOTAL':<30} | {'':<9} | {'':<9} | {'':<3} | "
        f"{total_L:<9.5f} | {total_error:<10.5f} | {total_w1:<10.5f} | {total_w2:<10.5f}"
    )
    print(total_str)
    print("-" * len(header_str))

    # Hitung rata-rata gradien dan Cost J
    N = len(df_train)
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
    Menguji model dengan bobot akhir pada data uji (STU_087159):
    - z = w1*f1 + w2*f2 + b
    - P(y=1|x) = sigma(z)
    - P(y=0|x) = 1 - sigma(z)
    - Keputusan: jika P(y=1|x) >= 0.5 maka Pass (1), selain itu Fail (0)
    - Menampilkan tabel hasil pengujian
    """
    print("=" * 90)
    print("PENGUJIAN MODEL (DATA UJI)")
    print(f"Bobot yang Digunakan: w1 = {w1:.5f}, w2 = {w2:.5f}, b = {b:.5f}")
    print("=" * 90)

    hasil_uji_list = []
    for _, row in df_test.iterrows():
        sid = row["student_id"]
        f1 = float(row["f1"])
        f2 = float(row["f2"])
        y_asli_num = int(row["y"])
        y_asli_str = row["pass_status"]

        # Hitung z dan probabilitas
        z = w1 * f1 + w2 * f2 + b
        p_pass = sigmoid(z)
        p_fail = 1.0 - p_pass

        # Keputusan klasifikasi
        pred_num = 1 if p_pass >= 0.5 else 0
        pred_str = REVERSE_STATUS_MAP[pred_num]
        evaluasi = "Benar" if pred_num == y_asli_num else "Salah"

        hasil = {
            "student_id": sid,
            "f1": f1,
            "f2": f2,
            "z": z,
            "p_pass": p_pass,
            "p_fail": p_fail,
            "prediksi": pred_str,
            "asli": y_asli_str,
            "evaluasi": evaluasi
        }
        hasil_uji_list.append(hasil)

    # Tampilkan tabel pengujian
    header = f"{'student_id':<12} | {'z':<9} | {'P(y=1|x)':<10} | {'P(y=0|x)':<10} | {'Prediksi':<10} | {'Asli':<10} | {'Evaluasi':<10}"
    print(header)
    print("-" * len(header))
    for h in hasil_uji_list:
        row_str = (
            f"{h['student_id']:<12} | {h['z']:<9.5f} | {h['p_pass']:<10.5f} | "
            f"{h['p_fail']:<10.5f} | {h['prediksi']:<10} | {h['asli']:<10} | {h['evaluasi']:<10}"
        )
        print(row_str)
    print("=" * 90 + "\n")

    return hasil_uji_list[0]


def main():
    # 1. Memuat dan Mempraproses Data
    df_train, df_test = muat_dan_praproses_data(CSV_PATH)

    # 2. Parameter Awal
    w1 = 0.9
    w2 = -0.1
    b = 0.1
    eta = 0.1

    # 3. Training Per Epoch
    # Epoch 1
    w1_ep1, w2_ep1, b_ep1 = hitung_epoch(1, df_train, w1, w2, b, eta)

    # Epoch 2 (menggunakan bobot hasil epoch 1)
    w1_ep2, w2_ep2, b_ep2 = hitung_epoch(2, df_train, w1_ep1, w2_ep1, b_ep1, eta)

    # 4. Pengujian dengan bobot akhir epoch 2
    uji(df_test, w1_ep2, w2_ep2, b_ep2)


if __name__ == "__main__":
    main()

