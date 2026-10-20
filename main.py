"""
Deteksi Dini Karat Daun Kopi dengan Neural Network
--------------------------------------------------
Fokus studi kasus:
1. Mencegah data leakage  -> stratified split + scaler hanya di-fit pada data latih
2. Menangani class imbalance -> class weight saat training
3. Evaluasi berbasis recall -> kelas minoritas (daun sakit) harus terdeteksi

CATATAN: dataset di sini adalah data SIMULASI (make_classification) agar skrip
bisa langsung dijalankan. Untuk laporan akhir, ganti fungsi `muat_data()`
dengan dataset nyata (mis. CSV fitur citra daun kopi dari Kaggle/lapangan).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FITUR = ["rata_hue", "rata_saturasi", "rata_brightness", "kontras",
         "homogenitas", "entropi", "luas_bercak", "rasio_tepi"]
KELAS = ["Sehat", "Karat Daun"]


def muat_data():
    """Data simulasi: 90% daun sehat, 10% daun terkena karat (imbalanced)."""
    X, y = make_classification(
        n_samples=4000, n_features=len(FITUR), n_informative=5,
        n_redundant=1, weights=[0.90, 0.10], class_sep=0.9,
        flip_y=0.02, random_state=SEED,
    )
    return pd.DataFrame(X, columns=FITUR), pd.Series(y, name="label")


def bagi_data(X, y):
    """Stratified split 64/16/20 (latih/validasi/uji) + cek leakage."""
    idx = np.arange(len(X))
    idx_tmp, idx_test = train_test_split(
        idx, test_size=0.20, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_tmp, test_size=0.20, stratify=y.iloc[idx_tmp], random_state=SEED)

    # Pemeriksaan leakage: tidak boleh ada indeks yang sama antar subset
    assert not set(idx_train) & set(idx_val)
    assert not set(idx_train) & set(idx_test)
    assert not set(idx_val) & set(idx_test)

    # Scaler HANYA di-fit pada data latih
    scaler = StandardScaler().fit(X.iloc[idx_train])
    f = lambda i: scaler.transform(X.iloc[i])
    return (f(idx_train), y.iloc[idx_train].values,
            f(idx_val), y.iloc[idx_val].values,
            f(idx_test), y.iloc[idx_test].values)


def bangun_model(n_fitur):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_fitur,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.Recall(name="recall")])
    return model


def latih(Xtr, ytr, Xva, yva, pakai_class_weight):
    cw = None
    if pakai_class_weight:
        bobot = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        cw = {int(k): float(v) for k, v in zip(np.unique(ytr), bobot)}
        print(f"  Class weight: {cw}")
    model = bangun_model(Xtr.shape[1])
    model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=64,
        class_weight=cw, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)],
    )
    return model


def evaluasi(nama, model, Xte, yte):
    pred = (model.predict(Xte, verbose=0).ravel() >= 0.5).astype(int)
    print(f"\n=== {nama} ===")
    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=KELAS, digits=3))
    return recall_score(yte, pred, pos_label=1)


def simpan_grafik(m0, m1, Xte, yte):
    """Simpan confusion matrix kedua model ke folder hasil/."""
    os.makedirs("hasil", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, (nama, m) in zip(ax, [("Tanpa Class Weight", m0), ("Dengan Class Weight", m1)]):
        pred = (m.predict(Xte, verbose=0).ravel() >= 0.5).astype(int)
        cm = confusion_matrix(yte, pred)
        a.imshow(cm, cmap="Greens")
        a.set_title(nama)
        a.set_xticks([0, 1]); a.set_xticklabels(KELAS)
        a.set_yticks([0, 1]); a.set_yticklabels(KELAS)
        a.set_xlabel("Prediksi"); a.set_ylabel("Aktual")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig("hasil/confusion_matrix.png", dpi=150)
    plt.close()


def main():
    X, y = muat_data()
    print("Distribusi kelas:\n", y.value_counts(normalize=True).round(3), "\n")

    Xtr, ytr, Xva, yva, Xte, yte = bagi_data(X, y)
    print(f"Latih={len(ytr)}  Validasi={len(yva)}  Uji={len(yte)}")
    print(f"Proporsi kelas 1 -> latih {ytr.mean():.3f} | "
          f"val {yva.mean():.3f} | uji {yte.mean():.3f}\n")

    print("Melatih TANPA class weight...")
    m0 = latih(Xtr, ytr, Xva, yva, pakai_class_weight=False)
    print("Melatih DENGAN class weight...")
    m1 = latih(Xtr, ytr, Xva, yva, pakai_class_weight=True)

    r0 = evaluasi("Tanpa Class Weight", m0, Xte, yte)
    r1 = evaluasi("Dengan Class Weight", m1, Xte, yte)

    simpan_grafik(m0, m1, Xte, yte)
    print(f"\nRecall Karat Daun: tanpa CW = {r0:.3f} | dengan CW = {r1:.3f}")


if __name__ == "__main__":
    main()
