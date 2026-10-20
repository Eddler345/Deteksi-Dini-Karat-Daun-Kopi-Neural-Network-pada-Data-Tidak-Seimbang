# Deteksi Dini Karat Daun Kopi: Neural Network pada Data Tidak Seimbang

Studi kasus Deep Learning untuk mengklasifikasikan daun kopi **sehat** vs
**terkena karat daun** (*Hemileia vastatrix*) dari fitur citra, dengan fokus pada
tiga praktik penting dalam data tidak seimbang.

## Fitur Proyek

- **Pencegahan Data Leakage**: *stratified split* (latih/validasi/uji), pemeriksaan
  indeks yang tumpang tindih, dan `StandardScaler` yang hanya di-fit pada data latih.
- **Penanganan Class Imbalance**: perbandingan model tanpa dan dengan *class weight*.
- **Evaluasi Berbasis Recall**: fokus pada kemampuan model mendeteksi daun sakit
  (kelas minoritas), dilengkapi confusion matrix dan classification report.

## Cara Menjalankan

```bash
pip install -r requirements.txt
python main.py
```

## Catatan Data

Dataset pada `main.py` adalah **data simulasi** (90% sehat : 10% sakit). Untuk hasil
yang sesungguhnya, ganti fungsi `muat_data()` dengan dataset nyata.

## Arsitektur Model

Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid),
optimizer Adam, loss binary cross-entropy, *early stopping* pada `val_loss`.

## Hasil Ringkas (data uji, 800 sampel)

| Skenario | Akurasi | Recall Karat Daun |
|---|---|---|
| Tanpa Class Weight | 0,945 | 0,614 |
| Dengan Class Weight | 0,920 | 0,898 |

Detail lengkap ada di [LAPORAN.md](LAPORAN.md). Grafik ada di `hasil/confusion_matrix.png`.
