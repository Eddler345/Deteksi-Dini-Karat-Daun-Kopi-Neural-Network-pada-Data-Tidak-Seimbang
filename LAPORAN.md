# Laporan Studi Kasus Deep Learning
## Deteksi Dini Karat Daun Kopi: Neural Network pada Data Tidak Seimbang

**Mata Kuliah:** Deep Learning

---

## 1. Latar Belakang
Karat daun (*Hemileia vastatrix*) adalah salah satu penyakit paling merusak pada tanaman kopi dan dapat menurunkan hasil panen secara signifikan. Deteksi dini membantu petani mengambil tindakan sebelum penyakit menyebar. Pada data nyata, daun sakit biasanya jauh lebih sedikit daripada daun sehat (*class imbalance*), sehingga model yang hanya mengejar akurasi bisa terlihat bagus padahal gagal mendeteksi daun sakit.

## 2. Rumusan Masalah
1. Bagaimana mencegah *data leakage* saat membagi dan memproses data?
2. Bagaimana *class weight* memengaruhi kemampuan model mendeteksi kelas minoritas?
3. Metrik apa yang tepat untuk menilai model pada kasus ini?

## 3. Tujuan
Membangun klasifikasi biner (Sehat vs Karat Daun) dengan Neural Network, serta membandingkan model tanpa dan dengan *class weight* menggunakan recall sebagai metrik utama.

## 4. Data
Data berupa **simulasi** (`sklearn.datasets.make_classification`) sebanyak 4.000 sampel dengan 8 fitur yang merepresentasikan fitur citra daun: rata_hue, rata_saturasi, rata_brightness, kontras, homogenitas, entropi, luas_bercak, rasio_tepi. Proporsi kelas ±89% sehat dan ±11% karat daun.

> Catatan: data simulasi dipakai agar eksperimen dapat direproduksi. Hasil pada data lapangan dapat berbeda.

## 5. Metodologi
### 5.1 Pencegahan Data Leakage
- **Stratified split** 64% latih (2.560), 16% validasi (640), 20% uji (800). Proporsi kelas sakit terjaga di ketiga subset (±11%).
- Pemeriksaan otomatis (`assert`) bahwa tidak ada indeks yang sama antar subset.
- `StandardScaler` **hanya di-fit pada data latih**, lalu dipakai untuk mentransformasi data validasi dan uji.

### 5.2 Arsitektur Model
Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid). Optimizer Adam (lr 0,001), loss *binary cross-entropy*, batch 64, maksimal 100 epoch dengan *early stopping* (patience 10, monitor `val_loss`).

### 5.3 Penanganan Class Imbalance
Bobot kelas dihitung dengan `compute_class_weight("balanced")` dari data latih: kelas Sehat = 0,56 dan kelas Karat Daun = 4,56. Dua skenario dibandingkan: tanpa dan dengan *class weight*.

### 5.4 Evaluasi
Confusion matrix, precision, recall, F1-score pada data uji. Recall kelas Karat Daun menjadi metrik utama karena kegagalan mendeteksi daun sakit (*false negative*) lebih merugikan daripada alarm palsu.

## 6. Hasil
| Skenario | Akurasi | Precision (Sakit) | **Recall (Sakit)** | F1 (Sakit) | FN | FP |
|---|---|---|---|---|---|---|
| Tanpa Class Weight | 0,945 | 0,844 | 0,614 | 0,711 | 34 | 10 |
| Dengan Class Weight | 0,920 | 0,590 | **0,898** | 0,712 | 9 | 55 |

Confusion matrix (baris = aktual, kolom = prediksi, urutan Sehat, Karat Daun):
- Tanpa class weight: [[702, 10], [34, 54]]
- Dengan class weight: [[657, 55], [9, 79]]

Visualisasi tersimpan pada `hasil/confusion_matrix.png`.

![Confusion Matrix](hasil/confusion_matrix.png)

## 7. Pembahasan
- Model **tanpa class weight** memiliki akurasi lebih tinggi (94,5%), tetapi hanya menangkap 54 dari 88 daun sakit (recall 61,4%). Akurasi menipu karena kelas mayoritas mendominasi.
- Model **dengan class weight** menangkap 79 dari 88 daun sakit (recall 89,8%), kenaikan sekitar 28 poin persentase. *False negative* turun dari 34 menjadi 9.
- Konsekuensinya, *false positive* naik (10 menjadi 55) dan precision turun. Pada deteksi penyakit tanaman, trade-off ini umumnya dapat diterima: daun sehat yang diperiksa ulang lebih murah daripada penyakit yang terlewat.
- F1-score kedua model relatif sama (±0,71), sehingga pemilihan model ditentukan oleh prioritas bisnis, yaitu recall.
- Hasil dapat bervariasi tipis antar eksekusi karena inisialisasi bobot dan operasi numerik TensorFlow; kesimpulan arah (recall naik dengan class weight) konsisten.

## 8. Kesimpulan
1. Stratified split dan scaler yang di-fit hanya pada data latih mencegah *data leakage*.
2. *Class weight* meningkatkan recall kelas minoritas secara jelas (61,4% menjadi 89,8%) dengan konsekuensi penurunan precision.
3. Pada data tidak seimbang, recall dan confusion matrix lebih informatif daripada akurasi.

## 9. Saran
Gunakan dataset citra daun kopi nyata, coba CNN pada gambar langsung, tuning *threshold* keputusan, serta bandingkan dengan SMOTE atau *focal loss*.

## 10. Cara Reproduksi
```bash
pip install -r requirements.txt
python main.py
```
