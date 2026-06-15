# Mushroom Classification dengan Support Vector Machine (SVM)

Klasifikasi Data menggunakan algoritma **Support Vector Machine (SVM)** berbasis Python di Google Colab.

---

## Dataset

**Nama Dataset:** Mushroom Classification

**Sumber:** [https://www.kaggle.com/datasets/uciml/mushroom-classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)

**Deskripsi:**
Dataset ini berisi sampel hipotetis dari 23 spesies jamur insang yang diklasifikasikan sebagai **edible (dapat dimakan)** atau **poisonous (beracun)**. Dataset terdiri dari 8.124 baris dan 23 kolom, di mana semua fitur bersifat kategorikal.

| Atribut | Keterangan |
|---------|------------|
| Jumlah sampel | 8.124 baris |
| Jumlah fitur | 22 fitur + 1 target |
| Tipe fitur | Semua kategorikal |
| Target kolom | `class` (`e` = edible, `p` = poisonous) |
| Missing value | Tidak ada |

---

## Teknologi yang Digunakan

- Python 3
- Google Colab
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Alur Pengerjaan

### 1. Preprocessing Data
Karena semua kolom bersifat kategorikal dan tidak terdapat missing value, preprocessing yang dilakukan adalah:
- **Label Encoding** — mengubah setiap nilai kategorikal menjadi angka integer menggunakan `LabelEncoder` dari scikit-learn
- **Scaling** — tidak dilakukan scaling tambahan karena model SVM dengan kernel RBF pada data yang sudah di-encode dinilai tidak sensitif terhadap perbedaan skala dalam konteks ini

### 2. Split Data
Data dibagi menjadi **80% training** dan **20% testing** menggunakan `train_test_split` dengan `random_state=42`.

```
X_train: (6499, 22)
X_test : (1625, 22)
```

### 3. Modeling SVM

Model yang digunakan adalah **SVC (Support Vector Classifier)** dengan kernel **RBF (Radial Basis Function)**.

```python
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
```

**Alasan pemilihan kernel RBF:**
Kernel RBF dipilih karena mampu menangani data yang polanya tidak linear. Kernel ini membantu SVM membedakan antar kelas dengan lebih baik dan umumnya memberikan hasil yang baik pada berbagai jenis data tanpa perlu mengetahui pola datanya terlebih dahulu.

### 4. Evaluasi Model

Evaluasi dilakukan menggunakan metrik berikut:

| Metrik | Nilai |
|--------|-------|
| Accuracy | *(lihat output notebook)* |
| Precision | *(lihat output notebook)* |
| Recall | *(lihat output notebook)* |
| F1 Score | *(lihat output notebook)* |

**Analisis overfitting/underfitting:**
Jika akurasi pada data latih tinggi (mendekati 1.0) dan akurasi pada data pengujian juga tinggi serta tidak berbeda jauh, maka model tidak menunjukkan tanda overfitting maupun underfitting yang signifikan — artinya model mampu menggeneralisasi dengan baik pada data baru.

### 5. Visualisasi

**Visualisasi 1 — Distribusi Kelas Jamur**

Bar chart yang menampilkan jumlah sampel untuk masing-masing kelas (`edible` vs `poisonous`).

**Visualisasi 2 — Confusion Matrix**

Heatmap confusion matrix yang menampilkan perbandingan antara label aktual dan label hasil prediksi model.

---

## Cara Menjalankan

1. Buka file `SVM.ipynb` di [Google Colab](https://colab.research.google.com/)
2. Upload file `mushrooms.csv` ke direktori `/content/sample_data/` di Colab
3. Jalankan semua cell secara berurutan (**Runtime → Run all**)

---

## Struktur Repository

```
├── SVM.ipynb        # Notebook utama
└── README.md        # Dokumentasi proyek
```

---

## 📚 Referensi

- Dataset: [UCI Mushroom Classification — Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- [scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
