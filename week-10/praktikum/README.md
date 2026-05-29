# Praktikum Week 10: Extensions of K-Means

Folder ini berisi demo Python untuk materi `K-Means-Extensions.md`.

## Dataset yang Dipakai

Praktikum ini memakai dataset umum yang sering dipakai untuk belajar clustering:

1. **Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`)
   - Untuk K-Modes
   - Fitur numeric didiskritisasi menjadi `Low`, `Medium`, `High`
   - Tujuan: melihat mode cluster dan matching dissimilarity pada categorical representation

2. **Palmer Penguins** (`seaborn-data/penguins.csv`)
   - Untuk K-Prototypes
   - Numeric: bill length, bill depth, flipper length, body mass
   - Categorical: island, sex
   - Tujuan: melihat prototype berupa mean + mode dan efek gamma

3. **Iris** (`sklearn.datasets.load_iris`)
   - Untuk K-Medoids
   - Tujuan: membandingkan centroid K-Means vs medoid K-Medoids pada benchmark klasik

4. **Digits** (`sklearn.datasets.load_digits`)
   - Untuk Spectral Clustering sebagai kernel/similarity-based clustering praktis
   - Tujuan: membandingkan K-Means dan Spectral pada high-dimensional image features

5. **Noisy circles** (`sklearn.datasets.make_circles`)
   - Benchmark klasik untuk non-linear clustering
   - Tujuan: menunjukkan K-Means gagal pada struktur lingkaran konsentris

## Instalasi

```bash
pip install -r requirements.txt
```

Demo praktikum menggunakan library siap pakai: `kmodes` untuk K-Modes/K-Prototypes, `kmedoids` untuk K-Medoids, dan `scikit-learn` untuk Spectral Clustering. Hindari implementasi from scratch kecuali untuk penjelasan konsep di kelas.

## Cara Menjalankan

Jalankan dari root repository:

```bash
python3 week-10/praktikum/k_modes_k_prototypes.py
python3 week-10/praktikum/k_medoids_kernel_kmeans.py
```

Output figure akan tersimpan di:

```text
week-10/figures/
```

## File Demo

| File | Isi |
|---|---|
| `k_modes_k_prototypes.py` | K-Modes untuk categorical data dan K-Prototypes untuk mixed data |
| `k_medoids_kernel_kmeans.py` | K-Medoids vs K-Means pada Iris, Spectral pada Digits, dan Kernel/Spectral untuk circles |

## Figure yang Dihasilkan

| Figure | Deskripsi |
|---|---|
| `01_kmodes_breast_cancer_pca.png` | K-Modes pada Breast Cancer setelah diskritisasi, divisualisasikan dengan PCA |
| `02_kprototypes_penguins.png` | K-Prototypes pada Palmer Penguins |
| `03_kmedoids_iris_comparison.png` | K-Means vs K-Medoids pada Iris |
| `04_spectral_digits_comparison.png` | K-Means vs SpectralClustering pada Digits |
| `05_kernel_kmeans_circles.png` | K-Means vs Kernel K-Means mini vs Spectral RBF pada noisy circles |
| `06_rbf_kernel_similarity_matrix.png` | Heatmap similarity matrix dari RBF kernel |

## Catatan Mengajar

- Untuk K-Modes, fokuskan diskusi pada **mode** dan **matching dissimilarity**.
- Untuk K-Prototypes, fokuskan diskusi pada **gamma** sebagai bobot categorical mismatch.
- Untuk K-Medoids, fokuskan diskusi pada **medoid sebagai data asli** dan robustness terhadap outlier.
- Untuk Kernel K-Means, fokuskan diskusi pada **kernel trick** dan kenapa non-linear cluster butuh similarity space. Untuk hasil praktikum yang stabil, gunakan `SpectralClustering(affinity="rbf")` sebagai kernel-based clustering praktis.

Untuk production use, gunakan library yang lebih matang:

- `kmodes` untuk K-Modes dan K-Prototypes
- `kmedoids` untuk K-Medoids
- `SpectralClustering` atau implementasi khusus untuk kernel-based clustering
