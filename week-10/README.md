# Week 10: Extensions of K-Means

Materi pembelajaran tentang perluasan K-Means untuk data kategorikal, mixed data, outlier, dan cluster non-linear.

## Struktur Folder

```
week-10/
├── README.md                         # File ini
├── K-Means-Extensions.md             # Materi utama
├── praktikum/                        # Demo Python
│   ├── README.md                      # Panduan praktikum
│   ├── requirements.txt               # Dependencies
│   ├── k_modes_k_prototypes.py        # Demo categorical & mixed data
│   └── k_medoids_kernel_kmeans.py     # Demo Iris, Digits, dan circles
└── figures/                          # Output visualisasi
```

## Materi

### Extensions of K-Means

- **Dokumentasi:** `K-Means-Extensions.md`
- **Topik:**
  - Recap keterbatasan K-Means
  - K-Modes untuk data kategorikal
  - K-Prototypes untuk mixed numeric-categorical data
  - K-Medoids untuk robust clustering dan custom distance
  - Kernel K-Means untuk non-linear cluster structure
  - Decision guide memilih metode clustering

## Fokus Pembelajaran

Setelah Week 8 membahas K-Means dasar, Week 10 menjawab pertanyaan:

> Kalau data kita bukan numeric-spherical-clean, extension K-Means mana yang harus dipakai?

| Problem | Metode |
|---|---|
| Data kategorikal | K-Modes |
| Data numeric + categorical | K-Prototypes |
| Banyak outlier / butuh representative object | K-Medoids |
| Cluster non-linear | Kernel K-Means |

## Praktikum yang Disarankan

Demo praktikum tersedia dalam dua script:

1. `praktikum/k_modes_k_prototypes.py`
   - K-Modes pada Breast Cancer Wisconsin yang didiskritisasi
   - K-Prototypes pada Palmer Penguins mixed data
   - Gamma sensitivity analysis

2. `praktikum/k_medoids_kernel_kmeans.py`
   - K-Means vs K-Medoids pada Iris
   - K-Means vs SpectralClustering pada Digits
   - K-Means vs Kernel K-Means mini vs SpectralClustering RBF pada noisy circles

Jalankan dari root repository:

```bash
python3 week-10/praktikum/k_modes_k_prototypes.py
python3 week-10/praktikum/k_medoids_kernel_kmeans.py
```

Figure akan tersimpan di `week-10/figures/`.

## Library

```bash
pip install numpy pandas matplotlib scikit-learn
```

Script praktikum dibuat self-contained untuk kebutuhan belajar. Untuk production use, lihat package `kmodes` dan `scikit-learn-extra`.

## Referensi Utama

- Huang, Z. (1998). *Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values*.
- Huang, Z. (1997). *Clustering large data sets with mixed numeric and categorical values*.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*.
- Dhillon, I. S., Guan, Y. & Kulis, B. (2004). *Kernel k-means, Spectral Clustering and Normalized Cuts*.
- `kmodes` Python package: https://github.com/nicodv/kmodes
- `scikit-learn-extra` KMedoids: https://scikit-learn-extra.readthedocs.io/
