# Week 11: Support Vector Machine

Materi pembelajaran tentang **Support Vector Machine (SVM)** untuk supervised learning, dengan fokus utama pada **Support Vector Classification (SVC)**.

## Struktur Folder

```
week-11/
├── README.md                  # File ini
└── Support-Vector-Machine.md  # Materi utama
```

## Materi

### Support Vector Machine

- **Dokumentasi:** `Support-Vector-Machine.md`
- **Topik:**
  - Basic concept SVM
  - Hyperplane, margin, dan support vectors
  - Hard margin vs soft margin
  - Intuisi convex optimization
  - Support Vector Classification (SVC)
  - Kernel trick dan kernel functions
  - Parameter penting SVC
  - Hyperparameter tuning: Grid Search, Random Search, Optuna

## Fokus Pembelajaran

Week 11 menjawab pertanyaan:

> Bagaimana SVM mencari decision boundary yang bukan hanya memisahkan class, tetapi juga punya margin paling aman?

| Konsep | Inti |
|---|---|
| Hyperplane | Garis/bidang pemisah antar class |
| Margin | Jarak aman dari hyperplane ke data terdekat |
| Support vectors | Data penting yang menentukan margin |
| Kernel | Cara membuat boundary non-linear tanpa eksplisit membuat fitur baru |
| `C` | Trade-off antara margin lebar dan training error |
| `gamma` | Jangkauan pengaruh satu data point pada kernel RBF |

## Library

```bash
pip install numpy pandas matplotlib scikit-learn optuna
```

Untuk materi inti, `optuna` hanya diperlukan jika ingin menjalankan contoh hyperparameter optimization berbasis trial.

## Dataset dan Kaggle Competition

Dataset latihan tersedia di Kaggle:

- **Nama:** Data Mining Sains Data ITS - SVC [Week 11]
- **URL:** https://www.kaggle.com/datasets/danzarchive/data-mining-sains-data-its-svc-week-11

File lokal berada di `week-11/data/`:

| File | Isi |
|---|---|
| `train.csv` | Data training dengan target `berlangganan_deposito` |
| `test.csv` | Data test tanpa target |
| `sample_submission.csv` | Format submission Kaggle |

Format submission:

```csv
customer_number,berlangganan_deposito
445420,0.12
585604,0.78
```

Metric kompetisi yang disarankan adalah **AUC**, sehingga kolom `berlangganan_deposito` pada submission berisi probabilitas untuk kelas `1`.

Peserta diminta menggunakan **SVC** sebagai model utama. Strategi preprocessing, split data, validasi, dan hyperparameter tuning dibebaskan kepada peserta. Prediksi akhir dilakukan pada `test.csv` dan dikumpulkan dalam bentuk probabilitas kelas `1`.

Target training adalah `berlangganan_deposito` dengan label:

- `0`: tidak berlangganan deposito
- `1`: berlangganan deposito

## Referensi Utama

- Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20, 273-297.
- Boser, B. E., Guyon, I. M. & Vapnik, V. N. (1992). *A Training Algorithm for Optimal Margin Classifiers*.
- Hsu, C.-W., Chang, C.-C. & Lin, C.-J. (2025). *A Practical Guide to Support Vector Classification*.
- Chang, C.-C. & Lin, C.-J. (2011). *LIBSVM: A Library for Support Vector Machines*.
- scikit-learn SVM User Guide: https://scikit-learn.org/stable/modules/svm.html
- scikit-learn SVC API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- scikit-learn hyperparameter tuning: https://scikit-learn.org/stable/modules/grid_search.html
- Optuna documentation: https://optuna.readthedocs.io/
