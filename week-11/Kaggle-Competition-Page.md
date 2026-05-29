# Kaggle Competition Page

## Title

Data Mining Sains Data ITS - SVC [Week 11]

## Subtitle

Prediksi Deposito Nasabah Menggunakan Support Vector Classification

## Overview

Dalam industri perbankan, memahami perilaku nasabah sangat penting untuk meningkatkan efektivitas kampanye pemasaran. Salah satu tantangan utama adalah mengidentifikasi nasabah yang berpotensi merespons positif terhadap penawaran produk keuangan, seperti deposito berjangka.

Pada kompetisi ini, peserta diminta membangun model klasifikasi untuk memprediksi apakah seorang nasabah akan berlangganan deposito berjangka berdasarkan data demografis, informasi kontak, riwayat kampanye, dan indikator ekonomi.

Kompetisi ini merupakan bagian dari Week 11 Praktikum Data Mining dengan fokus pada **Support Vector Classification (SVC)**. Peserta diharapkan menggunakan SVC sebagai model utama, lalu menentukan sendiri strategi preprocessing, splitting, validation, dan tuning yang paling sesuai.

**Goal:** prediksi probabilitas bahwa seorang nasabah akan berlangganan deposito berjangka (`berlangganan_deposito = 1`).

## Description

### Latar Belakang

Kampanye pemasaran bank sering kali membutuhkan biaya dan waktu yang besar. Jika bank dapat memprediksi nasabah mana yang lebih berpotensi membeli produk deposito, strategi pemasaran dapat menjadi lebih efisien dan tepat sasaran.

Dataset pada kompetisi ini berisi informasi nasabah, riwayat kontak pemasaran, hasil kampanye sebelumnya, serta beberapa indikator ekonomi. Berdasarkan informasi tersebut, peserta diminta membuat model yang mampu memprediksi peluang seorang nasabah untuk berlangganan deposito berjangka.

### Tujuan

Membuat model machine learning yang dapat memprediksi probabilitas nasabah masuk ke kelas berikut:

- `1`: nasabah berpotensi berlangganan deposito berjangka
- `0`: nasabah tidak berlangganan deposito berjangka

Karena evaluation metric yang digunakan adalah **AUC**, peserta sebaiknya mengirimkan nilai probabilitas untuk kelas `1`, bukan hanya label 0/1.

### Problem Statement

Diberikan data training dengan label `berlangganan_deposito`, bangun model klasifikasi yang dapat memprediksi probabilitas `berlangganan_deposito = 1` pada data test.

Peserta perlu memperhatikan beberapa hal berikut:

- Data memiliki kombinasi fitur numerik dan kategorikal.
- Model SVM/SVC sensitif terhadap skala fitur numerik, sehingga scaling sangat disarankan.
- Fitur kategorikal perlu diubah menjadi representasi numerik, misalnya menggunakan one-hot encoding.
- Hyperparameter seperti `C`, `kernel`, dan `gamma` dapat memengaruhi performa model secara signifikan.

### Dataset

Dataset terdiri dari tiga file:

| File | Deskripsi |
|---|---|
| `train.csv` | Data training yang berisi fitur dan target `berlangganan_deposito` |
| `test.csv` | Data test yang hanya berisi fitur tanpa target |
| `sample_submission.csv` | Contoh format file submission |

### Kolom Target

| Kolom | Deskripsi |
|---|---|
| `berlangganan_deposito` | Apakah nasabah berlangganan deposito berjangka. Nilai `1` berarti berlangganan, dan `0` berarti tidak berlangganan. |

### Deskripsi Kolom

| Nama Kolom | Deskripsi Kolom |
|---|---|
| `customer_number` | Angka unik yang diberikan kepada setiap nasabah untuk membedakan satu nasabah dengan yang lainnya |
| `usia` | Usia nasabah dalam tahun |
| `pekerjaan` | Jenis pekerjaan nasabah |
| `status_perkawinan` | Status perkawinan nasabah, termasuk duda/janda sebagai `cerai` |
| `pendidikan` | Tingkat pendidikan terakhir yang diselesaikan oleh nasabah |
| `gagal_bayar_sebelumnya` | Apakah nasabah pernah mengalami gagal bayar kredit sebelumnya |
| `pinjaman_rumah` | Apakah nasabah memiliki pinjaman rumah |
| `pinjaman_pribadi` | Apakah nasabah memiliki pinjaman pribadi |
| `jenis_kontak` | Jenis media komunikasi yang digunakan untuk menghubungi nasabah |
| `bulan_kontak_terakhir` | Bulan ketika nasabah terakhir kali dihubungi |
| `hari_kontak_terakhir` | Hari dalam seminggu ketika nasabah terakhir kali dihubungi |
| `jumlah_kontak_kampanye_ini` | Jumlah total kontak yang dilakukan selama kampanye saat ini |
| `hari_sejak_kontak_sebelumnya` | Jumlah hari sejak nasabah terakhir dihubungi dari kampanye sebelumnya. Nilai `999` berarti belum pernah dihubungi sebelumnya |
| `jumlah_kontak_sebelumnya` | Jumlah kontak yang dilakukan sebelum kampanye saat ini terhadap nasabah |
| `hasil_kampanye_sebelumnya` | Hasil dari kampanye pemasaran sebelumnya terhadap nasabah |
| `tingkat_variasi_pekerjaan` | Indikator variasi tingkat pekerjaan secara kuartalan |
| `indeks_harga_konsumen` | Indeks harga konsumen pada bulan terkait |
| `indeks_kepercayaan_konsumen` | Indeks tingkat kepercayaan konsumen pada bulan terkait |
| `suku_bunga_euribor_3bln` | Suku bunga Euribor untuk tenor 3 bulan |
| `jumlah_pekerja` | Jumlah total tenaga kerja dalam indikator kuartalan |
| `pulau` | Wilayah pulau nasabah |

### Catatan Praktikum

Pada kompetisi ini, peserta diminta menggunakan **Support Vector Classification (SVC)** sebagai model utama.

Strategi preprocessing, pemilihan fitur, splitting dataset, validation, dan hyperparameter tuning dibebaskan kepada peserta. Namun, karena data memiliki fitur numerik dan kategorikal, peserta perlu memastikan data sudah diproses dalam bentuk yang dapat digunakan oleh SVC.

Output akhir yang dikumpulkan adalah **probabilitas kelas `1`** untuk setiap baris pada `test.csv`, lalu disimpan mengikuti format `sample_submission.csv`.

## Evaluation

Submissions are evaluated using **Area Under the ROC Curve (AUC)** between the predicted probability and the observed target.

AUC mengukur kemampuan model membedakan nasabah yang berlangganan deposito (`1`) dan tidak berlangganan deposito (`0`) pada berbagai threshold klasifikasi. Nilai AUC yang lebih tinggi menunjukkan model lebih baik dalam memberikan ranking probabilitas terhadap kelas positif.

Secara matematis, ROC-AUC dapat dipahami sebagai luas area di bawah kurva ROC:

$$
\mathrm{AUC} = \int_0^1 \mathrm{TPR}(\mathrm{FPR})\, d\mathrm{FPR}
$$

dengan:

$$
\mathrm{TPR} = \frac{TP}{TP + FN}
$$

$$
\mathrm{FPR} = \frac{FP}{FP + TN}
$$

Interpretasi lain: AUC adalah peluang bahwa model memberi skor probabilitas lebih tinggi pada satu sampel positif dibandingkan satu sampel negatif yang dipilih secara acak.

Karena metric yang digunakan adalah AUC, peserta harus mengirimkan **probabilitas untuk kelas `1`**, bukan hanya label hard prediction 0/1.

### Submission File

Untuk setiap `customer_number` pada test set, peserta harus memprediksi probabilitas nasabah tersebut akan berlangganan deposito berjangka.

File submission harus memiliki header dan format berikut:

```csv
customer_number,berlangganan_deposito
445420,0.12
585604,0.78
888824,0.34
816820,0.05
```

Kolom yang wajib ada:

| Kolom | Deskripsi |
|---|---|
| `customer_number` | ID nasabah dari `test.csv` |
| `berlangganan_deposito` | Probabilitas nasabah berlangganan deposito berjangka (`berlangganan_deposito = 1`) |

Nilai pada kolom `berlangganan_deposito` sebaiknya berada pada rentang 0 sampai 1.
