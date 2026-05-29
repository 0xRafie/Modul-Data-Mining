# Data Description

Kompetisi ini menggunakan data kampanye pemasaran bank untuk memprediksi apakah seorang nasabah akan berlangganan deposito berjangka.

Peserta diberikan data training yang sudah memiliki label target dan data test tanpa label. Tugas peserta adalah membuat prediksi probabilitas untuk setiap nasabah pada `test.csv`.

## Files

| File | Deskripsi |
|---|---|
| `train.csv` | Data training. Berisi fitur nasabah dan target `berlangganan_deposito`. |
| `test.csv` | Data test. Berisi fitur nasabah tanpa target. Peserta harus membuat prediksi untuk file ini. |
| `sample_submission.csv` | Contoh format submission yang benar. |

Ukuran data:

| File | Jumlah Baris | Jumlah Kolom |
|---|---:|---:|
| `train.csv` | 22,916 | 22 |
| `test.csv` | 5,729 | 21 |
| `sample_submission.csv` | 5,729 | 2 |

## What You Are Predicting

Target yang diprediksi adalah:

```text
berlangganan_deposito
```

Target ini menunjukkan apakah nasabah berlangganan deposito berjangka.

| Nilai | Makna |
|---|---|
| `0` | Nasabah tidak berlangganan deposito |
| `1` | Nasabah berlangganan deposito |

Karena evaluation metric menggunakan **ROC-AUC**, peserta harus mengirimkan **probabilitas untuk kelas `1`**, bukan hanya label 0/1.

## Submission Format

File submission harus memiliki dua kolom:

| Kolom | Deskripsi |
|---|---|
| `customer_number` | ID unik nasabah dari `test.csv` |
| `berlangganan_deposito` | Probabilitas bahwa nasabah berlangganan deposito berjangka (`1`) |

Contoh format:

```csv
customer_number,berlangganan_deposito
445420,0.12
585604,0.78
888824,0.34
816820,0.05
```

Nilai pada kolom `berlangganan_deposito` sebaiknya berada pada rentang 0 sampai 1.

## Columns

### Identifier

| Nama Kolom | Deskripsi |
|---|---|
| `customer_number` | Angka unik yang diberikan kepada setiap nasabah untuk membedakan satu nasabah dengan yang lainnya. Kolom ini digunakan sebagai ID submission, bukan sebagai target. |

### Demographic Features

| Nama Kolom | Deskripsi |
|---|---|
| `usia` | Usia nasabah dalam tahun. |
| `pekerjaan` | Jenis pekerjaan nasabah. |
| `status_perkawinan` | Status perkawinan nasabah. Nilai `cerai` mencakup cerai, duda, atau janda. |
| `pendidikan` | Tingkat pendidikan terakhir yang diselesaikan oleh nasabah. |
| `pulau` | Wilayah pulau tempat nasabah berada. |

### Credit and Loan Features

| Nama Kolom | Deskripsi |
|---|---|
| `gagal_bayar_sebelumnya` | Apakah nasabah pernah mengalami gagal bayar kredit sebelumnya. |
| `pinjaman_rumah` | Apakah nasabah memiliki pinjaman rumah. |
| `pinjaman_pribadi` | Apakah nasabah memiliki pinjaman pribadi. |

### Campaign Contact Features

| Nama Kolom | Deskripsi |
|---|---|
| `jenis_kontak` | Jenis media komunikasi yang digunakan untuk menghubungi nasabah, misalnya `cellular` atau `telephone`. |
| `bulan_kontak_terakhir` | Bulan ketika nasabah terakhir kali dihubungi. |
| `hari_kontak_terakhir` | Hari dalam seminggu ketika nasabah terakhir kali dihubungi. |
| `jumlah_kontak_kampanye_ini` | Jumlah total kontak yang dilakukan terhadap nasabah selama kampanye saat ini. |

### Previous Campaign Features

| Nama Kolom | Deskripsi |
|---|---|
| `hari_sejak_kontak_sebelumnya` | Jumlah hari sejak nasabah terakhir dihubungi dari kampanye sebelumnya. Nilai `999` berarti nasabah belum pernah dihubungi sebelumnya. |
| `jumlah_kontak_sebelumnya` | Jumlah kontak yang dilakukan sebelum kampanye saat ini terhadap nasabah. |
| `hasil_kampanye_sebelumnya` | Hasil dari kampanye pemasaran sebelumnya terhadap nasabah. |

### Economic Indicator Features

| Nama Kolom | Deskripsi |
|---|---|
| `tingkat_variasi_pekerjaan` | Indikator variasi tingkat pekerjaan secara kuartalan. |
| `indeks_harga_konsumen` | Indeks harga konsumen pada bulan terkait. |
| `indeks_kepercayaan_konsumen` | Indeks tingkat kepercayaan konsumen pada bulan terkait. |
| `suku_bunga_euribor_3bln` | Suku bunga Euribor untuk tenor 3 bulan. |
| `jumlah_pekerja` | Jumlah total tenaga kerja dalam indikator kuartalan. |

### Target Column

| Nama Kolom | Deskripsi |
|---|---|
| `berlangganan_deposito` | Target pada `train.csv`. Nilai `1` berarti nasabah berlangganan deposito berjangka, sedangkan `0` berarti tidak berlangganan. Kolom ini tidak tersedia pada `test.csv`. |

## Notes

- `train.csv` memiliki kolom target `berlangganan_deposito`.
- `test.csv` tidak memiliki kolom target.
- `sample_submission.csv` hanya menunjukkan format file submission.
- Peserta diminta menggunakan **Support Vector Classification (SVC)** sebagai model utama.
