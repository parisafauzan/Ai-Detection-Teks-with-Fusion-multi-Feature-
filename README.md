# AI Text Detection

Repository ini berisi kode, data terpilih, model, dan hasil eksperimen untuk **penelitian deteksi teks berbahasa Indonesia yang dibuat manusia atau AI**.

Repository ini merupakan arsip dan artefak penelitian, **bukan tugas akhir atau dokumen tugas akhir**. Notebook, dataset, model, dan hasil di dalamnya disediakan agar alur eksperimen dapat diperiksa dan direproduksi secara terbatas.

## Ruang lingkup penelitian

Model memadukan empat kelompok fitur:

- stilometri, 18 fitur;
- perplexity dari GPT-2 bahasa Indonesia;
- TF-IDF dengan reduksi TruncatedSVD;
- embedding IndoBERT dengan reduksi PCA.

Vektor akhir berukuran 469 fitur dan diklasifikasikan menggunakan CatBoost. Evaluasi internal menggunakan `GroupKFold` berdasarkan `source_id` agar pasangan dokumen manusia dan AI dari sumber yang sama tidak terpisah ke fold berbeda.

## Struktur repository

```text
AI-Text-Detection/
├── notebooks/       # notebook utama penelitian
├── data/            # dataset internal dan eksternal terpilih
├── models/          # model hasil pelatihan
├── results/         # checkpoint, metrik, prediksi, dan analisis kesalahan
├── figures/         # visualisasi hasil eksperimen
├── scripts/         # script pendukung, termasuk ekspor model
└── apps/             # aplikasi web demonstrasi penelitian
```

## Notebook utama

Notebook utama adalah:

```text
notebooks/AI_Detection_revisi_bab4_terurut_github.ipynb
```

Alur notebook mencakup pemuatan dataset, ekstraksi fitur, reduksi dimensi, fusi fitur, validasi `GroupKFold`, studi ablasi 15 kombinasi fitur x 3 model, evaluasi, uji eksternal, dan ekspor model.

Jalankan sel notebook secara berurutan. Ekstraksi IndoBERT dan perplexity membutuhkan waktu serta koneksi internet saat model Hugging Face belum tersedia di komputer lokal.

## Dataset

- `data/dataset_berita_clean.csv`: dataset internal utama yang dibaca notebook.
- `data/berita_eksternal_clean_long.csv`: dataset untuk evaluasi eksternal.
- `data/id-newspaper_chatGPT (1).jsonl`: data pendukung/provenance, bila diperlukan untuk penelusuran dataset.

Pastikan lisensi, provenance, dan izin redistribusi dataset sudah diperiksa sebelum repository dipublikasikan.

## Aplikasi web

Aplikasi demonstrasi berada di:

```text
apps/detektor-ai-teks/
```

Aplikasi terdiri dari backend FastAPI dan frontend React/Vite. Petunjuk instalasi dan menjalankannya tersedia di `apps/detektor-ai-teks/README.md`.

Backend membutuhkan model di:

```text
apps/detektor-ai-teks/backend/artifacts/model_detektor_catboost_469.joblib
```

Model Transformer akan diunduh dari Hugging Face saat pertama kali backend melakukan warmup. Cache model tidak disimpan di repository.

## Reproduksibilitas

Versi library model yang perlu dijaga kompatibilitasnya antara lain:

- `scikit-learn==1.9.0`
- `catboost==1.2.10`

Pipeline backend harus tetap sama dengan definisi fitur pada notebook. Perubahan urutan fitur, rumus fitur, dimensi reduksi, atau hyperparameter memerlukan pelatihan ulang model.

## Catatan dataset dan privasi

Model ini ditujukan untuk teks berita berbahasa Indonesia dengan panjang minimal sekitar 20 kata. Hasil prediksi bukan bukti mutlak bahwa sebuah teks dibuat manusia atau AI. Jangan gunakan hasilnya sebagai satu-satunya dasar keputusan penting.

## Lisensi

Lisensi repository dan status redistribusi setiap dataset perlu ditetapkan setelah provenance masing-masing sumber diperiksa. Model dan data tidak boleh dianggap bebas digunakan hanya karena tersedia di repository ini.
