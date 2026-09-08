# Detektor Teks AI Indonesia (Web App Penelitian)

Aplikasi ini adalah demonstrator dari penelitian deteksi teks AI berbahasa Indonesia,
bukan dokumen tugas akhir dan bukan pengganti evaluasi ilmiah pada notebook utama.

Aplikasi web untuk mendeteksi apakah sebuah teks berita berbahasa Indonesia ditulis
**manusia** atau **AI**, memakai model dari penelitian Anda: fusi multi-fitur
(**stilometri + perplexity + TF-IDF + IndoBERT**, 469 dimensi) yang diklasifikasi
**CatBoost**.

```
tempel teks  ->  ekstraksi 4 fitur  ->  fusi 469-dim  ->  CatBoost  ->  AI / Manusia + keyakinan
```

## Arsitektur

- **backend/** - FastAPI (Python). Memuat model `.joblib` + IndoBERT + GPT-2 Indonesia,
  menjalankan ekstraksi fitur yang **identik** dengan notebook, lalu memprediksi.
- **frontend/** - React + Vite + Tailwind + **shadcn/ui**. Tampilan modern:
  kotak paste teks, verdict, gauge keyakinan, dan rincian fitur (perplexity, stilometri).

Karena mesin deteksinya butuh PyTorch + IndoBERT + GPT-2, aplikasi ini tidak bisa
"web murni" (HTML/JS saja) - harus ada backend Python. Frontend & backend berjalan
sebagai dua proses di laptop Anda.

---

## Langkah 0 - Buat artefak model (SEKALI, di lingkungan training)

Di notebook `AI_Detection.ipynb` (tempat `Xdf`, `y`, `build_preprocessor`, `make_models`
sudah ada), jalankan isi file **`notebook_export/export_model.py`**. Ini menghasilkan:

```
model_detektor_catboost_469.joblib
```

Salin file itu ke: **`backend/artifacts/model_detektor_catboost_469.joblib`**

> Catat versi `scikit-learn` & `catboost` yang dicetak, lalu samakan di
> `backend/requirements.txt` supaya pickle-nya kompatibel.

---

## Langkah 1 - Jalankan backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Saat pertama kali, model IndoBERT & GPT-2 Indonesia akan **diunduh dari Hugging Face**
(butuh internet, sekali saja). Untuk memanaskan lebih awal:
`curl -X POST http://localhost:8000/warmup`

## Langkah 2 - Jalankan frontend

```bash
cd frontend
npm install
npm run dev
```

Buka **http://localhost:5173**. Frontend otomatis meneruskan permintaan `/api/*`
ke backend di `:8000` (lihat `vite.config.ts`).

---

## Catatan

- **Kebutuhan**: RAM ~2-3 GB; tanpa GPU tetap jalan (CPU), latensi ~2-6 detik/teks.
- **Domain**: dilatih untuk **berita berbahasa Indonesia**; teks domain lain / bahasa lain
  kurang andal. Disarankan input >= 20 kata.
- Ekstraksi fitur di `backend/features.py` disalin **persis** dari notebook. Jika Anda
  mengubah definisi fitur di notebook, perbarui juga file ini lalu latih ulang model.
- Endpoint: `GET /health`, `POST /warmup`, `POST /predict` (body `{"text": "..."}`).

## Struktur

```
detektor-ai-teks/
  backend/
    main.py            # API FastAPI
    inference.py       # muat model + prediksi
    features.py        # ekstraksi fitur (salinan notebook)
    requirements.txt
    artifacts/         # <- taruh model_detektor_catboost_469.joblib di sini
  frontend/
    src/App.tsx        # UI utama
    src/components/ui/  # komponen shadcn (button, card, textarea, badge, progress)
    ...
  notebook_export/
    export_model.py    # sel untuk menghasilkan .joblib
  README.md
```
