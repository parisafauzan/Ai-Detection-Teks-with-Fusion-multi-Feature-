"""API FastAPI untuk Detektor Teks AI Indonesia.

Jalankan:  uvicorn main:app --reload --port 8000

Endpoint:
  GET  /health          -> status & apakah model sudah dimuat
  POST /warmup          -> muat model lebih awal
  POST /predict         -> hasil lengkap sekaligus (kompatibilitas lama)
  POST /predict/stream  -> hasil BERTAHAP (NDJSON) tiap tahap begitu selesai
"""
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import inference

app = FastAPI(title="Detektor Teks AI Indonesia", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": inference._state["pipe"] is not None}


@app.post("/warmup")
def warmup():
    """Muat model lebih awal (unduh dari Hugging Face saat pertama kali)."""
    inference.load()
    return {"status": "ready"}


@app.post("/predict")
def predict(inp: TextIn):
    text = (inp.text or "").strip()
    if not text:
        return {"error": "Teks kosong. Silakan tempel teks berita untuk dideteksi."}
    return inference.predict(text)


@app.post("/predict/stream")
def predict_stream(inp: TextIn):
    """Streaming NDJSON: mengirim keluaran tiap tahap (UC-01..UC-05) begitu
    tahap tersebut selesai. Tiap baris respons adalah satu objek JSON:
        {"stage": "preprocessing", "status": "done", "data": {...}}\n
    diakhiri baris ringkasan {"stage": "summary", ...}.

    Generator sinkron ini otomatis dijalankan Starlette di threadpool,
    sehingga komputasi berat (IndoBERT/perplexity) tidak memblokir server.
    """
    text = (inp.text or "").strip()

    def gen():
        if not text:
            yield json.dumps(
                {
                    "stage": "error",
                    "status": "error",
                    "data": {"error": "Teks kosong. Silakan tempel teks berita untuk dideteksi."},
                },
                ensure_ascii=False,
            ) + "\n"
            return
        try:
            for event in inference.predict_stream(text):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps(
                {"stage": "error", "status": "error", "data": {"error": str(e)}},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
