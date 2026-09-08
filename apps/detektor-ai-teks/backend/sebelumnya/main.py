"""API FastAPI untuk Detektor Teks AI Indonesia.

Jalankan:  uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import inference

app = FastAPI(title="Detektor Teks AI Indonesia", version="1.0.0")

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
