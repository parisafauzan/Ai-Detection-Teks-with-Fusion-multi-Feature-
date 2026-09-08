"""Mesin inferensi: ekstraksi fitur (IndoBERT + GPT-2 perplexity + stilometri + TF-IDF)
lalu klasifikasi dengan pipeline CatBoost 469-dim (.joblib).

Alur ini MENIRU persis notebook:
- IndoBERT / perplexity / stilometri dihitung dari teks MENTAH.
- Kolom 'text' diisi hasil preprocess_for_tfidf(teks mentah) untuk cabang TF-IDF.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from features import extract_stylometric_features, preprocess_for_tfidf, STY_LABELS

ARTIFACT = Path(__file__).parent / "artifacts" / "model_detektor_catboost_469.joblib"
PPL_MODEL = "cahya/gpt2-small-indonesian-522M"
BERT_MODEL = "indobenchmark/indobert-base-p1"
MIN_ARTICLE_WORDS = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

_state = {"pipe": None, "ppl_tok": None, "ppl_model": None, "bert_tok": None, "bert_model": None}


def load():
    """Muat pipeline + kedua model transformer sekali saja (lazy singleton)."""
    if _state["pipe"] is not None:
        return
    if not ARTIFACT.exists():
        raise FileNotFoundError(
            f"Artefak model tidak ditemukan: {ARTIFACT}\n"
            "Jalankan notebook_export/export_model.py di lingkungan training, "
            "lalu salin file .joblib ke backend/artifacts/."
        )
    print(f"[load] device = {device}")
    print("[load] memuat pipeline CatBoost ...")
    _state["pipe"] = joblib.load(ARTIFACT)
    print("[load] memuat GPT-2 Indonesia (perplexity) ...")
    _state["ppl_tok"] = AutoTokenizer.from_pretrained(PPL_MODEL)
    _state["ppl_model"] = AutoModelForCausalLM.from_pretrained(PPL_MODEL).to(device).eval()
    print("[load] memuat IndoBERT ...")
    _state["bert_tok"] = AutoTokenizer.from_pretrained(BERT_MODEL)
    _state["bert_model"] = AutoModel.from_pretrained(BERT_MODEL).to(device).eval()
    print("[load] siap.")


def calculate_perplexity(text, window=512):
    tok, model = _state["ppl_tok"], _state["ppl_model"]
    try:
        enc = tok(text, return_tensors="pt", truncation=False)
        ids_full = enc.input_ids[0]
        if ids_full.shape[0] < 2:
            return float("nan")
        nll_sum, n_tok = 0.0, 0
        for start in range(0, ids_full.shape[0], window):
            chunk = ids_full[start:start + window]
            if chunk.shape[0] < 2:
                break
            ids = chunk.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(ids, labels=ids)
            ntok = ids.shape[1] - 1
            nll_sum += float(out.loss.item()) * ntok
            n_tok += ntok
        if n_tok == 0:
            return float("nan")
        return float(np.exp(nll_sum / n_tok))
    except Exception:
        return float("nan")


def get_bert_embeddings(text, max_length=512):
    tok, model = _state["bert_tok"], _state["bert_model"]
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy()[0]


def _build_frame(text):
    bert = get_bert_embeddings(text)              # (768,)
    sty_dict = extract_stylometric_features(text)  # 18 fitur (dict berurutan)
    ppl = calculate_perplexity(text)
    cleaned = preprocess_for_tfidf(text)

    row = {}
    for i, v in enumerate(bert):
        row[f"bert_{i}"] = np.float32(v)
    for i, v in enumerate(sty_dict.values()):
        row[f"sty_{i}"] = np.float32(v)
    row["perplexity"] = np.float32(ppl)
    row["text"] = cleaned
    return pd.DataFrame([row]), sty_dict, ppl


def predict(text):
    load()
    t0 = time.time()
    n_words = len(text.split())
    X, sty_dict, ppl = _build_frame(text)
    pipe = _state["pipe"]
    prob_ai = float(pipe.predict_proba(X)[0, 1])
    label_int = int(prob_ai >= 0.5)

    highlight_keys = [
        "perplexity", "unique_word_ratio", "avg_sentence_length",
        "function_word_ratio", "punctuation_ratio", "avg_word_length",
    ]
    stylometry = {k: float(v) for k, v in sty_dict.items()}

    result = {
        "label": "AI" if label_int == 1 else "Manusia",
        "label_int": label_int,
        "prob_ai": prob_ai,
        "prob_human": 1.0 - prob_ai,
        "confidence": max(prob_ai, 1.0 - prob_ai),
        "n_words": n_words,
        "too_short": n_words < MIN_ARTICLE_WORDS,
        "features": {
            "perplexity": None if ppl != ppl else round(ppl, 2),
            "stylometry": stylometry,
            "labels": STY_LABELS,
            "highlight_keys": highlight_keys,
        },
        "timing_ms": int((time.time() - t0) * 1000),
    }

    # --- Explainability tambahan (defensif: kalau gagal, app tetap jalan) ---
    result["feature_groups"] = compute_group_contributions(pipe, X)
    result["tfidf_terms"] = compute_tfidf_terms(pipe, X["text"].iloc[0])
    result["bert_info"] = {
        "dims_original": 768,
        "dims_after_pca": 150,
        "note": "Embedding semantik kontekstual (768->150 PCA); tiap dimensi tidak interpretatif sendiri, tetapi berkontribusi pada keputusan.",
    }
    return result


# ====================================================================
# Explainability: kontribusi 4 kelompok fitur + kata kunci TF-IDF
# ====================================================================
_GROUP_LABELS = {
    "tfidf": "TF-IDF", "svd": "TF-IDF", "text": "TF-IDF",
    "bert": "IndoBERT", "pca": "IndoBERT",
    "sty": "Stilometri", "perp": "Perplexity",
}


def _classify_group(name: str):
    key = name.split("__")[0].lower()
    for k, label in _GROUP_LABELS.items():
        if k in key:
            return label
    return None


def _find_instance(est, cls):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    if isinstance(est, cls):
        return est
    if isinstance(est, Pipeline):
        for _, step in est.steps:
            f = _find_instance(step, cls)
            if f is not None:
                return f
    if isinstance(est, ColumnTransformer):
        for _, trans, _ in est.transformers_:
            f = _find_instance(trans, cls)
            if f is not None:
                return f
    return None


def compute_group_contributions(pipe, X):
    """Kontribusi tiap KELOMPOK fitur untuk satu prediksi (SHAP CatBoost)."""
    try:
        from catboost import Pool
        pre, clf = pipe[:-1], pipe[-1]
        Xt = pre.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(Xt.shape[1])]

        shap = clf.get_feature_importance(Pool(Xt), type="ShapValues")[0]
        contribs = np.asarray(shap[:-1], dtype=float)  # elemen terakhir = bias

        groups = [_classify_group(n) for n in names]
        # fallback: urutan tetap tfidf(300)->bert(150)->sty(18)->perp(1)
        if any(g is None for g in groups) and Xt.shape[1] == 469:
            groups = (["TF-IDF"] * 300 + ["IndoBERT"] * 150
                      + ["Stilometri"] * 18 + ["Perplexity"] * 1)

        agg = {}
        for g, c in zip(groups, contribs):
            agg[g or "Lainnya"] = agg.get(g or "Lainnya", 0.0) + float(c)
        total = sum(abs(v) for v in agg.values()) or 1.0

        out = []
        for g in ["Stilometri", "Perplexity", "TF-IDF", "IndoBERT"]:
            if g in agg:
                out.append({
                    "group": g,
                    "contribution": agg[g],            # + -> AI, - -> Manusia
                    "share": abs(agg[g]) / total,      # proporsi |kontribusi|
                    "direction": "AI" if agg[g] >= 0 else "Manusia",
                })
        return out
    except Exception:
        return None


def compute_tfidf_terms(pipe, clean_text, top_k=8):
    """Kata/frasa dengan bobot TF-IDF tertinggi pada teks ini."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = _find_instance(pipe, TfidfVectorizer)
        if vec is None:
            return None
        arr = vec.transform([clean_text]).toarray()[0]
        vocab = np.array(vec.get_feature_names_out())
        idx = np.argsort(arr)[::-1][:top_k]
        return [{"term": str(vocab[i]), "weight": float(arr[i])}
                for i in idx if arr[i] > 0]
    except Exception:
        return None
