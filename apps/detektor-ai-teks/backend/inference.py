"""Mesin inferensi: ekstraksi fitur (IndoBERT + GPT-2 perplexity + stilometri + TF-IDF)
lalu klasifikasi dengan pipeline CatBoost 469-dim (.joblib).

Alur ini MENIRU persis notebook:
- IndoBERT / perplexity / stilometri dihitung dari teks MENTAH.
- Kolom 'text' diisi hasil preprocess_for_tfidf(teks mentah) untuk cabang TF-IDF.

VERSI PENGUJIAN USE CASE:
predict() kini mengembalikan blok `stages` yang memuat keluaran NYATA tiap
tahap pipeline, sehingga tiap use case (UC-01..UC-05) dapat diuji langsung dari
aplikasi:
  UC-01 Melakukan Prapemrosesan Teks     -> stages.preprocessing
  UC-02 Mengekstraksi Fitur (4 fitur)     -> stages.extraction
  UC-03 Mereduksi dan Fusi Fitur          -> stages.reduction
  UC-04 Mengklasifikasikan Teks           -> stages.classification
  UC-05 Menampilkan Kontribusi Fitur      -> stages.contribution

Untuk TF-IDF & IndoBERT ditampilkan vektor SEBELUM dan SESUDAH reduksi
(keduanya keluaran UC-02 dan UC-03).
"""
import math
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

# Konfigurasi reduksi (sesuai notebook: svd_k=300, pca_k=150, fusi=469)
SVD_K = 300
PCA_K = 150
N_STY = 18
N_PERP = 1
FUSION_DIM = SVD_K + PCA_K + N_STY + N_PERP  # 469
BERT_DIM = 768
TFIDF_MAX_FEATURES = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

_state = {"pipe": None, "ppl_tok": None, "ppl_model": None, "bert_tok": None, "bert_model": None}

# Label ramah-tampilan untuk 18 fitur stilometri (urutan = sty_0..sty_17)
STY_LABELS_FULL = {
    "char_count": "Jumlah karakter",
    "word_count": "Jumlah kata",
    "sentence_count": "Jumlah kalimat",
    "avg_sentence_length": "Rata-rata panjang kalimat (kata/kalimat)",
    "avg_word_length": "Rata-rata panjang kata (karakter)",
    "unique_word_ratio": "Rasio kata unik (TTR)",
    "comma_count": "Jumlah koma",
    "semicolon_count": "Jumlah titik koma",
    "question_mark_count": "Jumlah tanda tanya",
    "exclamation_mark_count": "Jumlah tanda seru",
    "period_count": "Jumlah titik",
    "colon_count": "Jumlah titik dua",
    "punctuation_ratio": "Rasio tanda baca",
    "uppercase_ratio": "Rasio huruf kapital",
    "digit_ratio": "Rasio digit",
    "special_char_count": "Jumlah karakter khusus",
    "special_char_ratio": "Rasio karakter khusus",
    "function_word_ratio": "Rasio kata fungsi (FWR)",
}


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


def _ppl_token_count(text):
    """Jumlah token GPT-2 (untuk info tahap perplexity)."""
    try:
        enc = _state["ppl_tok"](text, return_tensors="pt", truncation=False)
        return int(enc.input_ids.shape[1])
    except Exception:
        return None


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
    return pd.DataFrame([row]), sty_dict, ppl, bert, cleaned


# ------------------------------------------------------------------ util
def _f(v):
    """float aman-JSON (NaN/inf -> None)."""
    try:
        v = float(v)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _round_vec(vec, nd=5):
    return [round(_f(x), nd) if _f(x) is not None else None for x in np.asarray(vec).ravel()]


def _vec_stats(vec):
    v = np.asarray(vec, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {
        "min": _f(np.min(v)),
        "max": _f(np.max(v)),
        "mean": _f(np.mean(v)),
        "std": _f(np.std(v)),
        "l2_norm": _f(np.linalg.norm(v)),
        "nonzero": int(np.count_nonzero(v)),
    }


def _sanitize(obj):
    """Bersihkan NaN/inf & tipe numpy agar aman diserialisasi JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _f(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return _f(obj)
    return obj


# ------------------------------------------------------------ akses cabang
def _column_transformer(pipe):
    try:
        return pipe.named_steps["pre"]
    except Exception:
        # fallback: cari ColumnTransformer
        from sklearn.compose import ColumnTransformer
        for _, step in getattr(pipe, "steps", []):
            if isinstance(step, ColumnTransformer):
                return step
    return None


def _named_branch(ct, name):
    try:
        return ct.named_transformers_[name]
    except Exception:
        return None


# ================================================================
#  Tahap-tahap pipeline (untuk pengujian tiap use case)
# ================================================================
def _stage_preprocessing(text, cleaned):
    raw_tokens = text.split()
    clean_tokens = cleaned.split()
    return {
        "raw_char_count": len(text),
        "raw_word_count": len(raw_tokens),
        "raw_preview": text[:800],
        "cleaned_text": cleaned,
        "cleaned_token_count": len(clean_tokens),
        "removed_token_count": max(len(raw_tokens) - len(clean_tokens), 0),
        "steps": [
            "Ubah semua huruf menjadi huruf kecil (lowercase)",
            "Buang karakter non-alfabet (angka, tanda baca, simbol)",
            "Buang stopword Indonesia (Sastrawi) & kata dengan panjang \u2264 2 huruf",
        ],
    }


def _stage_extraction(ct, X, sty_dict, ppl, bert, cleaned):
    # --- TF-IDF (sebelum reduksi) ---
    tfidf_out = {"max_features": TFIDF_MAX_FEATURES, "ngram_range": "(1, 2)", "min_df": 3}
    branch = _named_branch(ct, "tfidf")
    try:
        vec = branch.named_steps["tfidf"]
        sp = vec.transform([cleaned])
        vocab = np.array(vec.get_feature_names_out())
        arr = sp.toarray()[0]
        nz = np.nonzero(arr)[0]
        order = nz[np.argsort(arr[nz])[::-1]]
        terms = [{"term": str(vocab[i]), "weight": _f(arr[i])} for i in order]
        tfidf_out.update({
            "dim": int(len(vocab)),
            "nonzero": int(len(nz)),
            "terms": terms,
        })
    except Exception as e:
        tfidf_out.update({"dim": None, "nonzero": None, "terms": [], "error": str(e)})

    # --- IndoBERT (sebelum reduksi, 768) ---
    bert_out = {
        "dim": int(len(bert)),
        "vector": _round_vec(bert),
        "stats": _vec_stats(bert),
        "note": "Vektor token [CLS] dari IndoBERT (indobert-base-p1), 768 dimensi.",
    }

    # --- Stilometri (18) ---
    sty_items = []
    for i, (k, v) in enumerate(sty_dict.items()):
        sty_items.append({
            "index": i,
            "key": k,
            "label": STY_LABELS_FULL.get(k, k),
            "value": _f(v),
        })
    sty_out = {"dim": len(sty_items), "features": sty_items}

    # --- Perplexity (1) ---
    perp_out = {
        "dim": 1,
        "value": _f(ppl),
        "ppl_token_count": _ppl_token_count(X["text"].iloc[0]) if False else _ppl_token_count(cleaned),
        "model": PPL_MODEL,
    }

    return {"tfidf": tfidf_out, "indobert": bert_out, "stylometry": sty_out, "perplexity": perp_out}


def _explained_variance(branch, step_name):
    try:
        est = branch.named_steps[step_name]
        evr = getattr(est, "explained_variance_ratio_", None)
        if evr is not None:
            return _f(float(np.sum(evr)))
    except Exception:
        pass
    return None


def _stage_reduction(ct, X):
    fused = ct.transform(X)
    fused = fused.toarray()[0] if hasattr(fused, "toarray") else np.asarray(fused).ravel()

    # Urutan ColumnTransformer: tfidf(300) -> bert(150) -> sty(18) -> perp(1)
    tfidf_svd = fused[0:SVD_K]
    bert_pca = fused[SVD_K:SVD_K + PCA_K]
    sty_scaled = fused[SVD_K + PCA_K:SVD_K + PCA_K + N_STY]
    perp_scaled = fused[SVD_K + PCA_K + N_STY:FUSION_DIM]

    tfidf_branch = _named_branch(ct, "tfidf")
    bert_branch = _named_branch(ct, "bert")

    return {
        "tfidf_svd": {
            "dim_before": None,  # diisi oleh caller (vocab aktif)
            "dim_after": SVD_K,
            "vector": _round_vec(tfidf_svd),
            "stats": _vec_stats(tfidf_svd),
            "explained_variance_ratio_sum": _explained_variance(tfidf_branch, "svd"),
            "method": "TruncatedSVD",
        },
        "bert_pca": {
            "dim_before": BERT_DIM,
            "dim_after": PCA_K,
            "vector": _round_vec(bert_pca),
            "stats": _vec_stats(bert_pca),
            "explained_variance_ratio_sum": _explained_variance(bert_branch, "pca"),
            "method": "PCA (StandardScaler + PCA)",
        },
        "stylometry": {
            "dim": N_STY,
            "vector": _round_vec(sty_scaled),
            "reduced": False,
            "note": "Distandarisasi (SimpleImputer + StandardScaler); dimensi tetap 18.",
        },
        "perplexity": {
            "dim": N_PERP,
            "vector": _round_vec(perp_scaled),
            "reduced": False,
            "note": "Distandarisasi; dimensi tetap 1.",
        },
        "fusion": {
            "dim": int(len(fused)),
            "composition": {
                "TF-IDF (SVD)": SVD_K,
                "IndoBERT (PCA)": PCA_K,
                "Stilometri": N_STY,
                "Perplexity": N_PERP,
            },
            "vector": _round_vec(fused),
            "stats": _vec_stats(fused),
        },
    }


def predict(text):
    load()
    t0 = time.time()
    n_words = len(text.split())
    X, sty_dict, ppl, bert, cleaned = _build_frame(text)
    pipe = _state["pipe"]
    ct = _column_transformer(pipe)

    prob_ai = float(pipe.predict_proba(X)[0, 1])
    label_int = int(prob_ai >= 0.5)

    highlight_keys = [
        "perplexity", "unique_word_ratio", "avg_sentence_length",
        "function_word_ratio", "punctuation_ratio", "avg_word_length",
    ]
    stylometry = {k: float(v) for k, v in sty_dict.items()}

    # ---------------- keluaran tiap use case ----------------
    stage_pre = _stage_preprocessing(text, cleaned)
    stage_ext = _stage_extraction(ct, X, sty_dict, ppl, bert, cleaned)
    stage_red = _stage_reduction(ct, X)
    # sinkronkan dim_before TF-IDF (vocab aktif) dari tahap ekstraksi
    stage_red["tfidf_svd"]["dim_before"] = stage_ext["tfidf"].get("dim")

    stage_cls = {
        "label": "AI" if label_int == 1 else "Manusia",
        "label_int": label_int,
        "prob_ai": _f(prob_ai),
        "prob_human": _f(1.0 - prob_ai),
        "threshold": 0.5,
        "rule": "prob_ai \u2265 0,5 \u2192 AI; selain itu Manusia",
    }
    stage_contrib = compute_group_contributions(pipe, X)

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
        # ---- blok baru: keluaran tiap tahap (untuk pengujian use case) ----
        "stages": {
            "preprocessing": stage_pre,   # UC-01
            "extraction": stage_ext,      # UC-02
            "reduction": stage_red,       # UC-03
            "classification": stage_cls,  # UC-04
            "contribution": stage_contrib,  # UC-05
        },
        "timing_ms": int((time.time() - t0) * 1000),
    }

    # --- Explainability tambahan (kompatibilitas versi lama) ---
    result["feature_groups"] = stage_contrib
    result["tfidf_terms"] = compute_tfidf_terms(pipe, X["text"].iloc[0])
    result["bert_info"] = {
        "dims_original": BERT_DIM,
        "dims_after_pca": PCA_K,
        "note": "Embedding semantik kontekstual (768->150 PCA); tiap dimensi tidak interpretatif sendiri, tetapi berkontribusi pada keputusan.",
    }
    return _sanitize(result)


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


# ====================================================================
#  Versi BERTAHAP (streaming) untuk pengujian tiap use case
#  Dipakai oleh endpoint POST /predict/stream.
#  Generator ini meng-'yield' hasil tiap tahap BEGITU tahap itu selesai,
#  sehingga aplikasi bisa menampilkan keluaran UC-01..UC-05 satu per satu
#  (mis. prapemrosesan tampil dulu sambil menunggu ekstraksi fitur, dst).
#
#  Tiap item yang di-yield berbentuk:
#      {"stage": <nama>, "status": "done", "data": {...}}
#  dan sudah dibersihkan (_sanitize) agar aman diserialisasi JSON.
# ====================================================================
def predict_stream(text):
    load()
    t0 = time.time()
    n_words = len(text.split())

    # ---------------- UC-01: Prapemrosesan Teks ----------------
    cleaned = preprocess_for_tfidf(text)
    stage_pre = _stage_preprocessing(text, cleaned)
    yield _sanitize({"stage": "preprocessing", "status": "done", "data": stage_pre})

    # -------- Bangun fitur mentah (untuk UC-02 dan seterusnya) --------
    bert = get_bert_embeddings(text)               # (768,)
    sty_dict = extract_stylometric_features(text)  # 18 fitur
    ppl = calculate_perplexity(text)

    row = {}
    for i, v in enumerate(bert):
        row[f"bert_{i}"] = np.float32(v)
    for i, v in enumerate(sty_dict.values()):
        row[f"sty_{i}"] = np.float32(v)
    row["perplexity"] = np.float32(ppl)
    row["text"] = cleaned
    X = pd.DataFrame([row])

    pipe = _state["pipe"]
    ct = _column_transformer(pipe)

    # ---------------- UC-02: Mengekstraksi Fitur ----------------
    stage_ext = _stage_extraction(ct, X, sty_dict, ppl, bert, cleaned)
    yield _sanitize({"stage": "extraction", "status": "done", "data": stage_ext})

    # ---------------- UC-03: Mereduksi dan Fusi Fitur ----------------
    stage_red = _stage_reduction(ct, X)
    stage_red["tfidf_svd"]["dim_before"] = stage_ext["tfidf"].get("dim")
    yield _sanitize({"stage": "reduction", "status": "done", "data": stage_red})

    # ---------------- UC-04: Mengklasifikasikan Teks ----------------
    prob_ai = float(pipe.predict_proba(X)[0, 1])
    label_int = int(prob_ai >= 0.5)
    stage_cls = {
        "label": "AI" if label_int == 1 else "Manusia",
        "label_int": label_int,
        "prob_ai": _f(prob_ai),
        "prob_human": _f(1.0 - prob_ai),
        "threshold": 0.5,
        "rule": "prob_ai \u2265 0,5 \u2192 AI; selain itu Manusia",
    }
    yield _sanitize({"stage": "classification", "status": "done", "data": stage_cls})

    # ---------------- UC-05: Menampilkan Kontribusi Fitur ----------------
    stage_contrib = compute_group_contributions(pipe, X)
    yield _sanitize({"stage": "contribution", "status": "done", "data": stage_contrib})

    # ---------------- Ringkasan akhir (verdict keseluruhan) ----------------
    summary = {
        "label": "AI" if label_int == 1 else "Manusia",
        "label_int": label_int,
        "prob_ai": _f(prob_ai),
        "prob_human": _f(1.0 - prob_ai),
        "confidence": _f(max(prob_ai, 1.0 - prob_ai)),
        "n_words": n_words,
        "too_short": n_words < MIN_ARTICLE_WORDS,
        "timing_ms": int((time.time() - t0) * 1000),
    }
    yield _sanitize({"stage": "summary", "status": "done", "data": summary})
