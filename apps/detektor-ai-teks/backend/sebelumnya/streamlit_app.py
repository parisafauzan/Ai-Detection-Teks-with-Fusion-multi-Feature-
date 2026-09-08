"""
Detektor Teks AI Indonesia - versi Streamlit (satu file, satu proses).

Jalankan dari dalam folder backend/ (yang berisi features.py, inference.py, artifacts/):

    pip install streamlit
    streamlit run streamlit_app.py

UI ini MEMAKAI ULANG inference.predict(), jadi hasil & explainability-nya
identik dengan versi FastAPI (perplexity, stilometri, kontribusi 4 kelompok
fitur via SHAP, dan kata kunci TF-IDF).
"""
import streamlit as st

from inference import predict, load, MIN_ARTICLE_WORDS

st.set_page_config(
    page_title="Detektor Teks AI Indonesia",
    page_icon="🔍",
    layout="centered",
)


@st.cache_resource(show_spinner="Memuat model (IndoBERT + GPT-2 + CatBoost) — sekali saja...")
def _warmup():
    """Muat pipeline + transformer sekali, di-cache lintas rerun Streamlit."""
    load()
    return True


def fmt(v: float) -> str:
    if v is None:
        return "-"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


# ------------------------------------------------------------------ Header
st.title("🔍 Detektor Teks AI Indonesia")
st.caption(
    "Multi-fitur (stilometri + perplexity + TF-IDF + IndoBERT) · klasifikasi CatBoost 469-dim"
)

try:
    _warmup()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ------------------------------------------------------------------ Input
text = st.text_area(
    "Tempel teks berita berbahasa Indonesia",
    height=220,
    placeholder="Contoh: Jakarta - Pemerintah resmi mengumumkan ...",
)
n_words = len(text.split())
st.caption(f"{n_words} kata")

if st.button("🔎 Deteksi", type="primary", disabled=not text.strip()):
    with st.spinner("Mendeteksi... (unduh model pertama kali bisa 1-2 menit)"):
        try:
            res = predict(text.strip())
        except Exception as e:  # noqa: BLE001
            st.error(f"Terjadi error saat deteksi: {e}")
            st.stop()

    is_ai = res["label"] == "AI"

    # --------------------------------------------------------- Verdict
    if is_ai:
        st.error(f"### 🤖 Terindikasi AI")
    else:
        st.success(f"### 🧑 Terindikasi Manusia")

    if res.get("too_short"):
        st.warning(
            f"Teks < {MIN_ARTICLE_WORDS} kata ({res['n_words']} kata) — hasil mungkin kurang andal."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Keyakinan model", f"{round(res['confidence'] * 100)}%")
    c2.metric("Probabilitas AI", f"{round(res['prob_ai'] * 100)}%")
    c3.metric("Probabilitas Manusia", f"{round(res['prob_human'] * 100)}%")
    st.progress(int(round(res["confidence"] * 100)))

    # --------------------------------------------------------- Rincian fitur
    feats = res.get("features", {})
    labels = feats.get("labels", {})
    sty = feats.get("stylometry", {})
    st.subheader("Rincian fitur")
    cols = st.columns(3)
    cols[0].metric("Perplexity (GPT-2 ID)", fmt(feats.get("perplexity")))
    i = 1
    for k in feats.get("highlight_keys", []):
        if k == "perplexity":
            continue
        cols[i % 3].metric(labels.get(k, k), fmt(sty.get(k)))
        i += 1

    # --------------------------------------------------------- Kontribusi 4 kelompok
    groups = res.get("feature_groups")
    if groups:
        st.subheader("Kontribusi 4 kelompok fitur")
        st.caption(
            "Proporsi besarnya dorongan tiap kelompok terhadap keputusan untuk teks ini (via SHAP CatBoost)."
        )
        for g in groups:
            share = g["share"] * 100
            arrow = "🔴 AI" if g["direction"] == "AI" else "🟢 Manusia"
            st.write(f"**{g['group']}** — {share:.1f}% → {arrow}")
            st.progress(int(round(share)))

    # --------------------------------------------------------- TF-IDF terms
    terms = res.get("tfidf_terms")
    if terms:
        st.subheader("TF-IDF — kata kunci teratas")
        st.caption(
            "Term berbobot TF-IDF tertinggi pada teks ini (bukan indikator arah AI/Manusia)."
        )
        st.dataframe(
            {
                "term": [t["term"] for t in terms],
                "bobot TF-IDF": [round(t["weight"], 3) for t in terms],
            },
            hide_index=True,
            use_container_width=True,
        )

    # --------------------------------------------------------- IndoBERT note
    binfo = res.get("bert_info")
    if binfo:
        st.subheader("IndoBERT — vektor semantik")
        st.caption(
            f"{binfo['dims_original']} dim → {binfo['dims_after_pca']} (PCA). {binfo['note']}"
        )

    st.caption(f"Waktu proses: {res.get('timing_ms', 0)} ms")
