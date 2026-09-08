"""
Detektor Teks AI Indonesia - versi Streamlit (satu file, satu proses).

Jalankan dari dalam folder backend/ (yang berisi features.py, inference.py, artifacts/):

    pip install streamlit
    streamlit run streamlit_app.py

UI ini MEMAKAI ULANG inference.predict(), jadi hasil & explainability-nya
identik dengan versi FastAPI. Setiap tahap pipeline (UC-01..UC-05) ditampilkan
secara berurutan agar tiap use case dapat diuji langsung dari aplikasi.
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
    load()
    return True


def fmt(v) -> str:
    if v is None:
        return "-"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def vector_view(label, vector, sample=12, unit="dim"):
    """Tampilkan vektor dengan opsi perbesar/perkecil (sample)."""
    n = len(vector)
    show_all = False
    if n > sample:
        show_all = st.toggle(
            f"Perbesar {label} (tampilkan semua {n} {unit})", value=False, key=f"tg_{label}"
        )
    shown = vector if show_all else vector[:sample]
    txt = ", ".join("-" if v is None else f"{v:.4f}" for v in shown)
    if not show_all and n > sample:
        txt += f"  … (+{n - sample} {unit} lagi)"
    st.code(txt, language=None)


# ------------------------------------------------------------------ Header
st.title("🔍 Detektor Teks AI Indonesia")
st.caption(
    "Multi-fusi fitur (stilometri + perplexity + TF-IDF + IndoBERT) · klasifikasi CatBoost 469-dim"
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
    s = res["stages"]

    # --------------------------------------------------------- Verdict
    if is_ai:
        st.error("### 🤖 Terindikasi AI")
    else:
        st.success("### 🧑 Terindikasi Manusia")

    if res.get("too_short"):
        st.warning(
            f"Teks < {MIN_ARTICLE_WORDS} kata ({res['n_words']} kata) — hasil mungkin kurang andal."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Keyakinan model", f"{round(res['confidence'] * 100)}%")
    c2.metric("Probabilitas AI", f"{round(res['prob_ai'] * 100)}%")
    c3.metric("Probabilitas Manusia", f"{round(res['prob_human'] * 100)}%")
    st.progress(int(round(res["confidence"] * 100)))
    st.caption(f"Waktu proses: {res.get('timing_ms', 0)} ms")

    st.divider()
    st.subheader("Keluaran tiap tahap (pengujian use case)")

    # --------------------------------------------------- UC-01 Prapemrosesan
    pre = s["preprocessing"]
    with st.expander("UC-01 · Melakukan Prapemrosesan Teks", expanded=True):
        a, b, c, d = st.columns(4)
        a.metric("Karakter (mentah)", pre["raw_char_count"])
        b.metric("Kata (mentah)", pre["raw_word_count"])
        c.metric("Token bersih", pre["cleaned_token_count"])
        d.metric("Token dibuang", pre["removed_token_count"])
        st.markdown("**Langkah prapemrosesan:**")
        for i, stp in enumerate(pre["steps"], 1):
            st.markdown(f"{i}. {stp}")
        st.markdown("**Teks bersih (hasil prapemrosesan):**")
        st.text_area("cleaned", pre["cleaned_text"], height=120, label_visibility="collapsed")

    # --------------------------------------------------- UC-02 Ekstraksi Fitur
    ext = s["extraction"]
    with st.expander("UC-02 · Mengekstraksi Fitur (4 fitur, sebelum reduksi)", expanded=True):
        # TF-IDF
        st.markdown(
            f"**1. TF-IDF (sebelum reduksi)** — {ext['tfidf'].get('dim','?')} dim · "
            f"{ext['tfidf'].get('nonzero',0)} term aktif "
            f"(max_features={ext['tfidf']['max_features']}, ngram={ext['tfidf']['ngram_range']}, min_df={ext['tfidf']['min_df']})"
        )
        terms = ext["tfidf"].get("terms", [])
        if terms:
            show_all_terms = st.toggle(f"Perbesar TF-IDF (semua {len(terms)} term)", value=False, key="tg_tfidf_terms")
            shown = terms if show_all_terms else terms[:12]
            st.dataframe(
                {"term": [t["term"] for t in shown], "bobot TF-IDF": [round(t["weight"], 4) for t in shown]},
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("(tidak ada term aktif)")

        # IndoBERT
        st.markdown(f"**2. IndoBERT (sebelum reduksi)** — {ext['indobert']['dim']} dim")
        st.caption(ext["indobert"]["note"])
        vector_view("IndoBERT-768", ext["indobert"]["vector"], sample=12)

        # Stilometri
        st.markdown(f"**3. Stilometri** — {ext['stylometry']['dim']} fitur")
        st.dataframe(
            {
                "fitur": [f["label"] for f in ext["stylometry"]["features"]],
                "nilai": [fmt(f["value"]) for f in ext["stylometry"]["features"]],
            },
            hide_index=True, use_container_width=True,
        )

        # Perplexity
        st.markdown("**4. Perplexity** — 1 nilai")
        pc1, pc2 = st.columns(2)
        pc1.metric("Perplexity (GPT-2 ID)", fmt(ext["perplexity"]["value"]))
        pc2.metric("Jumlah token", ext["perplexity"].get("ppl_token_count") or "-")

    # --------------------------------------------------- UC-03 Reduksi & Fusi
    red = s["reduction"]
    with st.expander("UC-03 · Mereduksi dan Fusi Fitur", expanded=True):
        st.markdown(
            f"**TF-IDF setelah reduksi (TruncatedSVD)** — {red['tfidf_svd'].get('dim_before','?')} → {red['tfidf_svd']['dim_after']}"
        )
        if red["tfidf_svd"].get("explained_variance_ratio_sum") is not None:
            st.caption(f"Variansi terjelaskan (kumulatif): {red['tfidf_svd']['explained_variance_ratio_sum']*100:.1f}%")
        vector_view("TF-IDF-SVD-300", red["tfidf_svd"]["vector"], sample=12)

        st.markdown(
            f"**IndoBERT setelah reduksi (PCA)** — {red['bert_pca']['dim_before']} → {red['bert_pca']['dim_after']}"
        )
        if red["bert_pca"].get("explained_variance_ratio_sum") is not None:
            st.caption(f"Variansi terjelaskan (kumulatif): {red['bert_pca']['explained_variance_ratio_sum']*100:.1f}%")
        vector_view("IndoBERT-PCA-150", red["bert_pca"]["vector"], sample=12)

        st.markdown("**Stilometri (distandarisasi, tidak direduksi)**")
        st.caption(red["stylometry"]["note"])
        vector_view("Stilometri-18", red["stylometry"]["vector"], sample=18)

        st.markdown("**Perplexity (distandarisasi, tidak direduksi)**")
        st.caption(red["perplexity"]["note"])
        vector_view("Perplexity-1", red["perplexity"]["vector"], sample=1)

        comp = red["fusion"]["composition"]
        st.markdown(f"**Vektor fusi akhir** — {red['fusion']['dim']} dim")
        st.caption(" + ".join(f"{k}: {v}" for k, v in comp.items()))
        vector_view("Fusi-469", red["fusion"]["vector"], sample=12)

    # --------------------------------------------------- UC-04 Klasifikasi
    cls = s["classification"]
    with st.expander("UC-04 · Mengklasifikasikan Teks", expanded=True):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Label", cls["label"])
        k2.metric("Ambang", cls["threshold"])
        k3.metric("Prob. AI", fmt(cls["prob_ai"]))
        k4.metric("Prob. Manusia", fmt(cls["prob_human"]))
        st.progress(int(round(cls["prob_ai"] * 100)))
        st.caption(f"Aturan keputusan: {cls['rule']}")

    # --------------------------------------------------- UC-05 Kontribusi Fitur
    groups = s.get("contribution")
    with st.expander("UC-05 · Menampilkan Kontribusi Fitur (SHAP)", expanded=True):
        if groups:
            st.caption(
                "Proporsi besarnya dorongan tiap kelompok terhadap keputusan untuk teks ini (via SHAP CatBoost)."
            )
            for g in groups:
                share = g["share"] * 100
                arrow = "🔴 AI" if g["direction"] == "AI" else "🟢 Manusia"
                st.write(f"**{g['group']}** — {share:.1f}% → {arrow}")
                st.progress(int(round(share)))
        else:
            st.caption("Kontribusi fitur tidak tersedia untuk teks ini.")
