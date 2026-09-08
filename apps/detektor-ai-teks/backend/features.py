"""Ekstraksi fitur - SALINAN PERSIS dari notebook AI_Detection.ipynb.
Jangan ubah urutan/rumus apa pun: model .joblib dilatih dengan definisi ini.
"""
import re
import numpy as np

# --- 30 kata fungsi Indonesia (FW) ---
ID_FUNCTION_WORDS = {
    "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "dalam", "tidak", "akan", "juga", "sebagai", "oleh", "karena", "namun",
    "tetapi", "agar", "serta", "jika", "maka", "para", "bahwa", "yaitu", "telah", "dapat",
}


def _sentences(t):
    return [s for s in re.split(r"[.!?]+", t) if s.strip()]


def _words(t):
    return re.findall(r"\w+", t.lower())


def extract_stylometric_features(text):
    """18 fitur stilometri (urutan dict = urutan kolom sty_0..sty_17)."""
    f = {}
    f["char_count"] = len(text)
    f["word_count"] = len(text.split())
    sents = _sentences(text)
    f["sentence_count"] = len(sents)
    f["avg_sentence_length"] = f["word_count"] / max(f["sentence_count"], 1)
    words = _words(text)
    f["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    f["unique_word_ratio"] = len(set(words)) / max(len(words), 1)
    f["comma_count"] = text.count(",")
    f["semicolon_count"] = text.count(";")
    f["question_mark_count"] = text.count("?")
    f["exclamation_mark_count"] = text.count("!")
    f["period_count"] = text.count(".")
    f["colon_count"] = text.count(":")
    tot = max(len(text), 1)
    f["punctuation_ratio"] = (
        f["comma_count"] + f["semicolon_count"] + f["question_mark_count"]
        + f["exclamation_mark_count"] + f["period_count"]
    ) / tot
    f["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / tot
    f["digit_ratio"] = sum(1 for c in text if c.isdigit()) / tot
    f["special_char_count"] = len(re.findall(r"[^\w\s]", text))
    f["special_char_ratio"] = f["special_char_count"] / tot
    f["function_word_ratio"] = sum(1 for w in words if w in ID_FUNCTION_WORDS) / max(len(words), 1)
    return f


# label ramah-tampilan untuk beberapa fitur penting (dipakai di UI penjelasan)
STY_LABELS = {
    "unique_word_ratio": "Rasio kata unik (TTR)",
    "avg_sentence_length": "Rata-rata panjang kalimat",
    "avg_word_length": "Rata-rata panjang kata",
    "function_word_ratio": "Rasio kata fungsi",
    "punctuation_ratio": "Rasio tanda baca",
    "sentence_count": "Jumlah kalimat",
    "word_count": "Jumlah kata",
}

# --- stopword Indonesia (Sastrawi) untuk cabang TF-IDF ---
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    ID_STOP = set(StopWordRemoverFactory().get_stop_words())
except Exception:
    ID_STOP = set(
        "yang di ke dari dan atau ini itu dengan untuk pada adalah dalam tidak akan juga "
        "sebagai oleh karena namun tetapi agar serta jika maka para bahwa yaitu".split()
    )


def preprocess_for_tfidf(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in ID_STOP and len(w) > 2]
    return " ".join(words)
