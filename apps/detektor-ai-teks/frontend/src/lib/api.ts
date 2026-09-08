// ====================================================================
// Tipe respons /predict dan /predict/stream
// Backend mengembalikan blok `stages` berisi keluaran NYATA tiap tahap
// pipeline, agar tiap use case (UC-01..UC-05) bisa diuji dari aplikasi.
// Versi streaming mengirim tiap tahap secara bertahap (NDJSON).
// ====================================================================

export interface VecStats {
  min: number | null
  max: number | null
  mean: number | null
  std: number | null
  l2_norm: number | null
  nonzero: number
}

export interface TfidfTerm {
  term: string
  weight: number
}

// UC-01
export interface StagePreprocessing {
  raw_char_count: number
  raw_word_count: number
  raw_preview: string
  cleaned_text: string
  cleaned_token_count: number
  removed_token_count: number
  steps: string[]
}

// UC-02
export interface StageExtraction {
  tfidf: {
    max_features: number
    ngram_range: string
    min_df: number
    dim: number | null
    nonzero: number | null
    terms: TfidfTerm[]
    error?: string
  }
  indobert: {
    dim: number
    vector: number[]
    stats: VecStats | null
    note: string
  }
  stylometry: {
    dim: number
    features: { index: number; key: string; label: string; value: number | null }[]
  }
  perplexity: {
    dim: number
    value: number | null
    ppl_token_count: number | null
    model: string
  }
}

// UC-03
export interface ReducedVec {
  dim_before?: number | null
  dim_after?: number
  dim?: number
  vector: number[]
  stats?: VecStats | null
  explained_variance_ratio_sum?: number | null
  method?: string
  reduced?: boolean
  note?: string
}

export interface StageReduction {
  tfidf_svd: ReducedVec
  bert_pca: ReducedVec
  stylometry: ReducedVec
  perplexity: ReducedVec
  fusion: {
    dim: number
    composition: Record<string, number>
    vector: number[]
    stats: VecStats | null
  }
}

// UC-04
export interface StageClassification {
  label: "AI" | "Manusia"
  label_int: number
  prob_ai: number
  prob_human: number
  threshold: number
  rule: string
}

// UC-05
export interface FeatureGroup {
  group: string
  contribution: number
  share: number
  direction: "AI" | "Manusia"
}

// Ringkasan verdict keseluruhan (dikirim terakhir pada mode streaming)
export interface StageSummary {
  label: "AI" | "Manusia"
  label_int: number
  prob_ai: number
  prob_human: number
  confidence: number
  n_words: number
  too_short: boolean
  timing_ms: number
}

export interface Stages {
  preprocessing: StagePreprocessing
  extraction: StageExtraction
  reduction: StageReduction
  classification: StageClassification
  contribution: FeatureGroup[] | null
}

export interface PredictResponse {
  label: "AI" | "Manusia"
  label_int: number
  prob_ai: number
  prob_human: number
  confidence: number
  n_words: number
  too_short: boolean
  features: {
    perplexity: number | null
    stylometry: Record<string, number>
    labels: Record<string, string>
    highlight_keys: string[]
  }
  stages: Stages
  timing_ms: number
  error?: string
  feature_groups?: FeatureGroup[] | null
  tfidf_terms?: TfidfTerm[] | null
  bert_info?: { dims_original: number; dims_after_pca: number; note: string } | null
}

// -------------------- Mode lama: hasil sekaligus --------------------
export async function detectText(text: string): Promise<PredictResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    throw new Error(`Server error ${res.status}`)
  }
  return res.json()
}

// -------------------- Mode baru: streaming bertahap --------------------
// Tiap tahap dikirim sebagai satu baris JSON (NDJSON) begitu selesai.
export type StageEvent =
  | { stage: "preprocessing"; status: "done"; data: StagePreprocessing }
  | { stage: "extraction"; status: "done"; data: StageExtraction }
  | { stage: "reduction"; status: "done"; data: StageReduction }
  | { stage: "classification"; status: "done"; data: StageClassification }
  | { stage: "contribution"; status: "done"; data: FeatureGroup[] | null }
  | { stage: "summary"; status: "done"; data: StageSummary }
  | { stage: "error"; status: "error"; data: { error: string } }

/**
 * Memanggil endpoint streaming dan memanggil `onEvent` setiap satu tahap tiba.
 * Membaca respons sebagai aliran (ReadableStream) lalu memisahkan tiap baris JSON.
 */
export async function detectTextStream(
  text: string,
  onEvent: (ev: StageEvent) => void,
): Promise<void> {
  const res = await fetch("/api/predict/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!res.ok || !res.body) {
    throw new Error(`Server error ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  const flushLine = (line: string) => {
    const trimmed = line.trim()
    if (!trimmed) return
    try {
      onEvent(JSON.parse(trimmed) as StageEvent)
    } catch {
      // Abaikan baris yang belum utuh / bukan JSON valid.
    }
  }

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 1)
      flushLine(line)
    }
  }
  // Sisa buffer terakhir (jika ada tanpa newline penutup).
  flushLine(buffer)
}
