// ====================================================================
// Tipe respons /predict
// Backend kini mengembalikan blok `stages` berisi keluaran NYATA tiap tahap
// pipeline, agar tiap use case (UC-01..UC-05) bisa diuji dari aplikasi.
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
