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
  timing_ms: number
  error?: string
  feature_groups?: { group: string; contribution: number; share: number; direction: "AI" | "Manusia" }[] | null
  tfidf_terms?: { term: string; weight: number }[] | null
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
