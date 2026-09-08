import { useState, type ReactNode } from "react"
import {
  Sparkles,
  Bot,
  User,
  Loader2,
  AlertTriangle,
  Gauge,
  ScanText,
  Moon,
  Sun,
  Eraser,
  Layers,
  Shrink,
  Combine,
  PieChart,
  ChevronDown,
  ChevronUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { detectText, type PredictResponse } from "@/lib/api"

function fmt(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return "-"
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) >= 1) return v.toFixed(2)
  return v.toFixed(3)
}

/* ---------- Pembungkus tiap Use Case ---------- */
function UseCaseCard({
  code,
  title,
  desc,
  icon,
  children,
}: {
  code: string
  title: string
  desc?: string
  icon: ReactNode
  children: ReactNode
}) {
  return (
    <Card className="animate-fade-in">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            {icon}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="font-mono text-[10px]">
                {code}
              </Badge>
              <CardTitle className="text-base">{title}</CardTitle>
            </div>
            {desc && <CardDescription className="mt-1">{desc}</CardDescription>}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  )
}

function Chip({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border bg-background/50 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-base font-semibold tabular-nums">{value}</div>
    </div>
  )
}

/* ---------- Penampil vektor dengan tombol perbesar/perkecil ---------- */
function VectorView({
  vector,
  sampleSize = 12,
  unit = "dim",
}: {
  vector: number[]
  sampleSize?: number
  unit?: string
}) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? vector : vector.slice(0, sampleSize)
  const hasMore = vector.length > sampleSize
  return (
    <div>
      <div className="flex flex-wrap gap-1.5 rounded-md border bg-muted/40 p-2.5 font-mono text-[11px]">
        {shown.map((v, i) => (
          <span
            key={i}
            className="rounded bg-background px-1.5 py-0.5 tabular-nums text-muted-foreground"
            title={`[${i}]`}
          >
            {v == null ? "-" : v.toFixed(4)}
          </span>
        ))}
        {!expanded && hasMore && (
          <span className="px-1.5 py-0.5 text-muted-foreground">
            … +{vector.length - sampleSize} {unit} lagi
          </span>
        )}
      </div>
      {hasMore && (
        <Button
          variant="ghost"
          size="sm"
          className="mt-2 h-7 gap-1 text-xs"
          onClick={() => setExpanded((e) => !e)}
        >
          {expanded ? (
            <>
              <Shrink className="h-3.5 w-3.5" /> Perkecil (tampilkan {sampleSize} sampel)
            </>
          ) : (
            <>
              <Layers className="h-3.5 w-3.5" /> Perbesar (tampilkan semua {vector.length} {unit})
            </>
          )}
        </Button>
      )}
    </div>
  )
}

/* ---------- Penampil daftar term TF-IDF (perbesar/perkecil) ---------- */
function TermsView({
  terms,
  sampleSize = 12,
}: {
  terms: { term: string; weight: number }[]
  sampleSize?: number
}) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? terms : terms.slice(0, sampleSize)
  const hasMore = terms.length > sampleSize
  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {shown.map((t) => (
          <Badge key={t.term} variant="secondary" className="font-normal">
            {t.term} <span className="ml-1 tabular-nums text-muted-foreground">{t.weight.toFixed(3)}</span>
          </Badge>
        ))}
        {shown.length === 0 && (
          <span className="text-xs text-muted-foreground">(tidak ada term aktif)</span>
        )}
      </div>
      {hasMore && (
        <Button
          variant="ghost"
          size="sm"
          className="mt-2 h-7 gap-1 text-xs"
          onClick={() => setExpanded((e) => !e)}
        >
          {expanded ? (
            <>
              <ChevronUp className="h-3.5 w-3.5" /> Perkecil
            </>
          ) : (
            <>
              <ChevronDown className="h-3.5 w-3.5" /> Perbesar ({terms.length} term)
            </>
          )}
        </Button>
      )}
    </div>
  )
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      {children}
    </div>
  )
}

export default function App() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dark, setDark] = useState(true)

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  function toggleTheme() {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle("dark", next)
  }

  async function handleDetect() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await detectText(text.trim())
      if (res.error) {
        setError(res.error)
      } else {
        setResult(res)
      }
    } catch (e) {
      setError(
        "Gagal menghubungi server. Pastikan backend berjalan di http://localhost:8000.",
      )
    } finally {
      setLoading(false)
    }
  }

  const isAI = result?.label === "AI"
  const confidencePct = result ? Math.round(result.confidence * 100) : 0
  const aiPct = result ? Math.round(result.prob_ai * 100) : 0
  const s = result?.stages

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <div className="mx-auto max-w-3xl px-4 py-10">
        {/* Header */}
        <header className="mb-8 flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <ScanText className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Detektor Teks AI Indonesia
              </h1>
              <p className="text-sm text-muted-foreground">
                Multi-fusi fitur (stilometri + perplexity + TF-IDF + IndoBERT) &middot; CatBoost 469-dim
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Ubah tema">
            {dark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </header>

        {/* Input */}
        <Card className="animate-fade-in">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" /> Tempel teks berita
            </CardTitle>
            <CardDescription>
              Tempel satu artikel berita berbahasa Indonesia, lalu klik Deteksi.
              Aplikasi akan menampilkan keluaran <b>tiap tahap</b> (UC-01 s/d UC-05).
              Disarankan minimal 20 kata agar hasil andal.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Contoh: Jakarta - Pemerintah resmi mengumumkan ..."
              className="min-h-[200px] resize-y text-sm leading-relaxed"
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground tabular-nums">
                {wordCount} kata
              </span>
              <Button onClick={handleDetect} disabled={loading || !text.trim()}>
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Mendeteksi...
                  </>
                ) : (
                  <>
                    <ScanText className="h-4 w-4" /> Deteksi
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <Card className="mt-6 animate-fade-in border-destructive/40">
            <CardContent className="flex items-center gap-3 py-4 text-sm text-destructive">
              <AlertTriangle className="h-5 w-5 shrink-0" />
              {error}
            </CardContent>
          </Card>
        )}

        {/* Result */}
        {result && s && (
          <div className="mt-6 space-y-6">
            {/* Ringkasan verdict */}
            <Card className="animate-fade-in overflow-hidden">
              <div className={isAI ? "h-1.5 w-full bg-destructive" : "h-1.5 w-full bg-success"} />
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <Gauge className="h-4 w-4" /> Hasil Deteksi
                  </CardTitle>
                  <Badge variant={isAI ? "destructive" : "success"} className="gap-1">
                    {isAI ? <Bot className="h-3.5 w-3.5" /> : <User className="h-3.5 w-3.5" />}
                    {isAI ? "Terindikasi AI" : "Terindikasi Manusia"}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-5">
                {result.too_short && (
                  <div className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
                    <AlertTriangle className="h-4 w-4" />
                    Teks &lt; 20 kata ({result.n_words} kata) - hasil mungkin kurang andal.
                  </div>
                )}
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Keyakinan model</span>
                    <span className="font-semibold tabular-nums">{confidencePct}%</span>
                  </div>
                  <Progress value={confidencePct} indicatorClassName={isAI ? "bg-destructive" : "bg-success"} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border p-3">
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Bot className="h-3.5 w-3.5" /> Probabilitas AI
                    </div>
                    <div className="mt-1 text-lg font-bold tabular-nums">{aiPct}%</div>
                  </div>
                  <div className="rounded-lg border p-3">
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <User className="h-3.5 w-3.5" /> Probabilitas Manusia
                    </div>
                    <div className="mt-1 text-lg font-bold tabular-nums">{100 - aiPct}%</div>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">Waktu proses: {result.timing_ms} ms</p>
              </CardContent>
            </Card>

            <div className="flex items-center gap-2 pt-2">
              <Layers className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Keluaran tiap tahap (pengujian use case)
              </h2>
            </div>

            {/* UC-01 Prapemrosesan */}
            <UseCaseCard
              code="UC-01"
              title="Melakukan Prapemrosesan Teks"
              icon={<Eraser className="h-4 w-4" />}
              desc="Membersihkan teks masukan sebelum ekstraksi fitur TF-IDF."
            >
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Chip label="Karakter (mentah)" value={s.preprocessing.raw_char_count} />
                <Chip label="Kata (mentah)" value={s.preprocessing.raw_word_count} />
                <Chip label="Token setelah bersih" value={s.preprocessing.cleaned_token_count} />
                <Chip label="Token dibuang" value={s.preprocessing.removed_token_count} />
              </div>
              <div>
                <SectionLabel>Langkah prapemrosesan</SectionLabel>
                <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm">
                  {s.preprocessing.steps.map((st, i) => (
                    <li key={i}>{st}</li>
                  ))}
                </ol>
              </div>
              <div>
                <SectionLabel>Teks bersih (hasil prapemrosesan)</SectionLabel>
                <div className="mt-2 max-h-40 overflow-auto rounded-md border bg-muted/40 p-3 text-sm leading-relaxed">
                  {s.preprocessing.cleaned_text || "(kosong)"}
                </div>
              </div>
            </UseCaseCard>

            {/* UC-02 Ekstraksi Fitur */}
            <UseCaseCard
              code="UC-02"
              title="Mengekstraksi Fitur"
              icon={<Layers className="h-4 w-4" />}
              desc="Empat fitur diekstraksi (nilai SEBELUM reduksi): TF-IDF, IndoBERT, stilometri, perplexity."
            >
              {/* TF-IDF */}
              <div className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>1. TF-IDF (sebelum reduksi)</SectionLabel>
                  <Badge variant="outline" className="tabular-nums">
                    {s.extraction.tfidf.dim ?? "?"} dim &middot; {s.extraction.tfidf.nonzero ?? 0} aktif
                  </Badge>
                </div>
                <p className="mb-2 text-xs text-muted-foreground">
                  max_features={s.extraction.tfidf.max_features}, ngram={s.extraction.tfidf.ngram_range}, min_df={s.extraction.tfidf.min_df}.
                  Term dengan bobot tertinggi pada teks ini:
                </p>
                <TermsView terms={s.extraction.tfidf.terms} />
              </div>

              {/* IndoBERT */}
              <div className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>2. IndoBERT (sebelum reduksi)</SectionLabel>
                  <Badge variant="outline" className="tabular-nums">{s.extraction.indobert.dim} dim</Badge>
                </div>
                <p className="mb-2 text-xs text-muted-foreground">{s.extraction.indobert.note}</p>
                <VectorView vector={s.extraction.indobert.vector} />
                {s.extraction.indobert.stats && (
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
                    <span>mean: {fmt(s.extraction.indobert.stats.mean)}</span>
                    <span>std: {fmt(s.extraction.indobert.stats.std)}</span>
                    <span>min: {fmt(s.extraction.indobert.stats.min)}</span>
                    <span>max: {fmt(s.extraction.indobert.stats.max)}</span>
                  </div>
                )}
              </div>

              {/* Stilometri */}
              <div className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>3. Stilometri</SectionLabel>
                  <Badge variant="outline" className="tabular-nums">{s.extraction.stylometry.dim} fitur</Badge>
                </div>
                <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                  {s.extraction.stylometry.features.map((f) => (
                    <div key={f.key} className="flex items-center justify-between rounded border bg-background/50 px-2.5 py-1.5 text-xs">
                      <span className="text-muted-foreground">{f.label}</span>
                      <span className="font-mono tabular-nums">{fmt(f.value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Perplexity */}
              <div className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>4. Perplexity</SectionLabel>
                  <Badge variant="outline" className="tabular-nums">1 nilai</Badge>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Chip label="Perplexity (GPT-2 ID)" value={fmt(s.extraction.perplexity.value)} />
                  <Chip label="Jumlah token" value={s.extraction.perplexity.ppl_token_count ?? "-"} />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">Model: {s.extraction.perplexity.model}</p>
              </div>
            </UseCaseCard>

            {/* UC-03 Reduksi & Fusi */}
            <UseCaseCard
              code="UC-03"
              title="Mereduksi dan Fusi Fitur"
              icon={<Shrink className="h-4 w-4" />}
              desc="TF-IDF → SVD 300, IndoBERT → PCA 150, lalu digabung menjadi vektor fusi 469 dimensi."
            >
              {/* TF-IDF SVD */}
              <div className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>TF-IDF setelah reduksi (TruncatedSVD)</SectionLabel>
                  <Badge variant="outline" className="tabular-nums">
                    {s.reduction.tfidf_svd.dim_before ?? "?"} → {s.reduction.tfidf_svd.dim_after}
                  </Badge>
                </div>
                {s.reduction.tfidf_svd.explained_variance_ratio_sum != null && (
                  <p className="mb-2 text-xs text-muted-foreground">
                    Variansi terjelaskan (kumulatif): {(s.reduction.tfidf_svd.explained_variance_ratio_sum * 100).toFixed(1)}%
                  </p>
                )}
                <VectorView vector={s.reduction.tfidf_svd.vector} />
              </div>

              {/* BERT PCA */}
              <div className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>IndoBERT setelah reduksi (PCA)</SectionLabel>
                  <Badge variant="outline" className="tabular-nums">
                    {s.reduction.bert_pca.dim_before} → {s.reduction.bert_pca.dim_after}
                  </Badge>
                </div>
                {s.reduction.bert_pca.explained_variance_ratio_sum != null && (
                  <p className="mb-2 text-xs text-muted-foreground">
                    Variansi terjelaskan (kumulatif): {(s.reduction.bert_pca.explained_variance_ratio_sum * 100).toFixed(1)}%
                  </p>
                )}
                <VectorView vector={s.reduction.bert_pca.vector} />
              </div>

              {/* Sty + Perp (tidak direduksi) */}
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border p-3">
                  <SectionLabel>Stilometri (distandarisasi)</SectionLabel>
                  <p className="my-2 text-xs text-muted-foreground">{s.reduction.stylometry.note}</p>
                  <VectorView vector={s.reduction.stylometry.vector} sampleSize={18} />
                </div>
                <div className="rounded-lg border p-3">
                  <SectionLabel>Perplexity (distandarisasi)</SectionLabel>
                  <p className="my-2 text-xs text-muted-foreground">{s.reduction.perplexity.note}</p>
                  <VectorView vector={s.reduction.perplexity.vector} sampleSize={1} />
                </div>
              </div>

              {/* Fusi */}
              <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <SectionLabel>
                    <span className="inline-flex items-center gap-1"><Combine className="h-3.5 w-3.5" /> Vektor fusi akhir</span>
                  </SectionLabel>
                  <Badge className="tabular-nums">{s.reduction.fusion.dim} dim</Badge>
                </div>
                <div className="mb-2 flex flex-wrap gap-1.5 text-xs">
                  {Object.entries(s.reduction.fusion.composition).map(([k, v]) => (
                    <Badge key={k} variant="secondary" className="font-normal">
                      {k}: {v}
                    </Badge>
                  ))}
                </div>
                <VectorView vector={s.reduction.fusion.vector} />
              </div>
            </UseCaseCard>

            {/* UC-04 Klasifikasi */}
            <UseCaseCard
              code="UC-04"
              title="Mengklasifikasikan Teks"
              icon={<Gauge className="h-4 w-4" />}
              desc="CatBoost mengklasifikasikan vektor fusi. Aturan: prob_ai ≥ 0,5 → AI."
            >
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Chip label="Label" value={s.classification.label} />
                <Chip label="Ambang" value={s.classification.threshold} />
                <Chip label="Prob. AI" value={fmt(s.classification.prob_ai)} />
                <Chip label="Prob. Manusia" value={fmt(s.classification.prob_human)} />
              </div>
              <div>
                <div className="mb-1.5 flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Probabilitas AI vs ambang 0,5</span>
                  <span className="font-semibold tabular-nums">{(s.classification.prob_ai * 100).toFixed(1)}%</span>
                </div>
                <Progress value={s.classification.prob_ai * 100} indicatorClassName={isAI ? "bg-destructive" : "bg-success"} />
              </div>
              <p className="text-xs text-muted-foreground">Aturan keputusan: {s.classification.rule}</p>
            </UseCaseCard>

            {/* UC-05 Kontribusi Fitur */}
            <UseCaseCard
              code="UC-05"
              title="Menampilkan Kontribusi Fitur"
              icon={<PieChart className="h-4 w-4" />}
              desc="Proporsi dorongan tiap kelompok fitur terhadap keputusan untuk teks ini (SHAP CatBoost)."
            >
              {s.contribution && s.contribution.length > 0 ? (
                <div className="space-y-3">
                  {s.contribution.map((g) => (
                    <div key={g.group}>
                      <div className="flex justify-between text-sm">
                        <span>{g.group}</span>
                        <span className={g.direction === "AI" ? "text-destructive" : "text-success"}>
                          {(g.share * 100).toFixed(1)}% → {g.direction}
                        </span>
                      </div>
                      <Progress
                        value={g.share * 100}
                        indicatorClassName={g.direction === "AI" ? "bg-destructive" : "bg-success"}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Kontribusi fitur tidak tersedia untuk teks ini.</p>
              )}
            </UseCaseCard>
          </div>
        )}

        <footer className="mt-10 text-center text-xs text-muted-foreground">
          Model deteksi teks AI untuk berita berbahasa Indonesia &middot; skripsi/jurnal
        </footer>
      </div>
    </div>
  )
}
