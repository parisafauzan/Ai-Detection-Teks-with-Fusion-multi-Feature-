import { useState } from "react"
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

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-background/50 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-base font-semibold tabular-nums">{value}</div>
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

  function fmt(v: number) {
    if (Math.abs(v) >= 100) return v.toFixed(0)
    if (Math.abs(v) >= 1) return v.toFixed(2)
    return v.toFixed(3)
  }

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
                Multi-fitur (stilometri + perplexity + TF-IDF + IndoBERT) &middot; CatBoost
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
        {result && (
          <div className="mt-6 space-y-6">
            <Card className="animate-fade-in overflow-hidden">
              <div
                className={
                  isAI
                    ? "h-1.5 w-full bg-destructive"
                    : "h-1.5 w-full bg-success"
                }
              />
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
                  <Progress
                    value={confidencePct}
                    indicatorClassName={isAI ? "bg-destructive" : "bg-success"}
                  />
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
              </CardContent>
            </Card>

            {/* Penjelasan fitur */}
            <Card className="animate-fade-in">
              <CardHeader>
                <CardTitle className="text-base">Rincian fitur</CardTitle>
                {/* <CardDescription>
                  Nilai fitur yang dihitung dari teks (bahan pertimbangan model).
                </CardDescription> */}
              </CardHeader>
              <CardContent>
                {/* <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <StatCard
                    label="Perplexity (GPT-2 ID)"
                    value={result.features.perplexity == null ? "-" : fmt(result.features.perplexity)}
                  />
                  {result.features.highlight_keys
                    .filter((k) => k !== "perplexity")
                    .map((k) => (
                      <StatCard
                        key={k}
                        label={result.features.labels[k] ?? k}
                        value={
                          result.features.stylometry[k] != null
                            ? fmt(result.features.stylometry[k])
                            : "-"
                        }
                      />
                    ))}
                </div> */}
                <p className="mt-4 text-xs text-muted-foreground">
                  : {result.timing_ms} ms.
                </p>
                {result?.feature_groups && (
  <Card>
    <CardHeader><CardTitle>Kontribusi 4 Kelompok Fitur</CardTitle></CardHeader>
    <CardContent className="space-y-3">
      {result.feature_groups.map((g) => (
        <div key={g.group}>
          <div className="flex justify-between text-sm">
            <span>{g.group}</span>
            <span className={g.direction === "AI" ? "text-red-600" : "text-emerald-600"}>
              {(g.share * 100).toFixed(1)}% → {g.direction}
            </span>
          </div>
          <Progress value={g.share * 100} />
        </div>
      ))}
    </CardContent>
  </Card>
)}

{/* {result?.tfidf_terms && result.tfidf_terms.length > 0 && (
  <Card>
    <CardHeader><CardTitle>TF-IDF — kata kunci teratas</CardTitle></CardHeader>
    <CardContent className="flex flex-wrap gap-2">
      {result.tfidf_terms.map((t) => (
        <Badge key={t.term} variant="secondary">{t.term} · {t.weight.toFixed(3)}</Badge>
      ))}
    </CardContent>
  </Card>
)} */}

{/* {result?.bert_info && (
  <Card>
    <CardHeader><CardTitle>IndoBERT — vektor semantik</CardTitle></CardHeader>
    <CardContent className="text-sm text-muted-foreground">
      {result.bert_info.dims_original} dim → {result.bert_info.dims_after_pca} (PCA). {result.bert_info.note}
    </CardContent>
  </Card>
)} */}
              </CardContent>
            </Card>
          </div>
        )}

        <footer className="mt-10 text-center text-xs text-muted-foreground">
          Model deteksi teks AI untuk berita berbahasa Indonesia &middot; skripsi/jurnal
        </footer>
      </div>
    </div>
  )
}
