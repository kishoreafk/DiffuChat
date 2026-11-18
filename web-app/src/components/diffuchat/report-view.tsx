'use client'

import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import {
  Activity,
  ArrowLeft,
  Cpu,
  Database,
  Gauge,
  GitCompareArrows,
  Globe,
  Loader2,
  Sparkles,
  Timer,
  TrendingDown,
  Zap,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

interface MetricsPayload {
  available: boolean
  metrics?: Record<string, unknown>
}

interface HistPoint {
  step: number
  val_loss: number
  train_loss: number
}

export function ReportView({ onBack }: { onBack?: () => void }) {
  const [data, setData] = useState<MetricsPayload | null>(null)

  useEffect(() => {
    fetch('/api/metrics')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ available: false }))
  }, [])

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-400">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> loading run artifacts…
      </div>
    )
  }

  const m = data.metrics as any
  if (!data.available || !m) {
    return (
      <div className="min-h-screen bg-slate-950 px-4 py-20 text-center text-slate-300">
        <Activity className="mx-auto mb-4 h-10 w-10 text-blue-400" />
        <h1 className="text-xl font-semibold text-white">Training artifacts not deployed yet</h1>
        <p className="mx-auto mt-2 max-w-md text-sm text-slate-400">
          metrics.json could not be loaded from the model directory.
        </p>
      </div>
    )
  }

  const diffHist: HistPoint[] = m.diffusion?.history ?? []
  const arHist: HistPoint[] = m.ar?.history ?? []
  const curveData = diffHist.map((d, i) => ({
    step: d.step,
    diffusion: d.val_loss,
    ar: arHist[i]?.val_loss,
  }))
  const infill = (m.infill_comparison ?? {}) as Record<string, any>
  const speed = (m.speed_bench ?? {}) as any
  const gen = (m.generalization ?? {}) as Record<string, any>
  const coher = (m.coherence_summary ?? {}) as Record<string, any>
  const samples = (m.generation_samples ?? {}) as Record<string, any[]>

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        {onBack && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="mb-4 text-slate-400 hover:bg-white/10 hover:text-white"
          >
            <ArrowLeft className="mr-1.5 h-4 w-4" /> Back
          </Button>
        )}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">DiffuChat Training Report</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            E1 (DenoiseLM, masked diffusion) vs E2 (ARBaseline, causal GPT) — architecture-matched
            (d_model 384 · 8 layers · block 256), identical multi-domain stream (WikiText-103 +
            TinyStories, 175.7M tokens), byte-level BPE 8k, 12,000 steps each on a Kaggle T4.
            Evals use EMA weights.
          </p>
        </div>

        {/* headline metrics */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            icon={<TrendingDown className="h-4 w-4" />}
            label="E1 DenoiseLM val loss (EMA)"
            value={m.diffusion?.best_val_loss_ema?.toFixed(3) ?? '—'}
            sub={`step ${m.diffusion?.history?.slice(-1)[0]?.step ?? '—'} · masked-recovery ${
              gen?.wikitext_val_in_domain
                ? (100 * gen.wikitext_val_in_domain.diffusion_recovery).toFixed(1) + '%'
                : '—'
            } in-domain`}
          />
          <StatCard
            icon={<TrendingDown className="h-4 w-4" />}
            label="E2 AR baseline val loss (EMA)"
            value={m.ar?.best_val_loss_ema?.toFixed(3) ?? '—'}
            sub={`perplexity ${m.ar?.val_ppl_ema?.toFixed(1) ?? '—'}`}
          />
          <StatCard
            icon={<Cpu className="h-4 w-4" />}
            label="GPU"
            value={m.environment?.gpu ?? '—'}
            sub={`${Math.max(m.diffusion?.gpu_mem_gb ?? 0, m.ar?.gpu_mem_gb ?? 0)} GB peak · fp16 AMP`}
          />
          <StatCard
            icon={<Timer className="h-4 w-4" />}
            label="Training time"
            value={`${((m.diffusion?.train_time_s ?? 0) / 60).toFixed(1)} + ${((m.ar?.train_time_s ?? 0) / 60).toFixed(1)} min`}
            sub={`${m.diffusion?.s_per_step?.toFixed(3) ?? '—'} / ${m.ar?.s_per_step?.toFixed(3) ?? '—'} s per step`}
          />
        </div>

        {/* loss curves */}
        <Card className="mt-6 border-white/10 bg-slate-900">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="h-4 w-4 text-blue-400" /> Validation loss curves (EMA weights)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curveData}>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis dataKey="step" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="diffusion"
                    name="DenoiseLM (masked CE)"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="ar"
                    name="ARBaseline (next-token CE)"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Note: the two losses are structurally different objectives (bidirectional masked
              reconstruction vs causal next-token) — compare each model against itself over
              training, and use the task metrics below for cross-model comparison.
            </p>
          </CardContent>
        </Card>

        {/* OOD generalization */}
        <Card className="mt-6 border-white/10 bg-slate-900">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Globe className="h-4 w-4 text-blue-400" /> Out-of-distribution generalization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3 text-xs text-slate-400">
              Training mix: 77.1% WikiText-103 + 22.9% TinyStories. FineWeb-Edu was held out
              entirely. TinyStories tests transfer to the second training domain; FineWeb-Edu
              measures true unseen-corpus degradation.
            </p>
            <Table>
              <TableHeader>
                <TableRow className="border-white/10">
                  <TableHead className="text-slate-400">Eval suite</TableHead>
                  <TableHead className="text-slate-400">Tokens</TableHead>
                  <TableHead className="text-slate-400">Diffusion masked CE</TableHead>
                  <TableHead className="text-slate-400">Diffusion recovery</TableHead>
                  <TableHead className="text-slate-400">AR CE / ppl</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[
                  ['wikitext_val_in_domain', 'WikiText-103 (in-domain)'],
                  ['tinystories_ood', 'TinyStories (2nd domain)'],
                  ['fineweb_edu_ood', 'FineWeb-Edu (unseen)'],
                ].map(([key, label]) => {
                  const v = gen[key]
                  if (!v) return null
                  return (
                    <TableRow key={key} className="border-white/5">
                      <TableCell className="font-medium text-white">{label}</TableCell>
                      <TableCell className="text-slate-300">
                        {(v.tokens / 1e3).toFixed(0)}k
                      </TableCell>
                      <TableCell className="text-blue-300">
                        {v.diffusion_masked_ce.toFixed(3)}
                      </TableCell>
                      <TableCell className="text-emerald-300">
                        {(100 * v.diffusion_recovery).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-amber-300">
                        {v.ar_ce.toFixed(3)} / {v.ar_ppl.toFixed(1)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
            <p className="mt-2 text-xs text-slate-500">
              Diffusion recovers 29.8% of masked tokens on TinyStories (its best domain — simple,
              repetitive narrative) vs 15.5% on WikiText. Both models degrade sharply on the
              unseen FineWeb-Edu corpus, now quantified.
            </p>
          </CardContent>
        </Card>

        {/* infill + speed */}
        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <Card className="border-white/10 bg-slate-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <GitCompareArrows className="h-4 w-4 text-blue-400" /> Infilling: diffusion vs AR
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-3 text-xs text-slate-400">
                A span of L tokens is masked inside validation windows. Diffusion fills it
                bidirectionally in 10 denoise passes; the AR model sees only the left context
                (greedy, left-to-right).
              </p>
              <Table>
                <TableHeader>
                  <TableRow className="border-white/10">
                    <TableHead className="text-slate-400">Span L</TableHead>
                    <TableHead className="text-slate-400">Diffusion recovery</TableHead>
                    <TableHead className="text-slate-400">AR recovery</TableHead>
                    <TableHead className="text-slate-400">Δ</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(infill).map(([L, v]: [string, any]) => (
                    <TableRow key={L} className="border-white/5">
                      <TableCell className="font-medium text-white">{L}</TableCell>
                      <TableCell className="text-blue-300">
                        {(100 * v.diffusion_token_recovery).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-amber-300">
                        {(100 * v.ar_token_recovery).toFixed(1)}%
                      </TableCell>
                      <TableCell
                        className={
                          v.diffusion_token_recovery >= v.ar_token_recovery
                            ? 'text-emerald-400'
                            : 'text-red-400'
                        }
                      >
                        {(100 * (v.diffusion_token_recovery - v.ar_token_recovery)).toFixed(1)} pts
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <p className="mt-2 text-xs text-slate-500">
                Diffusion matches AR at short spans and is 1.9× better at L=16 — the native
                bidirectional advantage grows with span length.
              </p>
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-slate-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-4 w-4 text-blue-400" /> Decode efficiency (64 tokens)
              </CardTitle>
            </CardHeader>
            <CardContent>
              {speed.ar ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="rounded-xl border border-white/10 bg-slate-950 p-3">
                      <p className="text-2xl font-bold text-blue-300">
                        {speed.diffusion_greedy_10pass?.forward_passes ?? '—'}
                      </p>
                      <p className="text-[11px] text-slate-500">diffusion passes</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-slate-950 p-3">
                      <p className="text-2xl font-bold text-amber-300">
                        {speed.ar?.forward_passes ?? '—'}
                      </p>
                      <p className="text-[11px] text-slate-500">AR passes</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-slate-950 p-3">
                      <p className="text-2xl font-bold text-emerald-300">
                        {speed.diffusion_greedy_10pass
                          ? (speed.ar.tokens_per_s / speed.diffusion_greedy_10pass.tokens_per_s).toFixed(1) + '×'
                          : '—'}
                      </p>
                      <p className="text-[11px] text-slate-500">AR wall-clock cost</p>
                    </div>
                  </div>
                  <Table>
                    <TableBody>
                      <TableRow className="border-white/5">
                        <TableCell className="text-slate-400">Diffusion greedy (10-pass)</TableCell>
                        <TableCell className="text-white">
                          {speed.diffusion_greedy_10pass?.tokens_per_s?.toFixed(0)} tok/s ·{' '}
                          {speed.diffusion_greedy_10pass?.tokens_per_forward?.toFixed(1)} tokens/pass ·{' '}
                          {speed.diffusion_greedy_10pass?.time_s}s
                        </TableCell>
                      </TableRow>
                      <TableRow className="border-white/5">
                        <TableCell className="text-slate-400">Diffusion sampled + refine</TableCell>
                        <TableCell className="text-white">
                          {speed.diffusion_sampled_24pass_refine?.tokens_per_s?.toFixed(0)} tok/s ·{' '}
                          {speed.diffusion_sampled_24pass_refine?.tokens_per_forward?.toFixed(1)} tokens/pass ·{' '}
                          {speed.diffusion_sampled_24pass_refine?.time_s}s
                        </TableCell>
                      </TableRow>
                      <TableRow className="border-white/5">
                        <TableCell className="text-slate-400">AR greedy</TableCell>
                        <TableCell className="text-white">
                          {speed.ar?.tokens_per_s?.toFixed(0)} tok/s · 1 token/pass · {speed.ar?.time_s}s
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                  <p className="text-xs text-slate-500">
                    Greedy denoising is 6.8× faster than AR wall-clock on T4 (6.4 tokens per
                    forward pass vs 1). Quality-first sampled decoding still runs 2× faster than AR.
                  </p>
                </div>
              ) : (
                <p className="text-sm text-slate-500">speed benchmark not captured</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* coherence + samples */}
        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <Card className="border-white/10 bg-slate-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Gauge className="h-4 w-4 text-blue-400" /> Sample quality: greedy vs improved sampler
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3 text-center text-xs">
                <div className="rounded-xl border border-white/10 bg-slate-950 p-3">
                  <p className="text-slate-500">dup-bigram rate</p>
                  <p className="mt-1 text-lg font-bold text-red-300">
                    {(100 * (coher.greedy_v2_style?.dup_bigram_rate ?? 0)).toFixed(1)}%
                  </p>
                  <p className="text-[11px] text-slate-600">greedy</p>
                  <p className="mt-1 text-lg font-bold text-emerald-300">
                    {(100 * (coher.improved_temp_top_p?.dup_bigram_rate ?? 0)).toFixed(1)}%
                  </p>
                  <p className="text-[11px] text-slate-600">improved (temp 0.8 + top-p 0.95 + refine)</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-slate-950 p-3">
                  <p className="text-slate-500">top-token share</p>
                  <p className="mt-1 text-lg font-bold text-red-300">
                    {(100 * (coher.greedy_v2_style?.top_token_share ?? 0)).toFixed(1)}%
                  </p>
                  <p className="text-[11px] text-slate-600">greedy</p>
                  <p className="mt-1 text-lg font-bold text-emerald-300">
                    {(100 * (coher.improved_temp_top_p?.top_token_share ?? 0)).toFixed(1)}%
                  </p>
                  <p className="text-[11px] text-slate-600">improved</p>
                </div>
              </div>
              {(samples.improved_temp_top_p ?? []).slice(0, 2).map((s: any, i: number) => (
                <div key={i} className="rounded-xl border border-white/10 bg-slate-950 p-3">
                  <p className="text-xs text-slate-500">prompt</p>
                  <p className="text-sm text-slate-300">{s.prompt}</p>
                  <p className="mt-2 text-xs text-slate-500">greedy completion</p>
                  <p className="text-sm text-red-200/80 line-clamp-2">
                    {samples.greedy_v2_style?.[i]?.completion}
                  </p>
                  <p className="mt-2 text-xs text-slate-500">improved completion</p>
                  <p className="text-sm text-blue-200">{s.completion}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-slate-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Database className="h-4 w-4 text-blue-400" /> Run configuration
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {[
                    ['Experiment', m.config?.experiment],
                    ['Train tokens', `${((m.data?.train_tokens_total ?? 0) / 1e6).toFixed(1)}M (77% WT-103 / 23% TinyStories)`],
                    ['Tokenizer', `${m.data?.tokenizer} · vocab ${m.config?.vocab_size_bpe}`],
                    ['Block size', m.config?.block_size],
                    ['Batch', m.config?.batch_size],
                    ['Steps / model', m.config?.steps_per_model],
                    ['LR schedule', `cosine ${m.config?.lr}, warmup ${m.config?.warmup}`],
                    ['Dropout / EMA', `${m.config?.dropout} / ${m.config?.ema_decay}`],
                    ['DenoiseLM params', m.diffusion?.n_params?.toLocaleString()],
                    ['AR params', m.ar?.n_params?.toLocaleString()],
                  ].map(([k, v]) => (
                    <TableRow key={String(k)} className="border-white/5">
                      <TableCell className="text-slate-400">{k}</TableCell>
                      <TableCell className="text-white">{v}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>

        <p className="mt-8 flex items-center justify-center gap-1.5 text-center text-xs text-slate-600">
          <Sparkles className="h-3 w-3" />
          kernel kishore00afk/diffuchat-generalization-v3 · checkpoints diffusion_best.pt /
          ar_best.pt (EMA, step 12000) · metrics.json
        </p>
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub: string
}) {
  return (
    <Card className="border-white/10 bg-slate-900">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="text-blue-400">{icon}</span>
          {label}
        </div>
        <p className="mt-2 truncate text-2xl font-bold text-white" title={value}>
          {value}
        </p>
        <p className="mt-1 text-[11px] text-slate-500">{sub}</p>
      </CardContent>
    </Card>
  )
}
