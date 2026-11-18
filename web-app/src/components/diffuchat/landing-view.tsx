'use client'

import {
  Waves,
  Sparkles,
  Pin,
  GitCompareArrows,
  ArrowRight,
  LogIn,
  BarChart3,
  Server,
  Database,
} from 'lucide-react'

export function LandingView({
  onLogin,
  onChat,
  onReport,
}: {
  onLogin: () => void
  onChat: () => void
  onReport: () => void
}) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* nav */}
      <header className="sticky top-0 z-20 border-b border-white/10 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600">
              <Waves className="h-4.5 w-4.5 text-white" />
            </div>
            <span className="font-semibold tracking-tight">DiffuChat</span>
          </div>
          <nav className="flex items-center gap-2">
            <button
              onClick={onReport}
              className="rounded-lg px-3 py-1.5 text-sm text-slate-300 hover:bg-white/10 hover:text-white"
            >
              Report
            </button>
            <button
              onClick={onChat}
              className="rounded-lg px-3 py-1.5 text-sm text-slate-300 hover:bg-white/10 hover:text-white"
            >
              Playground
            </button>
            <button
              onClick={onLogin}
              className="rounded-lg bg-white px-3.5 py-1.5 text-sm font-medium text-slate-900 hover:bg-slate-200"
            >
              Sign in
            </button>
          </nav>
        </div>
      </header>

      {/* hero */}
      <main>
        <section className="relative overflow-hidden">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_50%_0%,rgba(37,99,235,0.18),transparent)]" />
          <div className="mx-auto max-w-6xl px-4 pb-20 pt-20 text-center sm:px-6">
            <h1 className="mx-auto max-w-3xl text-4xl font-bold leading-tight tracking-tight text-white sm:text-6xl">
              Chat with a model that writes in{' '}
              <span className="bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
                any order
              </span>
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-base text-slate-400 sm:text-lg">
              DiffuChat denoises an all-[MASK] canvas into an answer — in parallel, not
              left-to-right. Then edit any token: pinned tokens stay locked while the rest of the
              answer re-flows around your change.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              <button
                onClick={onChat}
                className="group inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white transition hover:bg-blue-500"
              >
                Try the playground
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </button>
              <button
                onClick={onLogin}
                className="inline-flex items-center gap-2 rounded-xl border border-white/15 px-5 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/5"
              >
                <LogIn className="h-4 w-4" /> Sign in
              </button>
              <button
                onClick={onReport}
                className="inline-flex items-center gap-2 rounded-xl border border-white/15 px-5 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/5"
              >
                <BarChart3 className="h-4 w-4" /> Training report
              </button>
            </div>

            {/* terminal demo strip */}
            <div className="mx-auto mt-14 max-w-2xl overflow-hidden rounded-2xl border border-white/10 bg-slate-900 text-left shadow-2xl shadow-blue-950/30">
              <div className="flex items-center gap-1.5 border-b border-white/10 px-4 py-2.5">
                <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
                <span className="ml-2 text-xs text-slate-500">diffuchat — interactive demo</span>
              </div>
              <div className="space-y-3 px-5 py-4 font-mono text-[13px] leading-relaxed">
                <p className="text-slate-400">
                  <span className="text-blue-400">you :</span> how do diffusion models write?
                </p>
                <p>
                  <span className="text-blue-400">ai :</span>{' '}
                  <span className="text-slate-500">▓▓▓▓▓</span>
                  <span className="text-slate-200"> they resolve all tokens</span>{' '}
                  <span className="text-slate-500">▓▓▓</span>
                  <span className="text-slate-200"> at once,</span>
                  <span className="text-slate-500">▓▓</span>
                  <span className="text-slate-200"> not one by one</span>
                </p>
                <p className="text-amber-300/90">
                  ✎ edit: &quot;tokens&quot; → &quot;positions&quot; · pinned · re-denoising…
                </p>
                <p>
                  <span className="text-blue-400">ai :</span> they resolve all{' '}
                  <span className="text-amber-300 underline decoration-amber-400/50">
                    positions
                  </span>{' '}
                  at once, not one by one
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* features */}
        <section className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                icon: <Sparkles className="h-4 w-4" />,
                title: 'Parallel denoising',
                body: 'The MDLM objective trains a bidirectional transformer to reconstruct text from random-ratio masks. Answers resolve in K passes instead of N sequential steps.',
              },
              {
                icon: <Pin className="h-4 w-4" />,
                title: 'Editable answers',
                body: 'Click any token, type a replacement: it is pinned (never re-masked) and the unlocked region re-denoises around your edit — the product primitive.',
              },
              {
                icon: <GitCompareArrows className="h-4 w-4" />,
                title: 'Matched AR baseline',
                body: 'Every claim is checked against an architecture-matched causal GPT trained on identical data, tokenizer and schedule — a controlled comparison, not a vibe.',
              },
              {
                icon: <Server className="h-4 w-4" />,
                title: 'Kaggle-native pipeline',
                body: 'BPE tokenizer, pre-tokenized shards, fp16 AMP, checkpoint resume and metrics.json — sized for free-tier T4 sessions and 20 GB working disk.',
              },
            ].map((f) => (
              <div
                key={f.title}
                className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 transition-colors hover:border-blue-500/30"
              >
                <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600/15 text-blue-400">
                  {f.icon}
                </div>
                <h3 className="text-sm font-semibold text-white">{f.title}</h3>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{f.body}</p>
              </div>
            ))}
          </div>

          {/* pipeline strip */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 rounded-2xl border border-white/10 bg-slate-900/40 px-6 py-5 text-xs text-slate-400">
            <span className="flex items-center gap-2">
              <Database className="h-3.5 w-3.5 text-blue-400" /> WikiText-103 · ~103M tokens ·
              byte-level BPE 8k
            </span>
            <span className="flex items-center gap-2">
              <Server className="h-3.5 w-3.5 text-blue-400" /> Kaggle T4 · fp16 · block 256
            </span>
            <span className="flex items-center gap-2">
              <GitCompareArrows className="h-3.5 w-3.5 text-blue-400" /> E1/E2 identical data +
              schedule
            </span>
          </div>
        </section>
      </main>

      <footer className="mt-auto border-t border-white/10 py-6 text-center text-xs text-slate-600">
        DiffuChat — miniature masked-diffusion LM · MDLM / LLaDA-style · trained on Kaggle free
        tier
      </footer>
    </div>
  )
}
