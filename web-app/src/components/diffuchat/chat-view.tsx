'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import {
  Send,
  Sparkles,
  Loader2,
  Settings2,
  Pin,
  PinOff,
  RefreshCw,
  Waves,
  Dices,
  ChevronDown,
  CircleAlert,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { toast } from 'sonner'

interface Frame {
  text: string
  answerText: string
  ids: number[]
  tokens: string[]
  maskId: number
}

interface Msg {
  id: number
  role: 'user' | 'assistant'
  text?: string
  prompt?: string
  frames?: Frame[]
  frameIdx?: number
  answerStart?: number
  finalIds?: number[]
  locked?: boolean[]
  done?: boolean
  meta?: { passes: number; timeMs: number; strategy: string; seed: number }
  error?: string
}

interface Settings {
  strategy: 'confidence' | 'random' | 'semi_ar'
  steps: number
  answerLen: number
  temperature: number
  seed: number | null
}

const MASK_CHAR = '\u2593'
const FRAME_MS = 160

const SUGGESTIONS = [
  'The city was founded in the year',
  'In economics, inflation is defined as',
  'The study found that patients treated with',
  'The meaning of the term became common in the media',
]

export function ChatView() {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState<Settings>({
    strategy: 'confidence',
    steps: 20,
    answerLen: 48,
    temperature: 0.8,
    seed: null,
  })
  const nextId = useRef(1)
  const timers = useRef<ReturnType<typeof setInterval>[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  const lastMsgRef = useRef<number | null>(null)

  useEffect(() => {
    return () => timers.current.forEach(clearInterval)
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [msgs])

  const animate = useCallback((msgId: number, frames: Frame[]) => {
    let i = 0
    const timer = setInterval(() => {
      i++
      setMsgs((prev) =>
        prev.map((m) =>
          m.id === msgId
            ? { ...m, frameIdx: i, done: i >= frames.length - 1 }
            : m,
        ),
      )
      if (i >= frames.length - 1) {
        clearInterval(timer)
        timers.current = timers.current.filter((t) => t !== timer)
      }
    }, FRAME_MS)
    timers.current.push(timer)
  }, [])

  const send = useCallback(
    async (text: string) => {
      const prompt = text.trim()
      if (!prompt || busy) return
      setInput('')
      setBusy(true)
      const userMsg: Msg = { id: nextId.current++, role: 'user', text: prompt }
      const aiId = nextId.current++
      setMsgs((prev) => [
        ...prev,
        userMsg,
        { id: aiId, role: 'assistant', prompt, frameIdx: 0, done: false },
      ])
      lastMsgRef.current = aiId
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, ...settings }),
        })
        const data = await res.json()
        if (!res.ok) {
          setMsgs((prev) =>
            prev.map((m) =>
              m.id === aiId
                ? { ...m, error: data.message ?? 'Generation failed', done: true }
                : m,
            ),
          )
          toast.error(data.message ?? 'Generation failed')
        } else {
          setMsgs((prev) =>
            prev.map((m) =>
              m.id === aiId
                ? {
                    ...m,
                    frames: data.frames,
                    answerStart: data.answerStart,
                    finalIds: data.finalIds,
                    locked: new Array(data.finalIds.length).fill(false),
                    meta: {
                      passes: data.nForwardPasses,
                      timeMs: data.wallTimeMs,
                      strategy: data.strategy,
                      seed: data.seed,
                    },
                  }
                : m,
            ),
          )
          animate(aiId, data.frames)
        }
      } catch {
        setMsgs((prev) =>
          prev.map((m) =>
            m.id === aiId ? { ...m, error: 'Network error — API unreachable', done: true } : m,
          ),
        )
      } finally {
        setBusy(false)
      }
    },
    [busy, settings, animate],
  )

  const applyEdit = useCallback(
    async (msg: Msg, pos: number, replacement: string) => {
      if (!msg.finalIds || msg.answerStart === undefined) return
      const aiId = msg.id
      setBusy(true)
      try {
        const res = await fetch('/api/edit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ids: msg.finalIds,
            answerStart: msg.answerStart,
            locked: msg.locked,
            edits: [{ pos, text: replacement }],
            steps: settings.steps,
            strategy: settings.strategy,
            seed: msg.meta?.seed ?? 42,
          }),
        })
        const data = await res.json()
        if (!res.ok) {
          toast.error(data.message ?? 'Edit failed')
        } else {
          const frameIdxOffset = 0
          setMsgs((prev) =>
            prev.map((m) =>
              m.id === aiId
                ? {
                    ...m,
                    frames: data.frames,
                    frameIdx: frameIdxOffset,
                    finalIds: data.finalIds,
                    locked: data.locked,
                    done: false,
                  }
                : m,
            ),
          )
          animate(aiId, data.frames)
          toast.success(`Token pinned at position ${pos - (msg.answerStart ?? 0)} — re-denoised around it`)
        }
      } catch {
        toast.error('Network error during edit')
      } finally {
        setBusy(false)
      }
    },
    [busy, settings, animate],
  )

  const regenerate = useCallback(() => {
    const lastUser = [...msgs].reverse().find((m) => m.role === 'user')
    if (lastUser?.text) send(lastUser.text)
  }, [msgs, send])

  return (
    <div className="flex h-screen flex-col bg-slate-950 text-slate-100">
      {/* header */}
      <header className="flex items-center justify-between border-b border-white/10 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600">
            <Waves className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-tight">DiffuChat Playground</h1>
            <p className="text-xs text-slate-400">DenoiseLM · masked-diffusion LM</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSettings((s) => !s)}
            aria-expanded={showSettings}
            className="text-slate-300 hover:bg-white/10"
          >
            <Settings2 className="mr-1.5 h-4 w-4" /> Settings
            <ChevronDown
              className={`ml-1 h-3.5 w-3.5 transition-transform ${showSettings ? 'rotate-180' : ''}`}
            />
          </Button>
        </div>
      </header>

      {/* settings panel */}
      {showSettings && (
        <div className="grid gap-x-8 gap-y-4 border-b border-white/10 bg-slate-900/60 px-4 py-4 sm:grid-cols-2 lg:grid-cols-4 sm:px-6">
          <div>
            <p className="mb-2 text-xs font-medium text-slate-400">Decoding strategy</p>
            <div className="flex rounded-lg border border-white/10 p-0.5">
              {(['confidence', 'random', 'semi_ar'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setSettings((v) => ({ ...v, strategy: s }))}
                  className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
                    settings.strategy === s
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {s === 'semi_ar' ? 'semi-AR' : s}
                </button>
              ))}
            </div>
            {settings.strategy === 'semi_ar' && (
              <p className="mt-1.5 text-[11px] text-slate-500">Block decoding, left → right</p>
            )}
          </div>
          <div>
            <p className="mb-2 text-xs font-medium text-slate-400">
              Denoising steps: <span className="text-blue-300">{settings.steps}</span>
            </p>
            <Slider
              value={[settings.steps]}
              min={2}
              max={64}
              step={1}
              onValueChange={([v]) => setSettings((s) => ({ ...s, steps: v }))}
            />
            <p className="mt-1.5 text-[11px] text-slate-500">
              forward passes to resolve the answer
            </p>
          </div>
          <div>
            <p className="mb-2 text-xs font-medium text-slate-400">
              Answer length: <span className="text-blue-300">{settings.answerLen} tokens</span>
            </p>
            <Slider
              value={[settings.answerLen]}
              min={8}
              max={160}
              step={8}
              onValueChange={([v]) => setSettings((s) => ({ ...s, answerLen: v }))}
            />
          </div>
          <div>
            <p className="mb-2 text-xs font-medium text-slate-400">
              Temperature:{' '}
              <span className="text-blue-300">{settings.temperature.toFixed(2)}</span>
              {settings.temperature === 0 && (
                <span className="ml-1 text-slate-500">(greedy)</span>
              )}
            </p>
            <Slider
              value={[settings.temperature]}
              min={0}
              max={1.5}
              step={0.05}
              onValueChange={([v]) => setSettings((s) => ({ ...s, temperature: v }))}
            />
            <div className="mt-2 flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className="h-7 border-white/10 text-xs text-slate-300 hover:bg-white/10"
                onClick={() =>
                  setSettings((s) => ({ ...s, seed: Math.floor(Math.random() * 1e9) }))
                }
              >
                <Dices className="mr-1 h-3 w-3" /> New seed
              </Button>
              <span className="text-[11px] text-slate-500">
                {settings.seed ?? 'random each send'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {msgs.length === 0 && <EmptyState onPick={(s) => send(s)} />}
          {msgs.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-lg shadow-blue-950/40">
                  {m.text}
                </div>
              </div>
            ) : (
              <AssistantMessage key={m.id} msg={m} onEdit={applyEdit} />
            ),
          )}
        </div>
      </div>

      {/* composer */}
      <div className="border-t border-white/10 bg-slate-950 px-4 py-4 sm:px-6">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send(input)
                }
              }}
              rows={1}
              placeholder="Type a prompt — the answer will denoise in front of you…"
              className="max-h-40 flex-1 resize-none rounded-xl border border-white/10 bg-slate-900 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-500/60 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              aria-label="Prompt"
            />
            {msgs.some((m) => m.role === 'user') && (
              <Button
                variant="outline"
                size="icon"
                onClick={regenerate}
                disabled={busy}
                aria-label="Regenerate last response"
                className="h-11 w-11 shrink-0 border-white/10 text-slate-300 hover:bg-white/10"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            )}
            <Button
              onClick={() => send(input)}
              disabled={busy || !input.trim()}
              className="h-11 w-11 shrink-0 rounded-xl bg-blue-600 p-0 hover:bg-blue-500"
              aria-label="Send"
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          <p className="mt-2 text-center text-[11px] text-slate-600">
            Click any token of a finished answer to edit it — pinned tokens stay locked while the
            rest re-denoises.
          </p>
        </div>
      </div>
    </div>
  )
}

function AssistantMessage({
  msg,
  onEdit,
}: {
  msg: Msg
  onEdit: (msg: Msg, pos: number, replacement: string) => void
}) {
  const [editPos, setEditPos] = useState<number | null>(null)
  const [editVal, setEditVal] = useState('')

  if (msg.error) {
    return (
      <div className="flex justify-start">
        <div className="flex items-start gap-2 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          {msg.error}
        </div>
      </div>
    )
  }

  const frames = msg.frames
  const frame = frames ? frames[Math.min(msg.frameIdx ?? 0, frames.length - 1)] : null
  const isDone = msg.done && frames
  const prevFrame =
    frames && (msg.frameIdx ?? 0) > 0 ? frames[Math.min((msg.frameIdx ?? 0) - 1, frames.length - 1)] : null

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-[92%] rounded-2xl rounded-bl-sm border border-white/10 bg-slate-900 px-4 py-3 shadow-lg shadow-black/20">
        {msg.prompt && (
          <p className="mb-2 border-l-2 border-blue-500/40 pl-2 text-xs text-slate-500">
            re: <span className="text-slate-400">{msg.prompt}</span>
          </p>
        )}
        {!frames && (
          <div className="flex items-center gap-2 py-2 text-sm text-slate-400">
            <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
            denoising from pure noise…
          </div>
        )}
        {frame && (
          <div className="flex flex-wrap items-baseline gap-x-0.5 gap-y-1.5">
            {frame.tokens.map((tok, i) => {
              const abs = (msg.answerStart ?? 0) + i
              const isMask = frame.ids[abs] === frame.maskId
              const justResolved =
                !isMask &&
                prevFrame &&
                prevFrame.ids[(msg.answerStart ?? 0) + i] === prevFrame.maskId
              const isPinned = isDone && msg.locked?.[abs]
              if (isDone && editPos === abs) {
                return (
                  <span key={i} className="inline-flex items-center">
                    <input
                      autoFocus
                      value={editVal}
                      onChange={(e) => setEditVal(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          onEdit(msg, abs, editVal)
                          setEditPos(null)
                        }
                        if (e.key === 'Escape') setEditPos(null)
                      }}
                      onBlur={() => setEditPos(null)}
                      className="w-24 rounded border border-blue-500 bg-slate-800 px-1.5 py-0.5 text-sm text-white focus:outline-none"
                      aria-label={`Replace token ${tok}`}
                    />
                  </span>
                )
              }
              return (
                <button
                  key={i}
                  disabled={!isDone}
                  onClick={() => {
                    setEditVal(isPinned ? tok : tok)
                    setEditPos(abs)
                  }}
                  title={isDone ? `position ${i} — click to edit & re-denoise` : undefined}
                  className={`rounded px-0.5 text-sm leading-relaxed transition-colors ${
                    isMask
                      ? 'animate-pulse rounded-[3px] bg-blue-500/25 text-transparent'
                      : justResolved
                        ? 'rounded-[3px] bg-blue-500/30 text-white'
                        : 'text-slate-100'
                  } ${isDone ? 'cursor-pointer hover:bg-blue-500/25' : 'cursor-default'} ${
                    isPinned ? 'bg-amber-500/20 text-amber-200' : ''
                  }`}
                >
                  {isMask ? MASK_CHAR.repeat(Math.max(2, tok.length || 2)) : tok}
                  {isPinned && (
                    <Pin className="ml-0.5 inline h-3 w-3 align-[-2px] text-amber-400" />
                  )}
                </button>
              )
            })}
          </div>
        )}
        {msg.meta && frames && (
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-white/5 pt-2 text-[11px] text-slate-500">
            <span className="flex items-center gap-1">
              <Sparkles className="h-3 w-3 text-blue-400" />
              {msg.meta.passes} forward {msg.meta.passes === 1 ? 'pass' : 'passes'}
            </span>
            <span>{(msg.meta.timeMs / 1000).toFixed(2)} s wall</span>
            <span>strategy: {msg.meta.strategy === 'semi_ar' ? 'semi-AR' : msg.meta.strategy}</span>
            <span>seed: {msg.meta.seed}</span>
            {isDone && (
              <span className="ml-auto flex items-center gap-1 text-amber-400/80">
                <Pin className="h-3 w-3" />
                {(msg.locked ?? []).filter(Boolean).length} pinned
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  return (
    <div className="py-10 text-center">
      <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-600/20 border border-blue-500/30">
        <Waves className="h-7 w-7 text-blue-400" />
      </div>
      <h2 className="text-xl font-semibold text-white">Watch answers denoise</h2>
      <p className="mx-auto mt-2 max-w-md text-sm text-slate-400">
        DiffuChat starts from an all-<span className="text-blue-300">[MASK]</span> canvas and
        resolves every token in parallel — then lets you pin-edit any token and re-denoise the
        rest around it.
      </p>
      <div className="mx-auto mt-6 grid max-w-xl gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm text-slate-300 transition-colors hover:border-blue-500/40 hover:bg-blue-500/10 hover:text-white"
          >
            {s}
            <span className="mt-0.5 block text-[11px] text-slate-500">
              ▓▓▓▓▓▓▓▓ → denoised text
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
