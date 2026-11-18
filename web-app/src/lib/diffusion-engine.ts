/**
 * DiffuChat diffusion engine — TypeScript port of diffuchat/generate.py.
 *
 * The canvas is the single source of truth: prompt tokens occupy
 * [0, answerStart) and are never re-masked; the answer region starts
 * all-[MASK] and is denoised iteratively. `locked` flags user pins (edits)
 * that are NEVER re-masked or replaced — the invariant that makes the
 * editable-chat product possible.
 *
 * Inference runs the real DenoiseLM checkpoint exported to ONNX (fp32) via
 * onnxruntime-node. When the checkpoint artifact has not been deployed yet,
 * `isModelReady()` returns false and API routes report an honest status.
 */
import * as ort from 'onnxruntime-node'
import { readFileSync, existsSync } from 'fs'
import path from 'path'
import { ByteLevelBpe, TokenizerJson } from './bpe-tokenizer'

export interface GenConfig {
  strategy: 'confidence' | 'random' | 'semi_ar'
  steps: number
  answerLen: number
  semiArBlock: number
  temperature: number
  seed: number | null
  remaskWindow: number
}

export interface Canvas {
  ids: number[]
  answerStart: number
  maskId: number
  locked: boolean[]
}

export interface Frame {
  ids: number[]
  text: string
  answerText: string
  tokens: string[]
  unmaskedPositions: number[]
  maskId: number
}

export interface GenerationResult {
  frames: Frame[]
  finalIds: number[]
  text: string
  answerText: string
  nForwardPasses: number
  wallTimeMs: number
  strategy: string
  modelReal: boolean
}

const BLOCK_SIZE = 256
const MAX_PASSES = 1024

/**
 * Resolve the directory that holds `diffusion_lm.onnx` / `tokenizer.json`.
 *
 * `DIFFUCHAT_MODEL_DIR` always wins. Otherwise probe the usual layouts:
 * the process cwd (`<cwd>/model`), the Next.js app root, and the package
 * root (the mini-service lives at `<pkg>/web-app/mini-services/diffusion-service`,
 * so the shipped model is three levels up).
 */
function resolveModelDir(): string {
  if (process.env.DIFFUCHAT_MODEL_DIR) return process.env.DIFFUCHAT_MODEL_DIR
  const cwd = process.cwd()
  const candidates = [
    path.join(cwd, 'model'),
    path.join(cwd, '..', 'model'),
    path.join(cwd, '..', '..', 'model'),
    path.join(cwd, '..', '..', '..', 'model'),
  ]
  for (const c of candidates) {
    if (existsSync(path.join(c, 'diffusion_lm.onnx'))) return c
  }
  return path.join(cwd, 'model')
}

const MODEL_DIR = resolveModelDir()
const MODEL_PATH = path.join(MODEL_DIR, 'diffusion_lm.onnx')
const TOKENIZER_PATH = path.join(MODEL_DIR, 'tokenizer.json')
const META_PATH = path.join(MODEL_DIR, 'model_meta.json')

interface ModelAssets {
  session: ort.InferenceSession
  tok: ByteLevelBpe
  meta: { nParams?: number; valLoss?: number; step?: number; exportedFrom?: string }
}

let assets: ModelAssets | null = null

export function isModelReady(): boolean {
  return assets !== null
}

export function modelStatus(): {
  ready: boolean
  nParams?: number
  valLoss?: number
  step?: number
  exportedFrom?: string
} {
  if (!assets) return { ready: false }
  return {
    ready: true,
    nParams: assets.meta.nParams,
    valLoss: assets.meta.valLoss,
    step: assets.meta.step,
    exportedFrom: assets.meta.exportedFrom,
  }
}

export async function loadModel(): Promise<ModelAssets | null> {
  if (assets) return assets
  if (!existsSync(MODEL_PATH) || !existsSync(TOKENIZER_PATH)) return null
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  })
  const tok = ByteLevelBpe.fromJson(
    JSON.parse(readFileSync(TOKENIZER_PATH, 'utf-8')) as TokenizerJson,
  )
  const meta = existsSync(META_PATH)
    ? JSON.parse(readFileSync(META_PATH, 'utf-8'))
    : {}
  assets = { session, tok, meta }
  return assets
}

/** Seeded PRNG (mulberry32) — reproducible generations. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function newCanvas(promptIds: number[], answerLen: number, maskId: number): Canvas {
  return {
    ids: [...promptIds, ...new Array(answerLen).fill(maskId)],
    answerStart: promptIds.length,
    maskId,
    locked: new Array(promptIds.length + answerLen).fill(false),
  }
}

function maskedPositions(c: Canvas): number[] {
  const out: number[] = []
  for (let i = c.answerStart; i < c.ids.length; i++) {
    if (c.ids[i] === c.maskId && !c.locked[i]) out.push(i)
  }
  return out
}

function decodeCanvas(a: ModelAssets, c: Canvas): { text: string; answerText: string } {
  const text = a.tok.decode(c.ids)
  const answerText = a.tok.decode(c.ids.slice(c.answerStart))
  return { text, answerText }
}

async function forward(a: ModelAssets, ids: number[]): Promise<Float32Array> {
  const input = new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [
    1,
    ids.length,
  ])
  const results = await a.session.run({ input_ids: input })
  const logits = results.logits as ort.Tensor
  return logits.data as Float32Array
}

/** One forward pass; unmask k masked positions (mirrors generate.denoise_step). */
async function denoiseStep(
  a: ModelAssets,
  c: Canvas,
  cfg: GenConfig,
  rng: () => number,
  k: number,
  allowed: number[] | null,
  onFrame?: (f: Frame, pos: number[]) => void,
): Promise<number[]> {
  const logits = await forward(a, c.ids)
  const T = c.ids.length
  const V = a.tok.vocabSize
  const lp = (pos: number, tok: number) => logits[pos * V + tok]

  // An UNMASK must never write [MASK] (LLaDA-style)
  const cands = allowed ?? maskedPositions(c)
  if (cands.length === 0 || k <= 0) return []

  interface Pick {
    pos: number
    tok: number
    conf: number
  }
  const picks: Pick[] = []
  const TOP_P = 0.95 // kernel improved sampler: nucleus filtering (v4)
  if (cfg.temperature > 0) {
    for (const p of cands) {
      // temperature softmax over non-mask vocab
      const row: { v: number; w: number }[] = []
      let sum = 0
      for (let v = 0; v < V; v++) {
        if (v === c.maskId) continue
        const w = Math.exp(lp(p, v) / cfg.temperature)
        row.push({ v, w })
        sum += w
      }
      row.sort((x, y) => y.w - x.w)
      // top-p nucleus: smallest set with cumulative prob >= TOP_P
      let cum = 0
      const kept: { v: number; w: number }[] = []
      for (const r of row) {
        kept.push(r)
        cum += r.w
        if (cum / sum >= TOP_P) break
      }
      const ksum = kept.reduce((s, x) => s + x.w, 0)
      let r = rng() * ksum
      let tokId = kept[kept.length - 1].v
      for (const k of kept) {
        r -= k.w
        if (r <= 0) {
          tokId = k.v
          break
        }
      }
      const pTok = (kept.find((x) => x.v === tokId)?.w ?? 0) / ksum
      picks.push({ pos: p, tok: tokId, conf: pTok })
    }
  } else {
    for (const p of cands) {
      let bestTok = -1
      let bestVal = -Infinity
      for (let v = 0; v < V; v++) {
        if (v === c.maskId) continue
        if (lp(p, v) > bestVal) {
          bestVal = lp(p, v)
          bestTok = v
        }
      }
      picks.push({ pos: p, tok: bestTok, conf: bestVal })
    }
  }

  let order: number[]
  if (cfg.strategy === 'random') {
    order = picks.map((_, i) => i)
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
  } else {
    order = picks
      .map((p, i) => i)
      .sort((i, j) => picks[j].conf - picks[i].conf || cands[i] - cands[j])
  }

  const chosen = order.slice(0, Math.min(k, order.length))
  const positions: number[] = []
  for (const j of chosen) {
    c.ids[picks[j].pos] = picks[j].tok
    positions.push(picks[j].pos)
  }
  if (onFrame) onFrame(snapshot(a, c), positions)
  return positions
}

function snapshot(a: ModelAssets, c: Canvas): Frame {
  const { text, answerText } = decodeCanvas(a, c)
  const MASK_CHAR = '\u2593'
  const tokens = c.ids.slice(c.answerStart).map((id) => {
    if (id === c.maskId) return MASK_CHAR
    if (id === a.tok.eosId) return '\u25c6' // end-of-sequence as a subtle glyph
    const s = a.tok.tokenText(id)
    return s === '' ? '\u00b7' : s
  })
  return { ids: [...c.ids], text, answerText, tokens, unmaskedPositions: [], maskId: c.maskId }
}

/**
 * Full generation loop (generate.py): strategy confidence | random | semi_ar.
 * k = ceil(L0/steps) computed ONCE from the initial masked length.
 */
export async function generate(
  prompt: string,
  cfg: GenConfig,
  canvasIn: Canvas | null = null,
  onFrame?: (f: Frame, pos: number[]) => void,
): Promise<GenerationResult & { canvas: Canvas }> {
  const a = (await loadModel())!
  const t0 = Date.now()
  const rng = makeRng(cfg.seed ?? 42)
  const maskId = a.tok.maskId
  let canvas: Canvas

  if (canvasIn) {
    canvas = canvasIn
  } else {
    let promptIds = stripSpecials(a.tok, prompt) ? a.tok.encode(stripSpecials(a.tok, prompt)) : []
    const answerLen = Math.max(1, Math.min(cfg.answerLen, BLOCK_SIZE - 1))
    // keep the END of the prompt (nearest the answer) if too long
    promptIds = promptIds.slice(-(BLOCK_SIZE - answerLen))
    canvas = newCanvas(promptIds, answerLen, maskId)
  }

  const frames: Frame[] = []
  let nPass = 0
  const pushFrame = (f: Frame, pos: number[]) => {
    frames.push({ ...f, unmaskedPositions: pos })
  }
  pushFrame(snapshot(a, canvas), [])

  if (cfg.strategy === 'semi_ar') {
    const blocks: [number, number][] = []
    for (let s = canvas.answerStart; s < canvas.ids.length; s += cfg.semiArBlock) {
      blocks.push([s, Math.min(s + cfg.semiArBlock, canvas.ids.length)])
    }
    const perBlock = Math.max(2, Math.floor(cfg.steps / Math.max(1, blocks.length)))
    for (const [s, e] of blocks) {
      let allowed = range(s, e).filter(
        (i) => canvas.ids[i] === canvas.maskId && !canvas.locked[i],
      )
      while (allowed.length && nPass < MAX_PASSES) {
        const k = Math.max(1, Math.ceil(allowed.length / perBlock))
        await denoiseStep(a, canvas, cfg, rng, Math.min(k, allowed.length), allowed, pushFrame)
        nPass++
        allowed = range(s, e).filter(
          (i) => canvas.ids[i] === canvas.maskId && !canvas.locked[i],
        )
      }
    }
  } else {
    const nMasked0 = maskedPositions(canvas).length
    const k = Math.max(1, Math.ceil(nMasked0 / Math.max(1, cfg.steps)))
    while (nPass < MAX_PASSES) {
      const left = maskedPositions(canvas)
      if (!left.length) break
      await denoiseStep(a, canvas, cfg, rng, Math.min(k, left.length), null, pushFrame)
      nPass++
    }
  }

  // remask-refine self-correction (kernel improved sampler, v4): when sampling,
  // re-mask the least-confident 15% of unlocked answer tokens and re-denoise.
  if (cfg.temperature > 0 && cfg.strategy !== 'semi_ar') {
    const logits = await forward(a, canvas.ids)
    const V = a.tok.vocabSize
    const ansPos = range(canvas.answerStart, canvas.ids.length).filter(
      (i) => !canvas.locked[i],
    )
    const confd = ansPos
      .map((i) => {
        let best = -Infinity
        for (let v = 0; v < V; v++) {
          if (v !== canvas.maskId && logits[i * V + v] > best) best = logits[i * V + v]
        }
        return { i, c: best }
      })
      .sort((x, y) => x.c - y.c)
    const nRemask = Math.max(1, Math.floor(0.15 * confd.length))
    const remasked = confd.slice(0, nRemask).map((x) => x.i)
    for (const i of remasked) canvas.ids[i] = canvas.maskId
    pushFrame(snapshot(a, canvas), remasked)
    let refinePass = 0
    while (nPass < MAX_PASSES && refinePass < 8) {
      const leftR = maskedPositions(canvas)
      if (!leftR.length) break
      const kR = Math.max(1, Math.ceil(leftR.length / 2))
      await denoiseStep(a, canvas, cfg, rng, Math.min(kR, leftR.length), null, pushFrame)
      nPass++
      refinePass++
    }
  }

  // safety net: force-argmax any residual [MASK] (never fires in normal flow)
  const left = range(canvas.answerStart, canvas.ids.length).filter(
    (i) => canvas.ids[i] === canvas.maskId,
  )
  if (left.length) {
    const logits = await forward(a, canvas.ids)
    const V = a.tok.vocabSize
    for (const i of left) {
      let bestTok = 0
      let bestVal = -Infinity
      for (let v = 0; v < V; v++) {
        if (v === canvas.maskId) continue
        if (logits[i * V + v] > bestVal) {
          bestVal = logits[i * V + v]
          bestTok = v
        }
      }
      canvas.ids[i] = bestTok
    }
    nPass++
    pushFrame(snapshot(a, canvas), left)
  }

  const { text, answerText } = decodeCanvas(a, canvas)
  return {
    frames,
    finalIds: canvas.ids,
    text,
    answerText,
    nForwardPasses: nPass,
    wallTimeMs: Date.now() - t0,
    strategy: cfg.strategy,
    modelReal: true,
    canvas,
  }
}

export function decodeAnswerTokens(ids: number[], answerStart: number): string[] {
  return tokenTextsFor(ids, answerStart)
}

/**
 * Edit -> re-mask unlocked answer tokens -> re-denoise (generate.edit).
 * Locked tokens are never re-masked and never replaced.
 */
export async function edit(
  canvas: Canvas,
  edits: { pos: number; newId: number }[],
  cfg: GenConfig,
  onFrame?: (f: Frame, pos: number[]) => void,
): Promise<GenerationResult & { canvas: Canvas }> {
  const a = (await loadModel())!
  if (!edits.length) throw new Error('edit() needs at least one edit')
  for (const e of edits) {
    if (e.pos < canvas.answerStart || e.pos >= canvas.ids.length) {
      throw new Error(`edit position ${e.pos} outside answer region`)
    }
    canvas.ids[e.pos] = e.newId
    canvas.locked[e.pos] = true
  }
  let targets: number[]
  if (cfg.remaskWindow > 0) {
    const set = new Set<number>()
    for (const e of edits) {
      const lo = Math.max(canvas.answerStart, e.pos - cfg.remaskWindow)
      const hi = Math.min(canvas.ids.length, e.pos + cfg.remaskWindow + 1)
      for (let i = lo; i < hi; i++) if (!canvas.locked[i]) set.add(i)
    }
    targets = [...set]
  } else {
    targets = range(canvas.answerStart, canvas.ids.length).filter(
      (i) => !canvas.locked[i],
    )
  }
  for (const i of targets) canvas.ids[i] = canvas.maskId
  return generate('', cfg, canvas, onFrame)
}

/** Encode a user edit word/token back to ids for the edit endpoint. */
export function encodeEdit(a: { encode(s: string): number[] }, text: string): number[] {
  return a.encode(text)
}

function stripSpecials(tok: ByteLevelBpe, text: string): string {
  return text.replaceAll('[MASK]', '').replaceAll('<|eos|>', '')
}

function range(a: number, b: number): number[] {
  const out: number[] = []
  for (let i = a; i < b; i++) out.push(i)
  return out
}

/** Per-token display texts for the answer region (mask renders as '\u2593'). */
export function tokenTextsFor(ids: number[], answerStart: number): string[] {
  if (!assets) return []
  return ids.slice(answerStart).map((id) => {
    const t = assets!.tok.tokenText(id)
    return t === '' ? '\u2593' : t
  })
}

export { BLOCK_SIZE }
