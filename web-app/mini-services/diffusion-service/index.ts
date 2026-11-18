/**
 * DiffuChat inference mini-service (port 3030).
 *
 * Runs the real DenoiseLM ONNX checkpoint in a dedicated process so the
 * Next.js dev server never touches the native onnxruntime addon. The Next
 * API routes (/api/chat, /api/edit) proxy here over 127.0.0.1.
 *
 * Endpoints:
 *   GET  /status   -> { ready, nParams, valLoss, step }
 *   POST /generate -> { prompt, answerLen, steps, strategy, temperature, seed, semiArBlock }
 *   POST /edit     -> { ids, answerStart, locked, edits, steps, strategy, seed, remaskWindow }
 */
import { createServer } from 'http'
import {
  generate,
  edit,
  loadModel,
  modelStatus,
  BLOCK_SIZE,
  type Canvas,
} from '../../src/lib/diffusion-engine'

const PORT = Number(process.env.PORT || 3030)

function writeJson(res: import('http').ServerResponse, code: number, data: unknown) {
  res.writeHead(code, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(data))
}

function readBody(req: import('http').IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let raw = ''
    req.on('data', (c) => {
      raw += c
      if (raw.length > 1e6) reject(new Error('body too large'))
    })
    req.on('end', () => {
      try {
        resolve(raw ? JSON.parse(raw) : {})
      } catch {
        reject(new Error('invalid_json'))
      }
    })
    req.on('error', reject)
  })
}

async function handle(req: import('http').IncomingMessage, res: import('http').ServerResponse) {
  const url = new URL(req.url ?? '/', `http://127.0.0.1:${PORT}`)

  if (req.method === 'GET' && url.pathname === '/status') {
    return writeJson(res, 200, modelStatus())
  }

  if (req.method === 'POST' && url.pathname === '/generate') {
    const body = await readBody(req)
    const prompt = String(body.prompt ?? '')
    if (!prompt.trim()) return writeJson(res, 400, { error: 'empty_prompt' })
    const cfg = {
      strategy: (body.strategy as 'confidence' | 'random' | 'semi_ar') ?? 'confidence',
      steps: Math.min(Math.max(2, Number(body.steps ?? 20)), 64),
      answerLen: Math.min(Math.max(8, Number(body.answerLen ?? 48)), BLOCK_SIZE - 32),
      semiArBlock: Math.min(Math.max(4, Number(body.semiArBlock ?? 24)), 64),
      temperature: Math.max(0, Number(body.temperature ?? 0)),
      seed: body.seed === null || body.seed === undefined ? Math.floor(Math.random() * 1e9) : Number(body.seed),
      remaskWindow: 0,
    }
    const result = await generate(prompt, cfg)
    return writeJson(res, 200, {
      frames: result.frames,
      finalText: result.text,
      answerText: result.answerText,
      nForwardPasses: result.nForwardPasses,
      wallTimeMs: result.wallTimeMs,
      strategy: result.strategy,
      answerStart: result.canvas.answerStart,
      finalIds: result.finalIds,
      seed: cfg.seed,
      model: modelStatus(),
    })
  }

  if (req.method === 'POST' && url.pathname === '/edit') {
    const body = await readBody(req)
    const ids = body.ids as number[]
    const answerStart = Number(body.answerStart)
    if (!Array.isArray(ids) || !Number.isFinite(answerStart)) {
      return writeJson(res, 400, { error: 'ids_and_answerStart_required' })
    }
    const canvas: Canvas = {
      ids,
      answerStart,
      maskId: -1, // resolved inside the engine via loaded assets
      locked: (body.locked as boolean[]) ?? new Array(ids.length).fill(false),
    }
    // maskId must come from the tokenizer — patch it through the edit() path
    const cfg = {
      strategy: (body.strategy as 'confidence' | 'random' | 'semi_ar') ?? 'confidence',
      steps: Math.min(Math.max(2, Number(body.steps ?? 20)), 64),
      answerLen: ids.length - answerStart,
      semiArBlock: 24,
      temperature: 0,
      seed: body.seed === null || body.seed === undefined ? 42 : Number(body.seed),
      remaskWindow: Number(body.remaskWindow ?? 0),
    }
    const edits = (body.edits as { pos: number; text?: string; newId?: number }[]) ?? []
    const result = await editWithResolvedMask(canvas, edits, cfg)
    return writeJson(res, 200, {
      frames: result.frames,
      finalText: result.text,
      answerText: result.answerText,
      finalIds: result.finalIds,
      locked: canvas.locked,
      answerStart,
      nForwardPasses: result.nForwardPasses,
      wallTimeMs: result.wallTimeMs,
      model: modelStatus(),
    })
  }

  return writeJson(res, 404, { error: 'not_found' })
}

/** edit() needs the real mask id; resolve it from the loaded tokenizer. */
async function editWithResolvedMask(
  canvas: Canvas,
  edits: { pos: number; text?: string; newId?: number }[],
  cfg: Parameters<typeof edit>[2],
) {
  const { edit: engineEdit } = await import('../../src/lib/diffusion-engine')
  const { loadModel } = await import('../../src/lib/diffusion-engine')
  const a = await loadModel()
  if (!a) throw new Error('model not loaded')
  canvas.maskId = a.tok.maskId
  const parsed = edits.map((e) =>
    typeof e.newId === 'number'
      ? { pos: e.pos, newId: e.newId }
      : { pos: e.pos, newId: a!.tok.encode(String(e.text ?? ''))[0] ?? 0 },
  )
  return engineEdit(canvas, parsed, cfg)
}

// warm the model once at boot so first request is fast
;(async () => {
  const a = await loadModel()
  console.log(
    `[diffusion-service] model ${a ? 'loaded (ready)' : 'NOT FOUND — waiting for artifacts'} on port ${PORT}`,
  )
})()

const server = createServer((req, res) => {
  handle(req, res).catch((e) => {
    writeJson(res, 500, { error: 'inference_failed', message: e instanceof Error ? e.message : String(e) })
  })
})
server.listen(PORT, '127.0.0.1', () => {
  console.log(`[diffusion-service] listening on 127.0.0.1:${PORT}`)
})
