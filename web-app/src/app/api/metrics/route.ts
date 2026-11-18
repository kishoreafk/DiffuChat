import { NextResponse } from 'next/server'
import { readFileSync, existsSync } from 'fs'
import path from 'path'

export const runtime = 'nodejs'

/**
 * `metrics.json` ships with the package root (`model/metrics.json` and
 * `metrics/metrics.json` next to the `web-app` folder), not inside the Next.js
 * app itself. The old code only looked at `<cwd>/model/metrics.json`, which
 * never existed, so the Report page always rendered "unavailable".
 *
 * The cwd differs per run mode (dev: `web-app/`, standalone: `web-app/.next/
 * standalone/`), so walk up the tree looking for the artifact.
 */
function resolveMetricsPath(): string | null {
  const explicit = [
    process.env.DIFFUCHAT_METRICS_FILE,
    process.env.DIFFUCHAT_MODEL_DIR
      ? path.join(process.env.DIFFUCHAT_MODEL_DIR, 'metrics.json')
      : null,
  ].filter(Boolean) as string[]

  const candidates: string[] = [...explicit]
  let dir = process.cwd()
  for (let i = 0; i < 5; i++) {
    candidates.push(path.join(dir, 'model', 'metrics.json'))
    candidates.push(path.join(dir, 'metrics', 'metrics.json'))
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }

  for (const c of candidates) {
    const abs = path.resolve(c)
    if (existsSync(abs)) return abs
  }
  return null
}

export async function GET() {
  const p = resolveMetricsPath()
  if (!p) {
    return NextResponse.json({ available: false }, { status: 200 })
  }
  try {
    const metrics = JSON.parse(readFileSync(p, 'utf-8'))
    return NextResponse.json({ available: true, metrics })
  } catch {
    return NextResponse.json({ available: false }, { status: 200 })
  }
}
