import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'

const SERVICE = 'http://127.0.0.1:3030'

/** Token-edit + re-denoise (proxied to the inference mini-service). */
export async function POST(req: NextRequest) {
  try {
    const body = await req.text()
    const r = await fetch(`${SERVICE}/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      // @ts-expect-error - bun/undici extended timeout
      timeout: 180_000,
    })
    const data = await r.json()
    return NextResponse.json(data, { status: r.status })
  } catch {
    return NextResponse.json(
      {
        error: 'service_unavailable',
        message: 'The inference service is unreachable.',
      },
      { status: 503 },
    )
  }
}
