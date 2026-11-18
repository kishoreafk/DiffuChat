import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'

const SERVICE = 'http://127.0.0.1:3030'

/** Model status badge (proxied to the inference mini-service). */
export async function GET() {
  try {
    const r = await fetch(`${SERVICE}/status`, { cache: 'no-store' })
    const data = await r.json()
    return NextResponse.json(data, { status: r.status })
  } catch {
    return NextResponse.json(
      {
        ready: false,
        message:
          'Inference service offline. The dedicated diffusion-service process (port 3030) hosts the ONNX checkpoint.',
      },
      { status: 200 },
    )
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.text()
    const r = await fetch(`${SERVICE}/generate`, {
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
        message:
          'The inference service is unreachable. Start it with: cd mini-services/diffusion-service && bun run dev',
      },
      { status: 503 },
    )
  }
}
