/**
 * Byte-level BPE tokenizer (GPT-2 style) for DiffuChat, driven by the
 * tokenizer.json produced during Kaggle training.
 *
 * Faithful port of HF `tokenizers` ByteLevel BPE:
 *   1. text -> UTF-8 bytes -> byte->unicode alphabet (bytes_to_unicode)
 *   2. GPT-2 regex pre-tokenization
 *   3. rank-based BPE merges per pre-token
 * Special tokens [MASK] and <|eos|> are never produced by encode(); input
 * occurrences are stripped upstream (they are structural, not text).
 */

const BYTES_TO_UNICODE = (): string[] => {
  // GPT-2 byte->unicode table: printable bytes map to themselves, the rest
  // are shifted into the 256.. codepoint range.
  const bs: number[] =
    Array.from({ length: 256 }, (_, i) => i)
  const cs: number[] = []
  for (const b of bs) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
      cs.push(b)
    } else {
      cs.push(256 + cs.length)
    }
  }
  const out: string[] = new Array(256)
  bs.forEach((b, i) => (out[b] = String.fromCharCode(cs[i])))
  return out
}

const BYTE_ENCODER = BYTES_TO_UNICODE()
const BYTE_DECODER: Record<string, string> = Object.fromEntries(
  BYTE_ENCODER.map((c, b) => [c, String.fromCharCode(b)]),
)

const PRETOKEN_RE =
  /'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

export interface BpeModel {
  vocab: Record<string, number>
  mergeRanks: Map<string, number>
}

export interface TokenizerJson {
  model: {
    type: string
    vocab: Record<string, number>
    merges: string[] | string[][]
  }
  added_tokens?: { id: number; content: string; special?: boolean }[]
}

const bpeCache = new Map<string, string[]>()

export class ByteLevelBpe {
  vocab: Record<string, number>
  idToToken: string[]
  mergeRanks: Map<string, number>
  specialIds: Record<string, number>

  constructor(model: BpeModel, addedTokens: { id: number; content: string }[] = []) {
    this.vocab = model.vocab
    this.idToToken = Object.keys(model.vocab).sort((a, b) => model.vocab[a] - model.vocab[b])
    this.mergeRanks = model.mergeRanks
    this.specialIds = Object.fromEntries(addedTokens.map((t) => [t.content, t.id]))
    for (const [content, id] of Object.entries(this.specialIds)) {
      this.idToToken[id] = content
    }
  }

  static fromJson(json: TokenizerJson): ByteLevelBpe {
    const ranks = new Map<string, number>()
    if (Array.isArray(json.model.merges)) {
      json.model.merges.forEach((m, i) => {
        if (Array.isArray(m)) ranks.set(m.join(' '), i)
        else ranks.set(m, i)
      })
    }
    const added = (json.added_tokens ?? []).map((t) => ({ id: t.id, content: t.content }))
    return new ByteLevelBpe(
      { vocab: json.model.vocab, mergeRanks: ranks },
      added,
    )
  }

  get vocabSize(): number {
    return this.idToToken.length
  }

  get maskId(): number {
    return this.specialIds['[MASK]']
  }

  get eosId(): number {
    return this.specialIds['<|eos|>']
  }

  /** Core BPE on one pre-token (byte-unicode string) with rank merging. */
  private bpe(token: string): string[] {
    const cached = bpeCache.get(token)
    if (cached) return cached
    let parts = Array.from(token)
    while (parts.length > 1) {
      let bestRank = Infinity
      let bestIdx = -1
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.mergeRanks.get(parts[i] + ' ' + parts[i + 1])
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank
          bestIdx = i
        }
      }
      if (bestIdx < 0) break
      parts = [
        ...parts.slice(0, bestIdx),
        parts[bestIdx] + parts[bestIdx + 1],
        ...parts.slice(bestIdx + 2),
      ]
    }
    bpeCache.set(token, parts)
    return parts
  }

  encode(text: string): number[] {
    const ids: number[] = []
    const matches = text.match(PRETOKEN_RE) ?? []
    for (const pre of matches) {
      const mapped = Array.from(pre)
        .map((ch) => BYTE_ENCODER[ch.charCodeAt(0) & 0xff])
        .join('')
      for (const piece of this.bpe(mapped)) {
        const id = this.vocab[piece]
        if (id !== undefined) ids.push(id)
      }
    }
    return ids
  }

  decodeToken(id: number): string {
    return this.idToToken[id] ?? ''
  }

  /** Human-readable text of one token (byte-unicode -> raw bytes -> utf-8). */
  tokenText(id: number): string {
    const token = this.idToToken[id] ?? ''
    let out = ''
    for (const ch of token) {
      const b = BYTE_DECODER[ch]
      out += b ?? ch
    }
    return out
  }

  /** ids -> raw text (specials decode to their literal form). */
  decode(ids: number[]): string {
    let joined = ''
    for (const id of ids) joined += this.decodeToken(id)
    const bytes: number[] = []
    for (const ch of joined) {
      const b = BYTE_DECODER[ch]
      if (b !== undefined) bytes.push(b.charCodeAt(0))
    }
    return Buffer.from(bytes).toString('utf-8')
  }
}
