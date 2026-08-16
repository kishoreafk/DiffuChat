"""DiffuChat terminal product (spec §5.6).

Interactive chat loop where answers materialize in denoising frames and can
be edited token-by-token; the model re-denoises the rest around the edit.

Usage:
    python -m diffuchat.interactive [--preset cpu] [--strategy confidence]
                                    [--steps 20] [--answer-len 96]
    python -m diffuchat.interactive --script "The king said"
                                    # non-interactive: frames + one edit demo

Commands (interactive mode):
    /gen <text>            generate an answer for <text>
    /edit <pos> <text>     pin: 1-indexed answer position <- first chars of
                           <text> (a char or a word); locks them, re-denoises
    /lock <pos-range>      lock positions without changing them, e.g. 5-12 or 7
    /redo [y|n]            re-denoise; y = keep locks (default), n = clear them
    /frames [n]            replay the last n frames of the latest generation
    /exit                  quit

Non-TTY sessions degrade to plain sequential prints (no ANSI codes).  All
state lives in memory; nothing is persisted (privacy by design).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import GenConfig, get_preset
from .data import CharTokenizer, chat_prefix
from .generate import Canvas, GenerationResult, edit, generate
from .train import DEFAULT_OUT, load_model

MASK_GLYPH = "\u2593"


def _utf8_stdout() -> None:
    """Windows-safe console output (Git Bash / cp1252 pipes)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def render_frame(tok: CharTokenizer, ids: list[int], answer_start: int) -> str:
    """Decode a frame with the answer region visibly separated."""
    prompt = tok.decode(ids[:answer_start])
    answer = tok.decode(ids[answer_start:])
    return f"{prompt}[{answer}]"


class ChatSession:
    """All product state (model + canvas + last result) in memory."""

    def __init__(self, ckpt: Path, gcfg: GenConfig):
        self.model, self.tok, self.meta = load_model(ckpt)
        self.gcfg = gcfg
        self.result: GenerationResult | None = None
        self.base_prompt = ""

    # -- generation -------------------------------------------------------
    def do_gen(self, text: str) -> None:
        prompt = chat_prefix(text)
        self.base_prompt = prompt
        cfg = GenConfig(**vars(self.gcfg))
        self.result = generate(self.model, self.tok, prompt, cfg)
        self._print_result("generated")

    def _print_result(self, verb: str) -> None:
        r = self.result
        assert r is not None and r.canvas is not None
        n_frames = len(r.frames) - 1
        print(
            f"\n  ai    : {render_frame(self.tok, r.canvas.ids, r.canvas.answer_start)}"
            f"   ({verb}; {n_frames} frames, {r.n_forward_passes} fwd passes,"
            f" {r.wall_time_s:.1f}s)"
        )

    # -- editing ----------------------------------------------------------
    def do_edit(self, pos1: int, val: str) -> None:
        """pos1 is 1-indexed within the ANSWER region; val = char or word."""
        if self.result is None or self.result.canvas is None:
            print("  (no answer yet — /gen something first)")
            return
        canvas = self.result.canvas
        if not val:
            print("  usage: /edit <pos> <char-or-word>")
            return
        edits = []
        for off, ch in enumerate(val):
            pos = canvas.answer_start + pos1 - 1 + off
            if pos >= len(canvas.ids):
                break
            edits.append((pos, self.tok.encode(ch)[0]))
        if not edits:
            print(f"  position {pos1} outside answer region (1..{canvas.answer_len})")
            return
        before = list(canvas.ids)
        cfg = GenConfig(**vars(self.gcfg))
        self.result = edit(self.model, self.tok, canvas, edits, cfg)
        # invariant display: show the pinned tokens survived
        pinned = self.tok.decode([new_id for _, new_id in edits])
        self._print_result(f"re-denoised around your edit '{pinned}'")

    def do_lock(self, spec: str) -> None:
        if self.result is None or self.result.canvas is None:
            print("  (no answer yet — /gen something first)")
            return
        canvas = self.result.canvas
        try:
            if "-" in spec:
                a, b = spec.split("-", 1)
                lo, hi = int(a), int(b)
            else:
                lo = hi = int(spec)
        except ValueError:
            print("  usage: /lock <pos> or /lock <lo>-<hi> (1-indexed, answer)")
            return
        n = 0
        for p in range(lo, hi + 1):
            pos = canvas.answer_start + p - 1
            if canvas.answer_start <= pos < len(canvas.ids):
                canvas.locked[pos] = True
                n += 1
        print(f"  locked {n} position(s); they will survive future re-denoising")

    def do_redo(self, keep: str = "y") -> None:
        if self.result is None or self.result.canvas is None:
            print("  (no answer yet — /gen something first)")
            return
        canvas = self.result.canvas
        if keep.strip().lower() == "n":
            for i in range(len(canvas.locked)):
                canvas.locked[i] = False
            print("  cleared all locks")
        cfg = GenConfig(**vars(self.gcfg))
        # re-mask unlocked answer tokens, keep prompt, re-denoise
        for i in range(canvas.answer_start, len(canvas.ids)):
            if not canvas.locked[i]:
                canvas.ids[i] = self.tok.mask_token_id
        self.result = generate(self.model, self.tok, "", cfg, canvas=canvas)
        self._print_result("redone")

    def do_frames(self, n: int = 8) -> None:
        if self.result is None:
            print("  (no frames yet — /gen something first)")
            return
        frames = self.result.frames
        answer_start = self.result.canvas.answer_start if self.result.canvas else 0
        total = len(frames) - 1
        idxs = sorted(set([0, *range(max(0, total - n + 1), total + 1)]))
        print()
        for i in idxs:
            tag = "init" if i == 0 else f"frame {i}/{total}"
            print(f"  {tag:>10}: {render_frame(self.tok, frames[i], answer_start)}")

    # -- edit example used by --script ------------------------------------
    def run_edit_example(self) -> None:
        if self.result is None or self.result.canvas is None:
            print("  (no generation to edit)")
            return
        canvas: Canvas = self.result.canvas
        mid = canvas.answer_start + canvas.answer_len // 2
        # pin the char 'q' — rare in English, so survival is visible in output
        new_id = self.tok.encode("q")[0]
        before = list(canvas.ids)
        cfg = GenConfig(**vars(self.gcfg))
        self.result = edit(self.model, self.tok, canvas, [(mid, new_id)], cfg)
        survived = self.result.canvas.ids[mid] == new_id
        print(f"  edit position {mid - canvas.answer_start + 1} (answer-indexed)"
              f" -> 'q'; locked-survival: {'OK' if survived else 'VIOLATED'}")
        self._print_result("re-denoised around your edit")
        changed = sum(
            1
            for i in range(canvas.answer_start, len(canvas.ids))
            if self.result.canvas.ids[i] != before[i]
        )
        print(f"  neighborhood re-flowed: {changed} answer token(s) changed")


def main() -> None:
    ap = argparse.ArgumentParser(description="DiffuChat — the editable chat")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strategy", default="confidence",
                    choices=["confidence", "random", "semi_ar"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--answer-len", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--remask-window", type=int, default=0)
    ap.add_argument("--script", type=str, default=None,
                    help="non-interactive demo: generate + one edit, then exit")
    args = ap.parse_args()

    _utf8_stdout()
    gcfg = GenConfig(
        strategy=args.strategy,
        steps=args.steps,
        answer_len=args.answer_len,
        temperature=args.temperature,
        seed=args.seed,
        remask_window=args.remask_window,
    )

    try:
        session = ChatSession(args.ckpt, gcfg)
    except FileNotFoundError:
        print(f"checkpoint {args.ckpt} not found — train first:")
        print('  python -m diffuchat.train --preset cpu')
        sys.exit(1)

    banner = (
        "DiffuChat — answers appear all at once, and you can edit them.\n"
        "commands: /gen <text>  /edit <pos> <char-or-word>  /lock <lo>-<hi>\n"
        "          /redo [y|n]  /frames [n]  /exit"
    )
    print(banner)

    if args.script is not None:
        print(f"\nyou   : {args.script}")
        session.do_gen(args.script)
        session.do_frames(6)
        print("\nyou   : /edit <mid> q   (scripted demo edit)")
        session.run_edit_example()
        session.do_frames(6)
        print("\n(script mode complete)")
        return

    while True:
        try:
            line = input("\nyou   : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd == "/exit":
            break
        elif cmd == "/gen":
            session.do_gen(rest.strip())
        elif cmd == "/edit":
            parts = rest.split(None, 1)
            if len(parts) != 2 or not parts[0].lstrip("-").isdigit():
                print("  usage: /edit <pos> <char-or-word>")
            else:
                session.do_edit(int(parts[0]), parts[1])
        elif cmd == "/lock":
            session.do_lock(rest.strip())
        elif cmd == "/redo":
            session.do_redo(rest.strip() or "y")
        elif cmd == "/frames":
            n = int(rest) if rest.strip().isdigit() else 8
            session.do_frames(n)
        else:
            # plain text = /gen
            session.do_gen(line)
    print("bye — nothing was saved (privacy by design)")


if __name__ == "__main__":
    main()
