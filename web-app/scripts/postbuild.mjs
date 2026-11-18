/**
 * Post-build step for the `output: "standalone"` bundle.
 *
 * Next.js does not copy `.next/static` or `public/` into the standalone
 * server directory by itself. The original package used shell-only
 * `cp -r ...` commands, which fail on Windows (no POSIX `cp`), so this
 * script does the same copy with Node's fs API — cross-platform.
 */
import { cpSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const standalone = path.join(root, ".next", "standalone");

if (!existsSync(standalone)) {
  console.error(
    "[postbuild] .next/standalone not found — did `next build` run with output: \"standalone\"?",
  );
  process.exit(1);
}

const copies = [
  [path.join(root, ".next", "static"), path.join(standalone, ".next", "static")],
  [path.join(root, "public"), path.join(standalone, "public")],
];

for (const [from, to] of copies) {
  if (!existsSync(from)) {
    console.log(`[postbuild] skip (not found): ${path.relative(root, from)}`);
    continue;
  }
  mkdirSync(path.dirname(to), { recursive: true });
  cpSync(from, to, { recursive: true, force: true });
  console.log(`[postbuild] copied ${path.relative(root, from)} -> ${path.relative(root, to)}`);
}

console.log("[postbuild] standalone bundle ready");
