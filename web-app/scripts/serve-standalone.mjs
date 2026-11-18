/**
 * Cross-platform launcher for the production standalone server.
 *
 * The original script was `NODE_ENV=production bun .next/standalone/server.js`,
 * which relies on a POSIX env-var prefix and on bun being installed. This
 * wrapper sets NODE_ENV and boots the same standalone entry point with the
 * Node runtime that is already required to run Next.js.
 */
import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const server = path.join(root, ".next", "standalone", "server.js");

if (!existsSync(server)) {
  console.error("[start] .next/standalone/server.js not found — run `npm run build` first.");
  process.exit(1);
}

process.env.NODE_ENV = process.env.NODE_ENV || "production";
process.chdir(path.join(root, ".next", "standalone"));

await import(pathToFileURL(server).href);
