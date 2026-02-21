// lib/powers/embedding-swap.ts
// EmbeddingSwap Power — replaces the default SimpleEmbeddingModel with a
// caller-supplied embedding function via the embed hook.
// When multiple Powers define `embed`, the last-registered one wins.

import type { Power } from "../semantic-engine.ts";

/**
 * EmbeddingSwap Power — overrides the embedding model for both indexing and
 * querying by providing a custom `embed` implementation.
 *
 * Use this to plug in a real ML model (e.g. `@huggingface/transformers`) at
 * runtime without touching the engine's constructor options.
 *
 * @param embedFn An async function that accepts an array of strings and
 *   returns a `Float32Array[]` of unit-normalised embeddings (one per input).
 *
 * @example
 * ```ts
 * import { EmbeddingSwap } from "./lib/powers/embedding-swap.ts";
 *
 * engine.use(EmbeddingSwap(async (texts) => {
 *   // call your ML model here
 *   return texts.map(() => new Float32Array(384).fill(0.1));
 * }));
 * ```
 */
export function EmbeddingSwap(
  embedFn: (texts: string[]) => Promise<Float32Array[]>,
): Power {
  return {
    name: "EmbeddingSwap",
    embed: embedFn,
  };
}
