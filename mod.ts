/**
 * # MySSE — In-Memory Semantic Search Engine
 *
 * A lightweight, **zero-dependency** vector search engine built entirely in
 * **pure TypeScript** for **Deno**. Add semantic search to any project in
 * minutes — no databases, no external services, no infrastructure.
 *
 * ## Why MySSE?
 *
 * Most vector search solutions require Postgres + pgvector, a Docker container
 * for Qdrant/Milvus, or a hosted service. MySSE is the opposite: import a
 * single module and you have a working semantic search engine running entirely
 * in-process.
 *
 * - **100% in-memory** — all vectors live in RAM as `Float32Array`, no disk I/O
 * - **HNSW approximate nearest neighbor** — pure-TypeScript implementation of
 *   the Malkov & Yashunin paper; auto-switches between brute-force and HNSW
 * - **Hybrid BM25 + semantic search** — built-in Reciprocal Rank Fusion blends
 *   keyword and dense retrieval
 * - **Pluggable embedding models** — swap to Ollama, Transformers.js, OpenAI,
 *   or any custom embedder at runtime
 * - **Powers plugin system** — lifecycle hooks for caching, hybrid search,
 *   metadata filtering, and custom models without touching core code
 * - **~1 000 lines of code** — read the whole engine in an afternoon
 *
 * ## Performance
 *
 * | Metric | Value |
 * | --- | --- |
 * | Brute-force (10k docs) | ~5 ms/query |
 * | HNSW (10k docs) | ~1.5 ms/query |
 * | HNSW recall@10 | ≥ 92% |
 *
 * ## Quick Start
 *
 * Install from JSR:
 *
 * ```bash
 * deno add jsr:@wxt/my-search-engine
 * ```
 *
 * @example Basic semantic search
 * ```ts
 * import { SemanticEngine } from "@wxt/my-search-engine";
 *
 * const engine = SemanticEngine.getInstance();
 *
 * await engine.add([
 *   { id: "1", content: "Deno is a secure runtime for JS and TS" },
 *   { id: "2", content: "Fresh is a modern web framework for Deno" },
 * ]);
 *
 * const results = await engine.search("secure typescript runtime");
 * console.log(results);
 * ```
 *
 * @example Hybrid search with caching
 * ```ts
 * import { SemanticEngine, QueryCache, HybridSearch } from "@wxt/my-search-engine";
 *
 * const engine = SemanticEngine.getInstance();
 * engine.use(QueryCache({ maxSize: 128, ttl: 60_000 }));
 * engine.use(HybridSearch({ alpha: 0.7 })); // 70% semantic, 30% BM25
 * ```
 *
 * @example Swap in a production embedding model (Ollama)
 * ```ts
 * import { SemanticEngine, EmbeddingSwap } from "@wxt/my-search-engine";
 *
 * const engine = SemanticEngine.getInstance();
 * engine.use(EmbeddingSwap(async (texts) => {
 *   const res = await fetch("http://localhost:11434/api/embed", {
 *     method: "POST",
 *     headers: { "Content-Type": "application/json" },
 *     body: JSON.stringify({ model: "nomic-embed-text", input: texts }),
 *   });
 *   const { embeddings } = await res.json();
 *   return embeddings.map((e: number[]) => new Float32Array(e));
 * }));
 * ```
 *
 * @example Filter results by metadata
 * ```ts
 * import { SemanticEngine, MetadataFilter } from "@wxt/my-search-engine";
 *
 * const engine = SemanticEngine.getInstance();
 * engine.use(MetadataFilter((meta) => meta.published === true));
 *
 * await engine.add([
 *   { id: "1", content: "Public article", metadata: { published: true } },
 *   { id: "2", content: "Draft article", metadata: { published: false } },
 * ]);
 *
 * const results = await engine.search("article"); // only returns published
 * ```
 *
 * See the {@link https://github.com/thewizster/MySSE | GitHub README} for full
 * API reference, architecture details, and the Getting Started guide.
 *
 * @module
 */

// Core engine
export {
  SemanticEngine,
  getDefaultEngine,
  engine,
} from "./lib/semantic-engine.ts";

// Types
export type {
  ANNOptions,
  Document,
  Power,
  SearchContext,
  SearchResult,
} from "./lib/semantic-engine.ts";

// HNSW index
export { HNSW } from "./lib/hnsw.ts";

// Powers
export { QueryCache } from "./lib/powers/cache.ts";
export type { QueryCacheOptions } from "./lib/powers/cache.ts";
export { EmbeddingSwap } from "./lib/powers/embedding-swap.ts";
export { HybridSearch } from "./lib/powers/hybrid-search.ts";
export type { HybridSearchOptions } from "./lib/powers/hybrid-search.ts";
export { MetadataFilter } from "./lib/powers/metadata-filter.ts";
