/**
 * MySSE — A fully in-RAM semantic search engine in pure TypeScript.
 *
 * Zero dependencies, HNSW approximate nearest-neighbor indexing, and a
 * pluggable "Powers" system for caching, hybrid BM25+semantic search,
 * metadata filtering, and custom embedding models.
 *
 * @example Basic usage
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
 * @example Using Powers
 * ```ts
 * import { SemanticEngine, QueryCache, HybridSearch } from "@wxt/my-search-engine";
 *
 * const engine = SemanticEngine.getInstance();
 * engine.use(QueryCache({ maxSize: 128, ttl: 60_000 }));
 * engine.use(HybridSearch({ alpha: 0.7 }));
 * ```
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
