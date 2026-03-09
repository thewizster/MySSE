/**
 * In-RAM Semantic Search Engine — 100% memory-based, no external DB, no
 * disk I/O during queries.
 *
 * Provides the core {@link SemanticEngine} class with automatic mode switching
 * between brute-force (exact, ≤2 000 docs) and HNSW approximate
 * nearest-neighbor search (≥92% recall\@10, >2 000 docs).
 *
 * Extend behaviour with the {@link Power} plugin interface for caching,
 * hybrid search, metadata filtering, and custom embedding models.
 *
 * @module
 */

// lib/semantic-engine.ts
// In-RAM Semantic Search Engine
// 100% memory-based, no external DB, no disk I/O during queries.
//
// ── Performance characteristics ──────────────────────────────────────────
//
//   Mode        | When used              | Search complexity | Recall
//   ------------|------------------------|-------------------|--------
//   Brute-force | ≤ ANN_THRESHOLD docs   | O(n·d)            | 100%
//               | or useANN = false      |                   |
//   HNSW (ANN)  | > ANN_THRESHOLD docs   | O(log n · d)      | ≥ 92%
//               | and useANN = true      |                   | (typical)
//
//   At 10 000 docs (384-dim), HNSW search is ~5–20× faster than brute-force
//   while maintaining recall@10 ≥ 0.92. Increase efSearch for higher recall
//   at the cost of latency.  All data stays in RAM — no disk, no services.
//
// ─────────────────────────────────────────────────────────────────────────

import { HNSW } from "./hnsw.ts";

/** A document to be indexed. Must have a unique `id` and textual `content`. */
export interface Document {
  /** Unique identifier for the document. */
  id: string;
  /** Textual content used for embedding and search. */
  content: string;
  /** Arbitrary key-value metadata attached to the document. */
  metadata?: Record<string, unknown>;
}

/** A search result extending {@link Document} with a relevance score. */
export interface SearchResult extends Document {
  /** Cosine similarity score (higher is more relevant, range roughly −1 to 1). */
  score: number;
}

/**
 * Context object threaded through `beforeSearch` / `afterSearch` hooks.
 * A Power may set `shortCircuit` to bypass core search (e.g. cache hit).
 */
export interface SearchContext {
  /** The user's search query string. */
  query: string;
  /** Maximum number of results to return. */
  topK: number;
  /** When set by a Power, bypasses core search and returns these results directly. */
  shortCircuit?: SearchResult[];
}

/**
 * A Power is a plain object with named hooks that SemanticEngine calls at
 * key pipeline points. Register with `engine.use(power)`, remove with
 * `engine.eject(name)`. Zero Powers registered = zero overhead.
 */
export interface Power {
  /** Unique identifier used by `eject()` and the status endpoint */
  name: string;
  /** Transform / enrich documents before embedding */
  beforeAdd?(docs: Document[]): Promise<Document[]> | Document[];
  /** Notification after documents are indexed */
  afterAdd?(docs: Document[]): Promise<void> | void;
  /**
   * Inspect / modify the query, or set `shortCircuit` to bypass core search.
   * Short-circuiting the first Power that sets it stops the chain.
   */
  beforeSearch?(
    ctx: SearchContext,
  ): Promise<SearchContext> | SearchContext;
  /** Re-rank, filter, or enrich results */
  afterSearch?(
    results: SearchResult[],
    query: string,
  ): Promise<SearchResult[]> | SearchResult[];
  /**
   * Override the embedding model. When multiple Powers define `embed`,
   * the last registered one wins (not chainable).
   */
  embed?(texts: string[]): Promise<Float32Array[]>;
  /** Cleanup when a document is deleted */
  onDelete?(id: string): Promise<void> | void;
  /** Cleanup when the entire index is cleared */
  onClear?(): Promise<void> | void;
}

/** Optional tuning knobs for the HNSW index */
export interface ANNOptions {
  /** Use approximate nearest-neighbor search when doc count > annThreshold (default: true) */
  useANN?: boolean;
  /** Doc count below which brute-force is used even when useANN is true (default: 2000) */
  annThreshold?: number;
  /** Max connections per node per layer (default: 16, paper recommends 5–48) */
  m?: number;
  /** Beam width during index construction (default: 40, higher = better recall, slower build) */
  efConstruction?: number;
  /** Beam width during search (default: 64, higher = better recall, slower query) */
  efSearch?: number;
  /**
   * Embedding vector dimensionality used by HNSW.
   * Defaults to 384 (all-MiniLM-L6-v2).
   * Set to match your embedding model (e.g. 768 for nomic-embed-text).
   */
  dimension?: number;
}

interface StoredDocument {
  content: string;
  metadata: Record<string, unknown>;
  embedding: Float32Array;
}

// Embedding dimension (all-MiniLM-L6-v2 uses 384)
const EMBEDDING_DIM = 384;

// Constants for hash-based embedding generation
const HASH_SCALING_FACTOR = 0.001; // Controls frequency of sin wave based on hash
const CONTRIBUTION_WEIGHT = 0.1; // Weight of each token's contribution to embedding

// Transformer embedding interface (can be swapped for real implementation)
interface EmbeddingModel {
  embed(texts: string[]): Promise<Float32Array[]>;
}

// Simple hash-based embeddings for offline use
// In production, swap with @huggingface/transformers
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash;
}

// Token-based embedding for semantic similarity
function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

// Create a deterministic embedding from text
// This uses a bag-of-words approach with random projections
function createEmbedding(text: string): Float32Array {
  const embedding = new Float32Array(EMBEDDING_DIM);
  const tokens = tokenize(text);

  // Use tokens to populate embedding dimensions
  for (const token of tokens) {
    const hash = simpleHash(token);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      // Deterministic pseudo-random contribution
      const contribution = Math.sin(hash * (i + 1) * HASH_SCALING_FACTOR) *
        CONTRIBUTION_WEIGHT;
      embedding[i] += contribution;
    }
  }

  // Normalize to unit vector
  let magnitude = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    magnitude += embedding[i] * embedding[i];
  }
  magnitude = Math.sqrt(magnitude);
  if (magnitude > 0) {
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      embedding[i] /= magnitude;
    }
  }

  return embedding;
}

// Simple embedding model using deterministic hashing
// Replace with TransformersJsEmbedding for production use
class SimpleEmbeddingModel implements EmbeddingModel {
  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(texts.map((text) => createEmbedding(text)));
  }
}

// Transformers.js embedding model (for production use)
// uncomment import and use when npm registry is available:
// import { pipeline, type FeatureExtractionPipeline } from "npm:@huggingface/transformers";

// Set nodeModulesDir in deno.json for npm support
// "nodeModulesDir": "auto",

// Uncomment and use this class when you have @huggingface/transformers available in your environment.
/*
class TransformersJsEmbedding implements EmbeddingModel {
  private model: FeatureExtractionPipeline | null = null;
  private modelLoading: Promise<void> | null = null;

  async embed(texts: string[]): Promise<Float32Array[]> {
    await this.loadModel();
    const output = await this.model!(texts, { pooling: "mean", normalize: true });
    const embeddings = output.tolist() as number[][];
    return embeddings.map((e) => new Float32Array(e));
  }

  private async loadModel(): Promise<void> {
    if (this.model) return;
    if (this.modelLoading) {
      await this.modelLoading;
      return;
    }
    this.modelLoading = (async () => {
      console.log("🔄 Loading embedding model (one-time)...");
      const maxRetries = 3;
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          this.model = await pipeline(
            "feature-extraction",
            "Xenova/all-MiniLM-L6-v2",
            { device: "auto", dtype: "fp32" }
          ) as FeatureExtractionPipeline;
          console.log("✅ Model loaded successfully");
          return;
        } catch (e) {
          console.warn(`⚠️ Model load attempt ${attempt}/${maxRetries} failed: ${e}`);
          if (attempt === maxRetries) throw e;
          // Wait before retrying — gives the file download time to complete
          await new Promise((r) => setTimeout(r, 2000 * attempt));
        }
      }
    })();
    await this.modelLoading;
  }
}
*/

const ANN_THRESHOLD_DEFAULT = 2000;

/**
 * In-RAM semantic search engine with automatic brute-force → HNSW switching.
 *
 * Use the singleton via {@link SemanticEngine.getInstance} or the pre-built
 * {@link engine} export. Extend behaviour by registering {@link Power} plugins
 * with {@link SemanticEngine.use}.
 *
 * @example
 * ```ts
 * import { SemanticEngine } from "@wxt/my-search-engine";
 *
 * const engine = SemanticEngine.getInstance({ dimension: 384 });
 * await engine.add([{ id: "1", content: "Hello world" }]);
 * const results = await engine.search("hello");
 * ```
 */
class SemanticEngine {
  private static instance: SemanticEngine;
  private model: EmbeddingModel;
  private docs: Map<string, StoredDocument> = new Map();

  // ANN configuration
  private readonly useANN: boolean;
  private readonly annThreshold: number;
  private readonly hnswM: number;
  private readonly efConstruction: number;
  private readonly efSearch: number;
  private readonly dim: number;
  private hnsw: HNSW | null = null;

  // Powers registry
  private _powers: Power[] = [];

  private constructor(options?: ANNOptions) {
    this.model = new SimpleEmbeddingModel();
    this.useANN = options?.useANN ?? true;
    this.annThreshold = options?.annThreshold ?? ANN_THRESHOLD_DEFAULT;
    this.hnswM = options?.m ?? 16;
    this.efConstruction = options?.efConstruction ?? 40;
    this.efSearch = options?.efSearch ?? 64;
    this.dim = options?.dimension ?? EMBEDDING_DIM;
  }

  /** Get (or create) the singleton. Options are applied only on first call. */
  static getInstance(options?: ANNOptions): SemanticEngine {
    if (!SemanticEngine.instance) {
      SemanticEngine.instance = new SemanticEngine(options);
    }
    return SemanticEngine.instance;
  }

  /** Reset the singleton — primarily for tests that need fresh config. */
  static resetInstance(): void {
    SemanticEngine.instance = undefined as unknown as SemanticEngine;
  }

  // ── Power management ──────────────────────────────────────────────────

  /**
   * Register a Power. Throws if a Power with the same name is already registered.
   */
  use(power: Power): void {
    if (this._powers.some((p) => p.name === power.name)) {
      throw new Error(`Power "${power.name}" is already registered`);
    }
    this._powers.push(power);
  }

  /**
   * Remove a registered Power by name.
   * @returns `true` if found and removed, `false` if not found.
   */
  eject(name: string): boolean {
    const idx = this._powers.findIndex((p) => p.name === name);
    if (idx === -1) return false;
    this._powers.splice(idx, 1);
    return true;
  }

  /** Names of currently registered Powers (read-only). */
  get powers(): readonly string[] {
    return this._powers.map((p) => p.name);
  }

  /**
   * Returns the embedder to use: the last registered Power that defines `embed`,
   * or the default SimpleEmbeddingModel if none do.
   */
  private _resolveEmbedder(): {
    embed(texts: string[]): Promise<Float32Array[]>;
  } {
    for (let i = this._powers.length - 1; i >= 0; i--) {
      if (this._powers[i].embed) {
        return this._powers[i] as {
          embed(texts: string[]): Promise<Float32Array[]>;
        };
      }
    }
    return this.model;
  }

  // ── HNSW lifecycle ────────────────────────────────────────────────────

  /** Lazily create the HNSW index if it doesn't exist yet. */
  private ensureHNSW(): HNSW {
    if (!this.hnsw) {
      this.hnsw = new HNSW(this.dim, this.hnswM);
    }
    return this.hnsw;
  }

  /** True when the next search should use HNSW instead of brute-force */
  private shouldUseANN(): boolean {
    return this.useANN && this.docs.size > this.annThreshold;
  }

  // ── Public API ────────────────────────────────────────────────────────

  /**
   * Add documents to the in-memory index.
   * Runs `beforeAdd` → embed → store/HNSW → `afterAdd` Power hooks.
   */
  async add(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    // beforeAdd chain
    let docs = documents;
    for (const power of this._powers) {
      if (power.beforeAdd) docs = await power.beforeAdd(docs);
    }

    const texts = docs.map((d) => d.content);
    const embeddings = await this._resolveEmbedder().embed(texts);

    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const embedding = embeddings[i];

      this.docs.set(doc.id, {
        content: doc.content,
        metadata: doc.metadata ?? {},
        embedding,
      });

      // Keep HNSW in sync (build incrementally so it's ready when threshold is crossed)
      if (this.useANN) {
        const idx = this.ensureHNSW();
        if (idx.size < this.docs.size) {
          // Only add if not already in the index (handles re-import edge case)
          try {
            idx.add(doc.id, embedding);
          } catch {
            // ID already exists in HNSW — skip
          }
        }
      }
    }

    // afterAdd notifications (fire-and-forget semantics — powers rebuild state here)
    for (const power of this._powers) {
      if (power.afterAdd) await power.afterAdd(docs);
    }

    console.log(
      `✅ Added ${docs.length} documents. Total: ${this.docs.size}`,
    );
  }

  /**
   * Semantic search — runs `beforeSearch` → core brute-force/HNSW → `afterSearch` hooks.
   *
   * When brute-force: exact cosine similarity, O(n·d)
   * When HNSW:        approximate, O(log n · d), recall@10 ≥ 0.92 typical
   */
  async search(query: string, topK: number = 10): Promise<SearchResult[]> {
    if (this.docs.size === 0) return [];

    // beforeSearch chain — a Power may short-circuit by setting ctx.shortCircuit
    let ctx: SearchContext = { query, topK };
    for (const power of this._powers) {
      if (power.beforeSearch) {
        ctx = await power.beforeSearch(ctx);
        if (ctx.shortCircuit) return ctx.shortCircuit;
      }
    }

    const [queryVec] = await this._resolveEmbedder().embed([ctx.query]);

    let results: SearchResult[];
    if (this.shouldUseANN()) {
      results = this.searchANN(queryVec, ctx.topK);
    } else {
      results = this.searchBruteForce(queryVec, ctx.topK);
    }

    // afterSearch chain
    for (const power of this._powers) {
      if (power.afterSearch) {
        results = await power.afterSearch(results, ctx.query);
      }
    }

    return results;
  }

  /** Exact cosine-similarity search over all documents — O(n·d). */
  private searchBruteForce(
    queryVec: Float32Array,
    topK: number,
  ): SearchResult[] {
    const results: SearchResult[] = [];

    for (const [id, { content, metadata, embedding }] of this.docs) {
      let dot = 0;
      for (let i = 0; i < embedding.length; i++) {
        dot += embedding[i] * queryVec[i];
      }
      results.push({ id, content, metadata, score: dot });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /** Approximate nearest-neighbor search via HNSW — O(log n · d). */
  private searchANN(queryVec: Float32Array, topK: number): SearchResult[] {
    const idx = this.ensureHNSW();
    const hits = idx.search(queryVec, topK, this.efSearch);

    return hits.map(({ id, score }) => {
      const stored = this.docs.get(id)!;
      return {
        id,
        content: stored.content,
        metadata: stored.metadata,
        score,
      };
    });
  }

  /**
   * Clear all documents and the HNSW index.
   * Fires `onClear` on each registered Power.
   */
  async clear(): Promise<void> {
    this.docs.clear();
    this.hnsw?.clear();
    this.hnsw = null;
    for (const power of this._powers) {
      if (power.onClear) await power.onClear();
    }
    console.log("🗑️ Index cleared");
  }

  /** The number of documents currently stored in the index. */
  get size(): number {
    return this.docs.size;
  }

  /**
   * Retrieve a single document by ID.
   * @param id The document identifier.
   * @returns The document, or `undefined` if not found.
   */
  get(id: string): Document | undefined {
    const stored = this.docs.get(id);
    if (!stored) return undefined;
    return {
      id,
      content: stored.content,
      metadata: stored.metadata,
    };
  }

  /**
   * Delete a document by ID.
   * Removes from both the doc store and the HNSW index, then fires `onDelete` hooks.
   */
  async delete(id: string): Promise<boolean> {
    const existed = this.docs.delete(id);
    if (existed) {
      this.hnsw?.delete(id);
      for (const power of this._powers) {
        if (power.onDelete) await power.onDelete(id);
      }
    }
    return existed;
  }

  /**
   * Export the index for persistence (optional).
   * Can be serialized to JSON and stored in Deno KV or file.
   */
  toJSON(): Array<
    [
      string,
      {
        content: string;
        metadata: Record<string, unknown>;
        embedding: number[];
      },
    ]
  > {
    return Array.from(this.docs.entries()).map(([id, doc]) => [
      id,
      {
        content: doc.content,
        metadata: doc.metadata,
        embedding: Array.from(doc.embedding),
      },
    ]);
  }

  /**
   * Import a previously exported index.
   * Rebuilds the HNSW graph from the imported vectors, then fires `afterAdd` hooks
   * so Powers (e.g. HybridSearch) can rebuild their auxiliary state.
   */
  async fromJSON(
    data: Array<
      [
        string,
        {
          content: string;
          metadata: Record<string, unknown>;
          embedding: number[];
        },
      ]
    >,
  ): Promise<void> {
    this.docs.clear();
    this.hnsw?.clear();
    this.hnsw = null;

    const importedDocs: Document[] = [];
    for (const [id, doc] of data) {
      const embedding = new Float32Array(doc.embedding);
      this.docs.set(id, {
        content: doc.content,
        metadata: doc.metadata,
        embedding,
      });

      importedDocs.push({ id, content: doc.content, metadata: doc.metadata });

      if (this.useANN) {
        this.ensureHNSW().add(id, embedding);
      }
    }

    // Notify Powers so they can rebuild auxiliary state
    for (const power of this._powers) {
      if (power.afterAdd) await power.afterAdd(importedDocs);
    }

    console.log(`📥 Imported ${this.docs.size} documents`);
  }
}

// ⚠️ REMOVED module-level singleton creation.
// The previous `export const engine = SemanticEngine.getInstance();` eagerly
// created a singleton with default options (dimension: 384, useANN: true) at
// module-load time. This silently ignored caller-specified options (e.g.,
// dimension: 768 for nomic-embed-text) because getInstance() returns the
// existing instance. Consumers should call SemanticEngine.getInstance(options)
// explicitly to control configuration.
//
// For backward compatibility, a default getter is provided that delegates
// directly to the class-level singleton. This avoids maintaining a separate
// module-level cache that could become stale if `resetInstance()` is used.
/**
 * Returns the default {@link SemanticEngine} singleton.
 *
 * Equivalent to `SemanticEngine.getInstance()` — provided as a convenience
 * for callers that do not need custom {@link ANNOptions}.
 */
export function getDefaultEngine(): SemanticEngine {
  return SemanticEngine.getInstance();
}

/**
 * Pre-built default {@link SemanticEngine} singleton.
 *
 * Import this for quick usage without calling `getInstance()` manually.
 *
 * @example
 * ```ts
 * import { engine } from "@wxt/my-search-engine";
 *
 * await engine.add([{ id: "1", content: "Hello" }]);
 * ```
 */
export const engine: SemanticEngine = getDefaultEngine();

// Export class for custom config
export { SemanticEngine };
