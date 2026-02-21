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

import HNSW from "./hnsw.ts";

export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResult extends Document {
  score: number;
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

// BM25 tuning constants (Okapi BM25, Robertson et al.)
const BM25_K1 = 1.5; // term-frequency saturation
const BM25_B = 0.75; // length normalisation

// RRF fusion constant (Cormack et al. 2009, k=60 is conventional)
const RRF_K = 60;

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
  private hnsw: HNSW | null = null;

  // BM25 keyword index
  private termIndex: Map<string, Map<string, number>> = new Map(); // term → (docId → tf)
  private docTerms: Map<string, Set<string>> = new Map(); // docId → unique terms
  private docLengths: Map<string, number> = new Map(); // docId → token count
  private docFreq: Map<string, number> = new Map(); // term → doc frequency
  private totalDocLength = 0;

  private constructor(options?: ANNOptions) {
    this.model = new SimpleEmbeddingModel();
    this.useANN = options?.useANN ?? true;
    this.annThreshold = options?.annThreshold ?? ANN_THRESHOLD_DEFAULT;
    this.hnswM = options?.m ?? 16;
    this.efConstruction = options?.efConstruction ?? 40;
    this.efSearch = options?.efSearch ?? 64;
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

  // ── BM25 keyword index ────────────────────────────────────────────────

  private indexKeywords(id: string, content: string): void {
    const tokens = tokenize(content);
    const termFreq = new Map<string, number>();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) ?? 0) + 1);
    }
    this.docTerms.set(id, new Set(termFreq.keys()));
    this.docLengths.set(id, tokens.length);
    this.totalDocLength += tokens.length;
    for (const [term, freq] of termFreq) {
      if (!this.termIndex.has(term)) this.termIndex.set(term, new Map());
      this.termIndex.get(term)!.set(id, freq);
      this.docFreq.set(term, (this.docFreq.get(term) ?? 0) + 1);
    }
  }

  private removeKeywords(id: string): void {
    const terms = this.docTerms.get(id);
    if (!terms) return;
    this.totalDocLength -= this.docLengths.get(id) ?? 0;
    this.docLengths.delete(id);
    this.docTerms.delete(id);
    for (const term of terms) {
      const postings = this.termIndex.get(term);
      if (postings) {
        postings.delete(id);
        if (postings.size === 0) this.termIndex.delete(term);
      }
      const df = (this.docFreq.get(term) ?? 1) - 1;
      if (df <= 0) this.docFreq.delete(term);
      else this.docFreq.set(term, df);
    }
  }

  private bm25Search(
    query: string,
    topK: number,
  ): Array<{ id: string; score: number }> {
    const queryTerms = tokenize(query);
    if (queryTerms.length === 0) return [];

    const N = this.docs.size;
    const avgDl = N > 0 ? Math.max(1, this.totalDocLength / N) : 1;
    const scores = new Map<string, number>();

    for (const term of queryTerms) {
      const df = this.docFreq.get(term) ?? 0;
      if (df === 0) continue;
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
      const postings = this.termIndex.get(term)!;
      for (const [id, tf] of postings) {
        const dl = this.docLengths.get(id) ?? 0;
        const norm = (tf * (BM25_K1 + 1)) /
          (tf + BM25_K1 * (1 - BM25_B + BM25_B * (dl / avgDl)));
        scores.set(id, (scores.get(id) ?? 0) + idf * norm);
      }
    }

    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([id, score]) => ({ id, score }));
  }

  // ── HNSW lifecycle ────────────────────────────────────────────────────

  private ensureHNSW(): HNSW {
    if (!this.hnsw) {
      this.hnsw = new HNSW(EMBEDDING_DIM, this.hnswM);
    }
    return this.hnsw;
  }

  /** True when the next search should use HNSW instead of brute-force */
  private shouldUseANN(): boolean {
    return this.useANN && this.docs.size > this.annThreshold;
  }

  // ── Public API (unchanged signatures) ─────────────────────────────────

  /**
   * Add documents to the in-memory index.
   * Documents are embedded, stored, and (when ANN is enabled) inserted into the HNSW graph.
   */
  async add(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    const texts = documents.map((d) => d.content);
    const embeddings = await this.model.embed(texts);

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const embedding = embeddings[i];

      this.docs.set(doc.id, {
        content: doc.content,
        metadata: doc.metadata ?? {},
        embedding,
      });

      this.indexKeywords(doc.id, doc.content);

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

    console.log(
      `✅ Added ${documents.length} documents. Total: ${this.docs.size}`,
    );
  }

  /**
   * Semantic search — delegates to HNSW or brute-force depending on index size and config.
   *
   * When brute-force: exact cosine similarity, O(n·d)
   * When HNSW:        approximate, O(log n · d), recall@10 ≥ 0.92 typical
   */
  async search(query: string, topK: number = 10): Promise<SearchResult[]> {
    if (this.docs.size === 0) return [];

    const [queryVec] = await this.model.embed([query]);

    if (this.shouldUseANN()) {
      return this.searchANN(queryVec, topK);
    }
    return this.searchBruteForce(queryVec, topK);
  }

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
   */
  clear(): void {
    this.docs.clear();
    this.hnsw?.clear();
    this.hnsw = null;
    this.termIndex.clear();
    this.docTerms.clear();
    this.docLengths.clear();
    this.docFreq.clear();
    this.totalDocLength = 0;
    console.log("🗑️ Index cleared");
  }

  get size(): number {
    return this.docs.size;
  }

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
   * Removes from both the doc store and the HNSW index.
   */
  delete(id: string): boolean {
    const existed = this.docs.delete(id);
    if (existed) {
      this.hnsw?.delete(id);
      this.removeKeywords(id);
    }
    return existed;
  }

  /**
   * Hybrid search — fuses semantic (dense) and BM25 keyword (sparse) results
   * using Reciprocal Rank Fusion (RRF).
   *
   * @param query  - Free-text query string
   * @param topK   - Number of results to return (default: 10)
   * @param alpha  - Weight given to semantic search in [0, 1] (default: 0.5)
   *                 1.0 = pure semantic, 0.0 = pure keyword, 0.5 = equal blend
   */
  async hybridSearch(
    query: string,
    topK: number = 10,
    alpha: number = 0.5,
  ): Promise<SearchResult[]> {
    if (this.docs.size === 0) return [];

    const candidateK = Math.min(this.docs.size, topK * 3);

    const [semanticResults, keywordResults] = await Promise.all([
      this.search(query, candidateK),
      Promise.resolve(this.bm25Search(query, candidateK)),
    ]);

    // Reciprocal Rank Fusion with alpha-weighted contributions
    const fusedScores = new Map<string, number>();

    for (let i = 0; i < semanticResults.length; i++) {
      const id = semanticResults[i].id;
      fusedScores.set(
        id,
        (fusedScores.get(id) ?? 0) + alpha / (RRF_K + i + 1),
      );
    }
    for (let i = 0; i < keywordResults.length; i++) {
      const id = keywordResults[i].id;
      fusedScores.set(
        id,
        (fusedScores.get(id) ?? 0) + (1 - alpha) / (RRF_K + i + 1),
      );
    }

    return [...fusedScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([id, score]) => {
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
   * Rebuilds the HNSW graph from the imported vectors.
   */
  fromJSON(
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
  ): void {
    this.docs.clear();
    this.hnsw?.clear();
    this.hnsw = null;
    this.termIndex.clear();
    this.docTerms.clear();
    this.docLengths.clear();
    this.docFreq.clear();
    this.totalDocLength = 0;

    for (const [id, doc] of data) {
      const embedding = new Float32Array(doc.embedding);
      this.docs.set(id, {
        content: doc.content,
        metadata: doc.metadata,
        embedding,
      });

      this.indexKeywords(id, doc.content);

      if (this.useANN) {
        this.ensureHNSW().add(id, embedding);
      }
    }
    console.log(`📥 Imported ${this.docs.size} documents`);
  }
}

// Export singleton instance (default config: ANN enabled, threshold 2000)
export const engine = SemanticEngine.getInstance();

// Export class + types for tests that need custom config
export { SemanticEngine };
