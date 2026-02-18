// lib/semantic-engine.ts
// In-RAM Semantic Search Engine
// 100% memory-based, no external DB, no disk I/O during queries.

export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResult extends Document {
  score: number;
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
// Uncomment and use when npm registry is available:
/*
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

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
      this.model = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        { device: "auto", dtype: "fp32" }
      );
      console.log("✅ Model loaded successfully");
    })();
    await this.modelLoading;
  }
}
*/

class SemanticEngine {
  private static instance: SemanticEngine;
  private model: EmbeddingModel;
  private docs: Map<string, StoredDocument> = new Map();

  private constructor() {
    // Use simple embedding model (swap with TransformersJsEmbedding for production)
    this.model = new SimpleEmbeddingModel();
  }

  static getInstance(): SemanticEngine {
    if (!SemanticEngine.instance) {
      SemanticEngine.instance = new SemanticEngine();
    }
    return SemanticEngine.instance;
  }

  /**
   * Add documents to the in-memory index
   * Documents are embedded and stored with pre-normalized vectors for fast cosine similarity
   */
  async add(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    const texts = documents.map((d) => d.content);
    const embeddings = await this.model.embed(texts);

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      this.docs.set(doc.id, {
        content: doc.content,
        metadata: doc.metadata ?? {},
        embedding: embeddings[i],
      });
    }

    console.log(
      `✅ Added ${documents.length} documents. Total: ${this.docs.size}`,
    );
  }

  /**
   * Semantic search using dot product (equivalent to cosine similarity for normalized vectors)
   * Brute-force search is extremely fast for <50k documents
   */
  async search(query: string, topK: number = 10): Promise<SearchResult[]> {
    if (this.docs.size === 0) {
      return [];
    }

    const [queryVec] = await this.model.embed([query]);
    const results: SearchResult[] = [];

    // Compute dot product (= cosine similarity for normalized vectors)
    // This is extremely cache-friendly and fast on modern CPUs
    for (const [id, { content, metadata, embedding }] of this.docs) {
      let dot = 0;
      for (let i = 0; i < embedding.length; i++) {
        dot += embedding[i] * queryVec[i];
      }
      results.push({ id, content, metadata, score: dot });
    }

    // Sort by score descending and return top K
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /**
   * Clear all documents from the index
   */
  clear(): void {
    this.docs.clear();
    console.log("🗑️ Index cleared");
  }

  /**
   * Get the number of documents in the index
   */
  get size(): number {
    return this.docs.size;
  }

  /**
   * Get a document by ID
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
   * Delete a document by ID
   */
  delete(id: string): boolean {
    return this.docs.delete(id);
  }

  /**
   * Export the index for persistence (optional)
   * Can be serialized to JSON and stored in Deno KV or file
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
   * Import a previously exported index
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
    for (const [id, doc] of data) {
      this.docs.set(id, {
        content: doc.content,
        metadata: doc.metadata,
        embedding: new Float32Array(doc.embedding),
      });
    }
    console.log(`📥 Imported ${this.docs.size} documents`);
  }
}

// Export singleton instance
export const engine = SemanticEngine.getInstance();
