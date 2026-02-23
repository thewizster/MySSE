// lib/powers/hybrid-search.ts
// HybridSearch Power — maintains a BM25 inverted keyword index alongside the
// engine's dense vector index and fuses both ranked lists via Reciprocal Rank
// Fusion (RRF) in afterSearch.
//
// BM25 reference: Robertson & Zaragoza 2009
// RRF reference:  Cormack, Clarke & Buettcher 2009 (k=60 conventional default)

import type { Document, Power, SearchResult } from "../semantic-engine.ts";

// BM25 tuning constants (Okapi BM25)
const BM25_K1 = 1.5; // term-frequency saturation
const BM25_B = 0.75; // length normalisation

// RRF fusion constant
const RRF_K = 60;

// Candidate pool sizing for the BM25 search pass in afterSearch
const MIN_CANDIDATE_K = 10;
const CANDIDATE_MULTIPLIER = 3;

export interface HybridSearchOptions {
  /**
   * Semantic search weight in [0, 1] (default: 0.5).
   * - 1.0 = pure semantic (dense)
   * - 0.0 = pure keyword (BM25)
   * - 0.5 = equal blend (recommended default)
   */
  alpha?: number;
  /** BM25 term-frequency saturation parameter (default: 1.5) */
  k1?: number;
  /** BM25 length normalisation parameter (default: 0.75) */
  b?: number;
}

/**
 * HybridSearch Power — combines dense semantic retrieval with sparse BM25
 * keyword retrieval using Reciprocal Rank Fusion.
 *
 * Hooks used:
 * - `afterAdd`:    index new documents into the BM25 inverted index
 * - `afterSearch`: fuse semantic results with BM25 results via RRF
 * - `onDelete`:    remove a document from the BM25 index
 * - `onClear`:     wipe the BM25 index
 *
 * @example
 * ```ts
 * import { HybridSearch } from "./lib/powers/hybrid-search.ts";
 * engine.use(HybridSearch({ alpha: 0.5 }));
 * // All engine.search() calls now return RRF-fused results
 * ```
 */
export function HybridSearch(opts: HybridSearchOptions = {}): Power {
  const alpha = opts.alpha ?? 0.5;
  const k1 = opts.k1 ?? BM25_K1;
  const b = opts.b ?? BM25_B;

  // BM25 inverted index state
  const termIndex = new Map<string, Map<string, number>>(); // term → (docId → tf)
  const docTerms = new Map<string, Set<string>>(); // docId → unique terms
  const docLengths = new Map<string, number>(); // docId → token count
  const docFreq = new Map<string, number>(); // term → document frequency
  let totalDocLength = 0;

  // Stored doc content/metadata for BM25-only result hydration
  const storedDocs = new Map<
    string,
    { content: string; metadata: Record<string, unknown> }
  >();

  // ── BM25 helpers ──────────────────────────────────────────────────────

  function tokenize(text: string): string[] {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 1);
  }

  function indexDocuments(docs: Document[]): void {
    for (const doc of docs) {
      storedDocs.set(doc.id, {
        content: doc.content,
        metadata: doc.metadata ?? {},
      });
      const tokens = tokenize(doc.content);
      const termFreq = new Map<string, number>();
      for (const token of tokens) {
        termFreq.set(token, (termFreq.get(token) ?? 0) + 1);
      }
      docTerms.set(doc.id, new Set(termFreq.keys()));
      docLengths.set(doc.id, tokens.length);
      totalDocLength += tokens.length;
      for (const [term, freq] of termFreq) {
        if (!termIndex.has(term)) termIndex.set(term, new Map());
        termIndex.get(term)!.set(doc.id, freq);
        docFreq.set(term, (docFreq.get(term) ?? 0) + 1);
      }
    }
  }

  function removeDocument(id: string): void {
    storedDocs.delete(id);
    const terms = docTerms.get(id);
    if (!terms) return;
    totalDocLength -= docLengths.get(id) ?? 0;
    docLengths.delete(id);
    docTerms.delete(id);
    for (const term of terms) {
      const postings = termIndex.get(term);
      if (postings) {
        postings.delete(id);
        if (postings.size === 0) termIndex.delete(term);
      }
      const df = (docFreq.get(term) ?? 1) - 1;
      if (df <= 0) docFreq.delete(term);
      else docFreq.set(term, df);
    }
  }

  function bm25Search(
    query: string,
    topK: number,
  ): Array<{ id: string; score: number }> {
    const queryTerms = tokenize(query);
    if (queryTerms.length === 0) return [];

    const N = docLengths.size;
    const avgDl = N > 0 ? Math.max(1, totalDocLength / N) : 1;
    const scores = new Map<string, number>();

    for (const term of queryTerms) {
      const df = docFreq.get(term) ?? 0;
      if (df === 0) continue;
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
      const postings = termIndex.get(term)!;
      for (const [id, tf] of postings) {
        const dl = docLengths.get(id) ?? 0;
        const norm = (tf * (k1 + 1)) /
          (tf + k1 * (1 - b + b * (dl / avgDl)));
        scores.set(id, (scores.get(id) ?? 0) + idf * norm);
      }
    }

    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([id, score]) => ({ id, score }));
  }

  // ── Power implementation ──────────────────────────────────────────────

  return {
    name: "HybridSearch",

    afterAdd(docs) {
      indexDocuments(docs);
    },

    onDelete(id) {
      removeDocument(id);
    },

    onClear() {
      termIndex.clear();
      docTerms.clear();
      docLengths.clear();
      docFreq.clear();
      storedDocs.clear();
      totalDocLength = 0;
    },

    afterSearch(results, query) {
      // Use a candidate pool larger than topK for better recall
      const topK = results.length;
      const candidateK = Math.max(topK, MIN_CANDIDATE_K) * CANDIDATE_MULTIPLIER;
      const kwResults = bm25Search(query, candidateK);

      // RRF fusion with alpha-weighted semantic contribution
      const fusedScores = new Map<string, number>();
      for (let i = 0; i < results.length; i++) {
        const id = results[i].id;
        fusedScores.set(
          id,
          (fusedScores.get(id) ?? 0) + alpha / (RRF_K + i + 1),
        );
      }
      for (let i = 0; i < kwResults.length; i++) {
        const id = kwResults[i].id;
        fusedScores.set(
          id,
          (fusedScores.get(id) ?? 0) + (1 - alpha) / (RRF_K + i + 1),
        );
      }

      // Hydrate the fused ranking: prefer the richer semantic result object,
      // but include BM25-only results from storedDocs if they aren't present.
      const semanticMap = new Map(results.map((r) => [r.id, r]));

      return [...fusedScores.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, topK)
        .flatMap(([id, score]): SearchResult[] => {
          if (semanticMap.has(id)) {
            return [{ ...semanticMap.get(id)!, score }];
          }
          const stored = storedDocs.get(id);
          if (!stored) return [];
          return [{
            id,
            content: stored.content,
            metadata: stored.metadata,
            score,
          }];
        });
    },
  };
}
