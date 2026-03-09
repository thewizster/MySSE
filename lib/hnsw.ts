/**
 * Pure TypeScript HNSW (Hierarchical Navigable Small World) index for
 * approximate nearest-neighbor search using cosine distance on unit vectors.
 *
 * Implements Malkov & Yashunin 2016, Algorithms 1–5.
 *
 * @module
 */

// lib/hnsw.ts  —  pure TS HNSW (cosine distance on unit vectors)
// Follows Malkov & Yashunin 2016, Algorithms 1–5

/** A dense floating-point vector (alias for `Float32Array`). */
export type Vector = Float32Array;
type Neighbor = { id: string; dist: number };

const DEFAULT_M = 16; // max connections per layer (paper §4.1, suggests 5–48)
const EF_CONSTRUCTION = 40; // higher = better recall, slower build

function cosineDistance(a: Vector, b: Vector): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return 1 - dot; // distance = 1 − similarity (smaller = closer)
}

/**
 * Hierarchical Navigable Small World (HNSW) graph index for fast
 * approximate nearest-neighbor search over cosine distance.
 *
 * @example
 * ```ts
 * import { HNSW } from "@wxt/my-search-engine/hnsw";
 *
 * const index = new HNSW(384);
 * index.add("doc1", embedding);
 * const results = index.search(queryVec, 10);
 * ```
 */
export class HNSW {
  /** Per-node neighbor sets, indexed by layer. */
  private layers: Map<string, Set<string>[]> = new Map();
  /** Stored vectors keyed by ID. */
  private vectors: Map<string, Vector> = new Map();
  /** Assigned HNSW level for each node. */
  private nodeLevel: Map<string, number> = new Map();
  /** Current graph entry point (highest-level node). */
  private enterPoint: string | null = null;
  /** Highest occupied layer in the graph. */
  private maxLayer = 0;

  /** Max neighbor connections for layers >= 1. */
  private readonly mMax: number;
  /** Max neighbor connections for layer 0 (2 * M). */
  private readonly mMax0: number;
  /** Level generation multiplier 1/ln(M). */
  private readonly mL: number;

  /**
   * Create a new HNSW index.
   * @param dim Dimensionality of the vectors.
   * @param m Maximum number of connections per layer (default: 16).
   */
  constructor(private dim: number, private m: number = DEFAULT_M) {
    this.mMax = m;
    this.mMax0 = 2 * m;
    this.mL = 1 / Math.log(m);
  }

  /** Draw a random level for a new node (Paper Algorithm 1, line 4). */
  private randomLevel(): number {
    return Math.floor(-Math.log(Math.random()) * this.mL);
  }

  /** Select the `k` closest candidates (Algorithm 3: SELECT-NEIGHBORS-SIMPLE). */
  private selectNeighborsSimple(
    _query: Vector,
    candidates: Neighbor[],
    k: number,
  ): string[] {
    candidates.sort((a, b) => a.dist - b.dist);
    return candidates.slice(0, k).map((c) => c.id);
  }

  /** Greedy beam search within a single graph layer (Algorithm 2: SEARCH-LAYER). */
  private searchLayer(
    q: Vector,
    ep: string[],
    ef: number,
    layer: number,
  ): Neighbor[] {
    const visited = new Set<string>(ep);
    const found: Neighbor[] = ep.map((id) => ({
      id,
      dist: cosineDistance(q, this.vectors.get(id)!),
    }));
    const candidates: Neighbor[] = [...found];

    candidates.sort((a, b) => a.dist - b.dist);
    found.sort((a, b) => a.dist - b.dist);

    while (candidates.length > 0) {
      const closest = candidates.shift()!;
      const farthestFound = found[found.length - 1];
      if (closest.dist > farthestFound.dist) break;

      const neighbors = this.layers.get(closest.id)?.[layer];
      if (!neighbors) continue;

      for (const nid of neighbors) {
        if (visited.has(nid)) continue;
        visited.add(nid);

        const dist = cosineDistance(q, this.vectors.get(nid)!);
        const farthest = found[found.length - 1];
        if (dist < farthest.dist || found.length < ef) {
          found.push({ id: nid, dist });
          candidates.push({ id: nid, dist });
          found.sort((a, b) => a.dist - b.dist);
          candidates.sort((a, b) => a.dist - b.dist);
          if (found.length > ef) found.pop();
        }
      }
    }

    return found;
  }

  /** Shrink a node's neighbor list if it exceeds M_max (Paper Alg 1, lines 12–17). */
  private shrinkNeighbors(nodeId: string, layer: number): void {
    const maxConn = layer === 0 ? this.mMax0 : this.mMax;
    const neighbors = this.layers.get(nodeId)?.[layer];
    if (!neighbors || neighbors.size <= maxConn) return;

    const nodeVec = this.vectors.get(nodeId)!;
    const scored: Neighbor[] = [];
    for (const nid of neighbors) {
      scored.push({
        id: nid,
        dist: cosineDistance(nodeVec, this.vectors.get(nid)!),
      });
    }
    const kept = this.selectNeighborsSimple(nodeVec, scored, maxConn);
    neighbors.clear();
    for (const id of kept) neighbors.add(id);
  }

  /**
   * Insert a vector into the index.
   * @param id Unique identifier for the vector.
   * @param vector Float32Array of length `dim` (should be unit-normalised).
   * @throws If `id` already exists or `vector.length !== dim`.
   */
  add(id: string, vector: Vector): void {
    if (vector.length !== this.dim) throw new Error("Dimension mismatch");
    if (this.vectors.has(id)) throw new Error("ID already exists");

    this.vectors.set(id, vector);

    const level = this.randomLevel();
    this.nodeLevel.set(id, level);

    // Initialize neighbor sets for layers 0..level
    const layerSets: Set<string>[] = [];
    for (let i = 0; i <= level; i++) layerSets.push(new Set());
    this.layers.set(id, layerSets);

    // First element — just set as entry point and return
    if (this.enterPoint === null) {
      this.enterPoint = id;
      this.maxLayer = level;
      return;
    }

    let ep = [this.enterPoint];
    const L = this.maxLayer;

    // Phase 1: Greedy walk from top layer down to level+1 (ef = 1, no connections)
    for (let l = L; l > level; l--) {
      const nearest = this.searchLayer(vector, ep, 1, l);
      ep = [nearest[0].id];
    }

    // Phase 2: Search + connect at layers min(level, L) down to 0
    for (let l = Math.min(level, L); l >= 0; l--) {
      const nearest = this.searchLayer(vector, ep, EF_CONSTRUCTION, l);
      const maxConn = l === 0 ? this.mMax0 : this.mMax;
      const selected = this.selectNeighborsSimple(vector, nearest, maxConn);

      // Connect bidirectionally
      for (const nid of selected) {
        this.layers.get(id)![l].add(nid);
        const nLayers = this.layers.get(nid)!;
        if (nLayers[l]) {
          nLayers[l].add(id);
          this.shrinkNeighbors(nid, l); // prune if over capacity
        }
      }

      ep = selected;
    }

    // Update entry point if new element is on a higher level
    if (level > L) {
      this.enterPoint = id;
      this.maxLayer = level;
    }
  }

  /**
   * Find the `k` nearest neighbors to `query`.
   * @param query Float32Array query vector (same dimensionality as indexed vectors).
   * @param k Number of results to return (default: 10).
   * @param efSearch Beam width — higher values improve recall at the cost of latency (default: 64).
   * @returns Array of `{ id, score }` sorted by descending cosine similarity.
   */
  search(
    query: Vector,
    k = 10,
    efSearch = 64,
  ): { id: string; score: number }[] {
    if (!this.enterPoint) return [];

    let ep = [this.enterPoint];

    // Layers L down to 1: greedy search with ef = 1
    for (let l = this.maxLayer; l >= 1; l--) {
      const nearest = this.searchLayer(query, ep, 1, l);
      ep = [nearest[0].id];
    }

    // Layer 0: search with ef = max(efSearch, k)
    const ef0 = Math.max(efSearch, k);
    const results = this.searchLayer(query, ep, ef0, 0);

    return results
      .slice(0, k)
      .map((n) => ({ id: n.id, score: 1 - n.dist }));
  }

  /**
   * Remove a vector from the index.
   * @param id The identifier of the vector to remove.
   * @returns `true` if the vector was found and removed, `false` otherwise.
   */
  delete(id: string): boolean {
    if (!this.vectors.has(id)) return false;

    const level = this.nodeLevel.get(id)!;

    // Remove bidirectional connections
    const nodeLayers = this.layers.get(id)!;
    for (let l = 0; l <= level; l++) {
      const neighbors = nodeLayers[l];
      if (!neighbors) continue;
      for (const nid of neighbors) {
        this.layers.get(nid)?.[l]?.delete(id);
      }
    }

    this.layers.delete(id);
    this.vectors.delete(id);
    this.nodeLevel.delete(id);

    // Update entry point if we deleted it
    if (this.enterPoint === id) {
      if (this.vectors.size === 0) {
        this.enterPoint = null;
        this.maxLayer = 0;
      } else {
        // Pick the node with the highest level as new entry point
        let bestId: string | null = null;
        let bestLevel = -1;
        for (const [nid, nl] of this.nodeLevel) {
          if (nl > bestLevel) {
            bestLevel = nl;
            bestId = nid;
          }
        }
        this.enterPoint = bestId;
        this.maxLayer = bestLevel;
      }
    }

    return true;
  }

  /** The number of vectors currently in the index. */
  get size(): number {
    return this.vectors.size;
  }

  /** Remove all vectors and reset the index to its initial empty state. */
  clear(): void {
    this.layers.clear();
    this.vectors.clear();
    this.nodeLevel.clear();
    this.enterPoint = null;
    this.maxLayer = 0;
  }
}
