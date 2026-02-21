// tests/hnsw_test.ts
import HNSW from "../lib/hnsw.ts";

// Simple assertion helpers (no external deps)
function assertEquals<T>(actual: T, expected: T, msg?: string): void {
  if (actual !== expected) {
    throw new Error(msg ?? `Expected ${expected} but got ${actual}`);
  }
}

function assertExists<T>(
  value: T,
  msg?: string,
): asserts value is NonNullable<T> {
  if (value === null || value === undefined) {
    throw new Error(msg ?? `Expected value to exist but got ${value}`);
  }
}

function assertThrows(fn: () => void, msg?: string): void {
  try {
    fn();
    throw new Error(msg ?? "Expected function to throw but it did not");
  } catch (e) {
    if (
      (e as Error).message ===
        (msg ?? "Expected function to throw but it did not")
    ) {
      throw e;
    }
  }
}

function assert(condition: boolean, msg?: string): void {
  if (!condition) {
    throw new Error(msg ?? "Assertion failed");
  }
}

// Helpers
const DIM = 8;

/** Create a unit-normalized random vector */
function randomVec(dim = DIM): Float32Array {
  const v = new Float32Array(dim);
  let mag = 0;
  for (let i = 0; i < dim; i++) {
    v[i] = Math.random() - 0.5;
    mag += v[i] * v[i];
  }
  mag = Math.sqrt(mag);
  for (let i = 0; i < dim; i++) v[i] /= mag;
  return v;
}

/** Create a unit vector pointing mostly in dimension `axis` */
function axisVec(axis: number, dim = DIM): Float32Array {
  const v = new Float32Array(dim);
  v[axis] = 1;
  return v;
}

// ──────────────────── Basic Operations ────────────────────

Deno.test("HNSW - add single element and search", () => {
  const index = new HNSW(DIM);
  const vec = randomVec();
  index.add("a", vec);

  assertEquals(index.size, 1);

  const results = index.search(vec, 1);
  assertEquals(results.length, 1);
  assertEquals(results[0].id, "a");
  assert(results[0].score > 0.99, "Self-search should return ~1.0 similarity");
});

Deno.test("HNSW - size tracks insertions", () => {
  const index = new HNSW(DIM);
  assertEquals(index.size, 0);

  index.add("a", randomVec());
  assertEquals(index.size, 1);

  index.add("b", randomVec());
  assertEquals(index.size, 2);

  index.add("c", randomVec());
  assertEquals(index.size, 3);
});

Deno.test("HNSW - search empty index returns empty", () => {
  const index = new HNSW(DIM);
  const results = index.search(randomVec(), 5);
  assertEquals(results.length, 0);
});

Deno.test("HNSW - dimension mismatch throws", () => {
  const index = new HNSW(DIM);
  assertThrows(() => index.add("bad", new Float32Array(DIM + 1)));
});

Deno.test("HNSW - duplicate ID throws", () => {
  const index = new HNSW(DIM);
  index.add("dup", randomVec());
  assertThrows(() => index.add("dup", randomVec()));
});

// ──────────────────── Search Quality ────────────────────

Deno.test("HNSW - nearest neighbor is correct for small set", () => {
  const index = new HNSW(DIM);

  // Insert orthogonal axis vectors — maximally separated
  for (let i = 0; i < DIM; i++) {
    index.add(`axis-${i}`, axisVec(i));
  }

  // Query with each axis vector; nearest should be itself
  for (let i = 0; i < DIM; i++) {
    const results = index.search(axisVec(i), 1);
    assertEquals(
      results[0].id,
      `axis-${i}`,
      `Axis ${i} should be its own nearest neighbor`,
    );
    assert(results[0].score > 0.99, "Score should be ~1.0 for exact match");
  }
});

Deno.test("HNSW - search returns results sorted by score descending", () => {
  const index = new HNSW(DIM);

  for (let i = 0; i < 50; i++) {
    index.add(`v${i}`, randomVec());
  }

  const results = index.search(randomVec(), 10);
  for (let i = 1; i < results.length; i++) {
    assert(
      results[i - 1].score >= results[i].score,
      `Results should be sorted descending: ${results[i - 1].score} >= ${
        results[i].score
      }`,
    );
  }
});

Deno.test("HNSW - search respects k limit", () => {
  const index = new HNSW(DIM);

  for (let i = 0; i < 20; i++) {
    index.add(`v${i}`, randomVec());
  }

  const r3 = index.search(randomVec(), 3);
  assertEquals(r3.length, 3);

  const r1 = index.search(randomVec(), 1);
  assertEquals(r1.length, 1);

  const r10 = index.search(randomVec(), 10);
  assertEquals(r10.length, 10);
});

Deno.test("HNSW - k larger than index size returns all", () => {
  const index = new HNSW(DIM);

  index.add("a", randomVec());
  index.add("b", randomVec());
  index.add("c", randomVec());

  const results = index.search(randomVec(), 100);
  assertEquals(results.length, 3);
});

Deno.test("HNSW - cosine similarity: similar vectors score higher", () => {
  const index = new HNSW(DIM);

  // Create a target vector and a similar one (small perturbation)
  const target = axisVec(0);
  const similar = new Float32Array(DIM);
  similar[0] = 0.95;
  similar[1] = 0.05;
  // Normalize
  let mag = 0;
  for (let i = 0; i < DIM; i++) mag += similar[i] * similar[i];
  mag = Math.sqrt(mag);
  for (let i = 0; i < DIM; i++) similar[i] /= mag;

  const dissimilar = axisVec(DIM - 1); // orthogonal

  index.add("similar", similar);
  index.add("dissimilar", dissimilar);

  const results = index.search(target, 2);
  assertEquals(results[0].id, "similar", "Similar vector should rank first");
  assert(
    results[0].score > results[1].score,
    "Similar vector should have higher score than orthogonal one",
  );
});

// ──────────────────── Delete ────────────────────

Deno.test("HNSW - delete existing element", () => {
  const index = new HNSW(DIM);

  index.add("a", randomVec());
  index.add("b", randomVec());
  index.add("c", randomVec());

  assertEquals(index.size, 3);

  const deleted = index.delete("b");
  assertEquals(deleted, true);
  assertEquals(index.size, 2);

  // Deleted element should not appear in results
  const results = index.search(randomVec(), 10);
  for (const r of results) {
    assert(r.id !== "b", "Deleted element should not appear in search results");
  }
});

Deno.test("HNSW - delete non-existent returns false", () => {
  const index = new HNSW(DIM);
  index.add("a", randomVec());

  const deleted = index.delete("non-existent");
  assertEquals(deleted, false);
  assertEquals(index.size, 1);
});

Deno.test("HNSW - delete all elements empties index", () => {
  const index = new HNSW(DIM);

  index.add("a", randomVec());
  index.add("b", randomVec());

  index.delete("a");
  index.delete("b");

  assertEquals(index.size, 0);

  const results = index.search(randomVec(), 5);
  assertEquals(results.length, 0);
});

Deno.test("HNSW - delete then add back same ID", () => {
  const index = new HNSW(DIM);
  const vec = randomVec();

  index.add("reuse", vec);
  index.delete("reuse");
  assertEquals(index.size, 0);

  // Re-insert with same ID should not throw
  index.add("reuse", vec);
  assertEquals(index.size, 1);

  const results = index.search(vec, 1);
  assertEquals(results[0].id, "reuse");
});

Deno.test("HNSW - search still works after deleting entry point", () => {
  const index = new HNSW(DIM);

  // Insert enough elements so there's a graph to navigate
  const vecs: Float32Array[] = [];
  for (let i = 0; i < 20; i++) {
    const v = randomVec();
    vecs.push(v);
    index.add(`v${i}`, v);
  }

  // Delete the first element (likely the entry point since it was inserted first)
  index.delete("v0");
  assertEquals(index.size, 19);

  // Search should still function
  const results = index.search(vecs[1], 5);
  assert(
    results.length > 0,
    "Search should return results after entry point deletion",
  );
});

// ──────────────────── Clear ────────────────────

Deno.test("HNSW - clear resets everything", () => {
  const index = new HNSW(DIM);

  for (let i = 0; i < 10; i++) {
    index.add(`v${i}`, randomVec());
  }

  assertEquals(index.size, 10);

  index.clear();
  assertEquals(index.size, 0);

  const results = index.search(randomVec(), 5);
  assertEquals(results.length, 0);
});

Deno.test("HNSW - can add elements after clear", () => {
  const index = new HNSW(DIM);

  index.add("before", randomVec());
  index.clear();

  index.add("after", randomVec());
  assertEquals(index.size, 1);

  const results = index.search(randomVec(), 5);
  assertEquals(results.length, 1);
  assertEquals(results[0].id, "after");
});

// ──────────────────── Scale ────────────────────

Deno.test("HNSW - handles 500 vectors with reasonable recall", () => {
  const index = new HNSW(DIM);

  const vectors: Map<string, Float32Array> = new Map();
  for (let i = 0; i < 500; i++) {
    const v = randomVec();
    vectors.set(`v${i}`, v);
    index.add(`v${i}`, v);
  }

  assertEquals(index.size, 500);

  // Pick a random query and brute-force the true nearest neighbor
  const query = randomVec();
  let bestId = "";
  let bestDot = -Infinity;
  for (const [id, v] of vectors) {
    let dot = 0;
    for (let i = 0; i < DIM; i++) dot += v[i] * query[i];
    if (dot > bestDot) {
      bestDot = dot;
      bestId = id;
    }
  }

  // HNSW should find the true nearest neighbor in top-5
  const results = index.search(query, 5);
  const foundIds = results.map((r) => r.id);
  assert(
    foundIds.includes(bestId),
    `True nearest neighbor '${bestId}' should be in top-5 HNSW results`,
  );
});

Deno.test("HNSW - scores are valid cosine similarities in [-1, 1]", () => {
  const index = new HNSW(DIM);

  for (let i = 0; i < 30; i++) {
    index.add(`v${i}`, randomVec());
  }

  const results = index.search(randomVec(), 10);
  for (const r of results) {
    assert(
      r.score >= -1 && r.score <= 1,
      `Score ${r.score} should be in [-1, 1]`,
    );
  }
});
