// tests/hybrid-search_test.ts
// Tests for HybridSearch: BM25 keyword index + RRF fusion with semantic search

import { SemanticEngine } from "../lib/semantic-engine.ts";

// Simple assertion helpers (no external deps)
function assert(condition: boolean, msg?: string): void {
  if (!condition) {
    throw new Error(msg ?? "Assertion failed");
  }
}

function assertEquals<T>(actual: T, expected: T, msg?: string): void {
  if (actual !== expected) {
    throw new Error(msg ?? `Expected ${expected} but got ${actual}`);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

function freshEngine(
  options?: Parameters<typeof SemanticEngine.getInstance>[0],
) {
  SemanticEngine.resetInstance();
  const eng = SemanticEngine.getInstance(options);
  eng.clear();
  return eng;
}

// ── Basic Functionality ───────────────────────────────────────────────────

Deno.test("HybridSearch - returns results from a small corpus", async () => {
  const eng = freshEngine();

  await eng.add([
    { id: "doc1", content: "Deno is a secure runtime for TypeScript" },
    { id: "doc2", content: "Python is excellent for data science" },
    { id: "doc3", content: "TypeScript adds static types to JavaScript" },
  ]);

  const results = await eng.hybridSearch("TypeScript runtime", 3);
  assertEquals(results.length, 3);
  assert(results[0].score > 0, "Top result should have a positive score");
  // doc1 and doc3 both contain "TypeScript"; doc1 also contains "runtime"
  const topIds = results.slice(0, 2).map((r) => r.id);
  assert(
    topIds.includes("doc1") || topIds.includes("doc3"),
    "TypeScript-bearing docs should rank in top-2",
  );

  eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - empty corpus returns empty array", async () => {
  const eng = freshEngine();

  const results = await eng.hybridSearch("anything", 10);
  assertEquals(results.length, 0);

  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - topK limit is respected", async () => {
  const eng = freshEngine();

  const docs = Array.from({ length: 10 }, (_, i) => ({
    id: `d${i}`,
    content: `document number ${i} about programming`,
  }));
  await eng.add(docs);

  const r3 = await eng.hybridSearch("programming", 3);
  assertEquals(r3.length, 3);

  const r1 = await eng.hybridSearch("programming", 1);
  assertEquals(r1.length, 1);

  eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - results are sorted by score descending", async () => {
  const eng = freshEngine();

  const docs = Array.from({ length: 15 }, (_, i) => ({
    id: `d${i}`,
    content: `word${i} topic${i % 5} category${i % 3}`,
  }));
  await eng.add(docs);

  const results = await eng.hybridSearch("topic1 word3", 10);
  for (let i = 1; i < results.length; i++) {
    assert(
      results[i - 1].score >= results[i].score,
      `Scores should be non-increasing: ${results[i - 1].score} >= ${
        results[i].score
      }`,
    );
  }

  eng.clear();
  SemanticEngine.resetInstance();
});

// ── alpha parameter ────────────────────────────────────────────────────────

Deno.test("HybridSearch - alpha=1.0 produces same order as pure semantic search", async () => {
  const eng = freshEngine();

  await eng.add([
    { id: "doc1", content: "machine learning neural networks deep learning" },
    { id: "doc2", content: "banana apple fruit healthy diet" },
    { id: "doc3", content: "artificial intelligence language models" },
  ]);

  const query = "deep learning AI";
  const hybrid = await eng.hybridSearch(query, 3, 1.0);
  const semantic = await eng.search(query, 3);

  // With alpha=1, hybrid should match the semantic ranking exactly
  for (let i = 0; i < hybrid.length; i++) {
    assertEquals(
      hybrid[i].id,
      semantic[i].id,
      `At rank ${i}: hybrid (alpha=1) should equal semantic order`,
    );
  }

  eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - alpha=0.0 ranks exact keyword matches highest", async () => {
  const eng = freshEngine();

  // "zygote" is a rare distinctive word — only doc1 contains it
  await eng.add([
    { id: "doc1", content: "zygote cell biology embryo fertilisation" },
    { id: "doc2", content: "machine learning neural networks" },
    { id: "doc3", content: "database query optimisation index" },
  ]);

  const results = await eng.hybridSearch("zygote", 3, 0.0);
  // The doc containing the exact term should rank first
  assertEquals(
    results[0].id,
    "doc1",
    "alpha=0 (pure keyword) should rank exact-match doc first",
  );

  eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - default alpha=0.5 blends both signals", async () => {
  const eng = freshEngine();

  await eng.add([
    { id: "doc1", content: "Deno is a JavaScript runtime" },
    { id: "doc2", content: "Node.js is also a JavaScript runtime environment" },
    { id: "doc3", content: "Python is used for data analysis" },
  ]);

  const results = await eng.hybridSearch("JavaScript runtime", 3);
  assertEquals(results.length, 3);
  // doc1 and doc2 both contain the exact words — they should rank above doc3
  const topTwo = results.slice(0, 2).map((r) => r.id);
  assert(
    topTwo.includes("doc1"),
    "doc1 should be in top-2 for JS runtime query",
  );
  assert(
    topTwo.includes("doc2"),
    "doc2 should be in top-2 for JS runtime query",
  );

  eng.clear();
  SemanticEngine.resetInstance();
});

// ── Index maintenance ─────────────────────────────────────────────────────

Deno.test("HybridSearch - deleted docs do not appear in results", async () => {
  const eng = freshEngine();

  await eng.add([
    { id: "keep", content: "programming languages TypeScript JavaScript" },
    { id: "remove", content: "programming languages TypeScript JavaScript" },
  ]);

  eng.delete("remove");
  assertEquals(eng.size, 1);

  const results = await eng.hybridSearch("TypeScript JavaScript", 10);
  for (const r of results) {
    assert(
      r.id !== "remove",
      "Deleted doc should not appear in hybrid results",
    );
  }

  eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - clear resets BM25 index", async () => {
  const eng = freshEngine();

  await eng.add([
    { id: "doc1", content: "search engine ranking relevance" },
    { id: "doc2", content: "vector database embedding similarity" },
  ]);

  eng.clear();
  assertEquals(eng.size, 0);

  const results = await eng.hybridSearch("ranking similarity", 5);
  assertEquals(results.length, 0);

  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - fromJSON rebuilds BM25 index", async () => {
  const eng = freshEngine();

  await eng.add([
    { id: "doc1", content: "retrieval augmented generation language model" },
    { id: "doc2", content: "convolutional neural network image recognition" },
    { id: "doc3", content: "transformer attention mechanism BERT GPT" },
  ]);

  const exported = eng.toJSON();
  eng.clear();

  eng.fromJSON(exported);
  assertEquals(eng.size, 3);

  // BM25 should work after import — "transformer" is in doc3 only
  const results = await eng.hybridSearch("transformer attention", 3, 0.0);
  assertEquals(
    results[0].id,
    "doc3",
    "BM25 should find transformer doc after fromJSON",
  );

  eng.clear();
  SemanticEngine.resetInstance();
});
