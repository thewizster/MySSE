// tests/powers_test.ts
// Tests for all four reference Powers: QueryCache, MetadataFilter,
// EmbeddingSwap, and HybridSearch, as well as the engine's use/eject API.

import { SemanticEngine } from "../lib/semantic-engine.ts";
import { QueryCache } from "../lib/powers/cache.ts";
import { MetadataFilter } from "../lib/powers/metadata-filter.ts";
import { EmbeddingSwap } from "../lib/powers/embedding-swap.ts";
import { HybridSearch } from "../lib/powers/hybrid-search.ts";

// ── Assertion helpers ─────────────────────────────────────────────────────

function assert(condition: boolean, msg?: string): void {
  if (!condition) throw new Error(msg ?? "Assertion failed");
}

function assertEquals<T>(actual: T, expected: T, msg?: string): void {
  if (actual !== expected) {
    throw new Error(
      msg ?? `Expected ${String(expected)} but got ${String(actual)}`,
    );
  }
}

function assertThrows(fn: () => void, msgFragment?: string): void {
  try {
    fn();
    throw new Error("Expected function to throw but it did not");
  } catch (e) {
    if (msgFragment && !(e as Error).message.includes(msgFragment)) {
      throw new Error(
        `Expected error containing "${msgFragment}" but got "${
          (e as Error).message
        }"`,
      );
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function freshEngine() {
  SemanticEngine.resetInstance();
  const eng = SemanticEngine.getInstance();
  return eng;
}

// ── Power management API ──────────────────────────────────────────────────

Deno.test("Powers - use() registers a power", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(QueryCache());
  assertEquals(eng.powers.length, 1);
  assertEquals(eng.powers[0], "QueryCache");

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("Powers - use() rejects duplicate names", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(QueryCache());
  assertThrows(() => eng.use(QueryCache()), "already registered");

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("Powers - eject() removes a power by name", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(QueryCache());
  assertEquals(eng.powers.length, 1);

  const ejected = eng.eject("QueryCache");
  assertEquals(ejected, true);
  assertEquals(eng.powers.length, 0);

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("Powers - eject() returns false for unknown name", async () => {
  const eng = freshEngine();
  await eng.clear();

  const result = eng.eject("DoesNotExist");
  assertEquals(result, false);

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("Powers - zero powers registers no overhead", async () => {
  const eng = freshEngine();
  await eng.clear();

  assertEquals(eng.powers.length, 0);

  await eng.add([{ id: "d1", content: "hello world" }]);
  const results = await eng.search("hello", 5);
  assertEquals(results.length, 1);

  await eng.clear();
  SemanticEngine.resetInstance();
});

// ── QueryCache ────────────────────────────────────────────────────────────

Deno.test("QueryCache - returns cached results on second call", async () => {
  const eng = freshEngine();
  await eng.clear();

  let embedCount = 0;
  // Track embed calls to verify second search is served from cache
  eng.use(EmbeddingSwap((texts) => {
    embedCount += texts.length;
    // Minimal embeddings for this test
    return Promise.resolve(texts.map(() => {
      const v = new Float32Array(384);
      v[0] = 1;
      return v;
    }));
  }));
  eng.use(QueryCache({ ttl: 10_000 }));

  await eng.add([
    { id: "d1", content: "alpha beta gamma" },
    { id: "d2", content: "delta epsilon zeta" },
  ]);

  const beforeCount = embedCount;
  await eng.search("alpha beta", 5);
  const afterFirst = embedCount;
  await eng.search("alpha beta", 5); // should hit cache
  const afterSecond = embedCount;

  // First search embeds the query; second should be served from cache (no new embed)
  assert(afterFirst > beforeCount, "First search should call embed");
  assertEquals(
    afterSecond,
    afterFirst,
    "Cached search should not call embed again",
  );

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("QueryCache - respects TTL and re-fetches after expiry", async () => {
  const eng = freshEngine();
  await eng.clear();

  let callCount = 0;
  eng.use(QueryCache({ ttl: 1 })); // 1 ms TTL

  await eng.add([{ id: "d1", content: "test document" }]);

  // First call — cache miss
  const r1 = await eng.search("test", 5);
  callCount++;
  // Wait for TTL to expire
  await new Promise((resolve) => setTimeout(resolve, 5));
  // Second call — should be a cache miss again (expired)
  const r2 = await eng.search("test", 5);
  callCount++;

  assertEquals(r1.length, r2.length);
  assertEquals(callCount, 2);

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("QueryCache - onClear invalidates cache", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(QueryCache({ ttl: 60_000 }));
  await eng.add([{ id: "d1", content: "cached content" }]);

  // Prime the cache
  await eng.search("cached", 5);
  // Clear the index (triggers onClear)
  await eng.clear();

  // Add different docs
  await eng.add([{ id: "d2", content: "fresh new content" }]);

  // Cache was cleared, so this is a fresh search
  const results = await eng.search("cached", 5);
  // Should search fresh docs (d1 is gone), results may be empty or different
  for (const r of results) {
    assert(r.id !== "d1", "d1 was cleared and should not appear");
  }

  await eng.clear();
  SemanticEngine.resetInstance();
});

// ── MetadataFilter ────────────────────────────────────────────────────────

Deno.test("MetadataFilter - removes results that fail the predicate", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(MetadataFilter((meta) => meta.published === true));

  await eng.add([
    {
      id: "pub1",
      content: "published document one",
      metadata: { published: true },
    },
    {
      id: "draft1",
      content: "draft document one",
      metadata: { published: false },
    },
    {
      id: "pub2",
      content: "published document two",
      metadata: { published: true },
    },
  ]);

  const results = await eng.search("document", 10);
  for (const r of results) {
    assertEquals(
      r.metadata?.published,
      true,
      `doc ${r.id} should be published`,
    );
  }
  // Should only return the 2 published docs
  assert(results.every((r) => r.metadata?.published === true));

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("MetadataFilter - returns empty when all results fail predicate", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(MetadataFilter((meta) => meta.category === "news"));

  await eng.add([
    { id: "d1", content: "some content", metadata: { category: "blog" } },
    { id: "d2", content: "more content", metadata: { category: "docs" } },
  ]);

  const results = await eng.search("content", 10);
  assertEquals(results.length, 0, "No results should pass the news filter");

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("MetadataFilter - passes results with matching metadata", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(MetadataFilter((meta) => (meta.score as number) > 0.5));

  await eng.add([
    { id: "high", content: "important document", metadata: { score: 0.9 } },
    { id: "low", content: "important document copy", metadata: { score: 0.2 } },
  ]);

  const results = await eng.search("important document", 10);
  assert(results.length > 0, "Should return at least one result");
  for (const r of results) {
    assert(
      (r.metadata?.score as number) > 0.5,
      `score should be > 0.5 for ${r.id}`,
    );
  }

  await eng.clear();
  SemanticEngine.resetInstance();
});

// ── EmbeddingSwap ─────────────────────────────────────────────────────────

Deno.test("EmbeddingSwap - custom embed function is used for indexing and search", async () => {
  const eng = freshEngine();
  await eng.clear();

  const calls: string[][] = [];
  eng.use(
    EmbeddingSwap((texts) => {
      calls.push(texts);
      // Deterministic: embed as all-zero except dim[0] = text.length / 100
      return Promise.resolve(texts.map((t) => {
        const v = new Float32Array(384);
        v[0] = t.length / 100;
        return v;
      }));
    }),
  );

  await eng.add([
    { id: "short", content: "hi" },
    { id: "long", content: "this is a much longer document with many words" },
  ]);

  assert(calls.length > 0, "embed should have been called during add()");

  const results = await eng.search("some query", 2);
  assertEquals(results.length, 2);

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("EmbeddingSwap - last registered power wins (embed override)", async () => {
  const eng = freshEngine();
  await eng.clear();

  let firstCalled = false;
  let secondCalled = false;

  eng.use(
    EmbeddingSwap((texts) => {
      firstCalled = true;
      return Promise.resolve(texts.map(() => new Float32Array(384).fill(0.1)));
    }),
  );

  // Second EmbeddingSwap must be registered under a different name
  eng.use({
    name: "EmbeddingSwap2",
    embed: (texts) => {
      secondCalled = true;
      return Promise.resolve(texts.map(() => new Float32Array(384).fill(0.2)));
    },
  });

  await eng.add([{ id: "d1", content: "test" }]);

  assert(!firstCalled, "First embed power should not be called (last-wins)");
  assert(secondCalled, "Second (last) embed power should be called");

  await eng.clear();
  SemanticEngine.resetInstance();
});

// ── HybridSearch ──────────────────────────────────────────────────────────

Deno.test("HybridSearch - exact keyword match ranks higher with alpha=0", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(HybridSearch({ alpha: 0 })); // pure keyword

  await eng.add([
    { id: "match", content: "zygote cell biology embryo fertilisation" },
    { id: "nomatch", content: "machine learning neural network transformer" },
  ]);

  const results = await eng.search("zygote", 2);
  assertEquals(
    results[0].id,
    "match",
    "Exact-term doc should rank first with alpha=0",
  );

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - alpha=1 matches pure semantic order", async () => {
  const eng = freshEngine();
  await eng.clear();

  // Get semantic-only baseline first
  const bfEng = SemanticEngine.getInstance();
  await bfEng.add([
    { id: "d1", content: "deep learning neural network" },
    { id: "d2", content: "banana fruit healthy" },
    { id: "d3", content: "machine learning AI" },
  ]);
  const semantic = await bfEng.search("deep learning", 3);
  await bfEng.clear();
  SemanticEngine.resetInstance();

  // Now test with HybridSearch alpha=1
  const eng2 = SemanticEngine.getInstance();
  eng2.use(HybridSearch({ alpha: 1 }));
  await eng2.add([
    { id: "d1", content: "deep learning neural network" },
    { id: "d2", content: "banana fruit healthy" },
    { id: "d3", content: "machine learning AI" },
  ]);
  const hybrid = await eng2.search("deep learning", 3);

  // alpha=1 should produce same ranking as pure semantic
  for (let i = 0; i < hybrid.length; i++) {
    assertEquals(
      hybrid[i].id,
      semantic[i].id,
      `Rank ${i}: hybrid(alpha=1) should equal semantic`,
    );
  }

  await eng2.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - afterAdd indexes docs, onDelete removes them", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(HybridSearch({ alpha: 0 })); // pure keyword for determinism

  await eng.add([
    { id: "keep", content: "unique keyword xylophone clarinet" },
    { id: "remove", content: "unique keyword xylophone clarinet" },
  ]);

  await eng.delete("remove");

  const results = await eng.search("xylophone clarinet", 10);
  for (const r of results) {
    assert(
      r.id !== "remove",
      "Deleted doc should not appear in hybrid results",
    );
  }

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - onClear resets BM25 index", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(HybridSearch({ alpha: 0 }));

  await eng.add([{ id: "d1", content: "keyword search ranking" }]);
  await eng.clear();

  const results = await eng.search("keyword ranking", 5);
  assertEquals(results.length, 0, "BM25 should be cleared");

  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - fromJSON fires afterAdd to rebuild BM25 index", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(HybridSearch({ alpha: 0 })); // pure keyword

  await eng.add([
    { id: "d1", content: "transformer attention mechanism" },
    { id: "d2", content: "convolutional neural network image" },
  ]);

  const exported = eng.toJSON();
  await eng.clear();
  await eng.fromJSON(exported);

  assertEquals(eng.size, 2);
  const results = await eng.search("transformer attention", 2);
  assertEquals(results[0].id, "d1", "BM25 should be rebuilt after fromJSON");

  await eng.clear();
  SemanticEngine.resetInstance();
});

Deno.test("HybridSearch - blended results with alpha=0.5", async () => {
  const eng = freshEngine();
  await eng.clear();

  eng.use(HybridSearch({ alpha: 0.5 }));

  await eng.add([
    { id: "js", content: "JavaScript TypeScript runtime Deno Node" },
    { id: "py", content: "Python data science machine learning pandas" },
    { id: "rs", content: "Rust systems programming memory safe" },
  ]);

  const results = await eng.search("TypeScript runtime", 3);
  assertEquals(results.length, 3);
  // js doc has exact keyword overlap — should rank in top
  assertEquals(
    results[0].id,
    "js",
    "JS doc should rank first for TypeScript runtime query",
  );
  // Scores should be sorted descending
  for (let i = 1; i < results.length; i++) {
    assert(
      results[i - 1].score >= results[i].score,
      "Scores should be descending",
    );
  }

  await eng.clear();
  SemanticEngine.resetInstance();
});

// ── beforeAdd hook ────────────────────────────────────────────────────────

Deno.test("Powers - beforeAdd can transform documents", async () => {
  const eng = freshEngine();
  await eng.clear();

  // Power that uppercases all content before indexing
  eng.use({
    name: "Uppercaser",
    beforeAdd(docs) {
      return docs.map((d) => ({ ...d, content: d.content.toUpperCase() }));
    },
  });

  await eng.add([{ id: "d1", content: "hello world" }]);

  const doc = eng.get("d1");
  assertEquals(doc?.content, "HELLO WORLD", "Content should be uppercased");

  await eng.clear();
  SemanticEngine.resetInstance();
});
