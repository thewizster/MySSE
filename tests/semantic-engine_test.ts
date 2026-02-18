// tests/semantic-engine_test.ts
import { engine } from "../lib/semantic-engine.ts";

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

Deno.test("SemanticEngine - add and search documents", async () => {
  // Clear any existing documents
  engine.clear();
  assertEquals(engine.size, 0);

  // Add test documents
  await engine.add([
    {
      id: "doc1",
      content: "Deno is a secure runtime for JavaScript and TypeScript",
    },
    { id: "doc2", content: "Fresh is a modern web framework for Deno" },
    {
      id: "doc3",
      content: "Python is a programming language used for data science",
    },
  ]);

  assertEquals(engine.size, 3);

  // Search for relevant documents
  const results = await engine.search("secure typescript runtime", 2);
  assertEquals(results.length, 2);
  assertExists(results[0].score);
  assertExists(results[0].id);
  assertExists(results[0].content);
});

Deno.test("SemanticEngine - get document by ID", async () => {
  engine.clear();

  await engine.add([
    {
      id: "test-doc",
      content: "Test document content",
      metadata: { source: "test" },
    },
  ]);

  const doc = engine.get("test-doc");
  assertExists(doc);
  assertEquals(doc!.id, "test-doc");
  assertEquals(doc!.content, "Test document content");
  assertEquals(doc!.metadata?.source, "test");

  const nonExistent = engine.get("non-existent");
  assertEquals(nonExistent, undefined);
});

Deno.test("SemanticEngine - delete document", async () => {
  engine.clear();

  await engine.add([
    { id: "to-delete", content: "This will be deleted" },
    { id: "to-keep", content: "This will be kept" },
  ]);

  assertEquals(engine.size, 2);

  const deleted = engine.delete("to-delete");
  assertEquals(deleted, true);
  assertEquals(engine.size, 1);

  const notDeleted = engine.delete("non-existent");
  assertEquals(notDeleted, false);
});

Deno.test("SemanticEngine - clear all documents", async () => {
  engine.clear();

  await engine.add([
    { id: "doc1", content: "First document" },
    { id: "doc2", content: "Second document" },
  ]);

  assertEquals(engine.size, 2);

  engine.clear();
  assertEquals(engine.size, 0);
});

Deno.test("SemanticEngine - export and import JSON", async () => {
  engine.clear();

  await engine.add([
    {
      id: "export-doc",
      content: "Document for export",
      metadata: { key: "value" },
    },
  ]);

  const exported = engine.toJSON();
  assertEquals(exported.length, 1);
  assertEquals(exported[0][0], "export-doc");

  engine.clear();
  assertEquals(engine.size, 0);

  engine.fromJSON(exported);
  assertEquals(engine.size, 1);

  const doc = engine.get("export-doc");
  assertExists(doc);
  assertEquals(doc!.content, "Document for export");
});

Deno.test("SemanticEngine - empty search returns empty array", async () => {
  engine.clear();

  const results = await engine.search("anything", 10);
  assertEquals(results.length, 0);
});

Deno.test("SemanticEngine - search respects topK limit", async () => {
  engine.clear();

  await engine.add([
    { id: "doc1", content: "First document about programming" },
    { id: "doc2", content: "Second document about coding" },
    { id: "doc3", content: "Third document about development" },
    { id: "doc4", content: "Fourth document about software" },
    { id: "doc5", content: "Fifth document about engineering" },
  ]);

  const results = await engine.search("programming", 3);
  assertEquals(results.length, 3);
});
