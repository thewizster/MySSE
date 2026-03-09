# MySSE - My Semantic Search Engine

[![JSR](https://jsr.io/badges/@wxt/my-search-engine)](https://jsr.io/@wxt/my-search-engine)
[![JSR Score](https://jsr.io/badges/@wxt/my-search-engine/score)](https://jsr.io/@wxt/my-search-engine)

A fully in-RAM semantic search engine built with **Deno** in pure TypeScript —
minimalist, elegant, blazing-fast, and built for 2026+.

![MySSE Home Page](https://github.com/user-attachments/assets/0cc9ec0e-ac94-4cd3-a0ea-11f95c89f515)

## ✨ Features

- **100% In-Memory**: No external DB, no disk I/O during queries
- **HNSW Indexing**: Pure-TypeScript implementation of the 2016 Malkov &
  Yashunin paper — automatic approximate nearest-neighbor search above 2 000
  documents, exact brute-force below
- **Adaptive Search**: Brute-force (recall = 100%) when the index is small; HNSW
  (recall@10 ≥ 92%, 4–6× faster) when it grows — fully automatic, no config
  needed
- **AI-Ready**: Pluggable embedding interface (deterministic hash-based
  included, Transformers.js ready)
- **Powers Extensibility**: Plugin system with lifecycle hooks — add caching,
  hybrid BM25+semantic search, metadata filtering, or custom embedding models
  without touching core code
- **Type-Safe**: Pure TypeScript end-to-end, zero external dependencies
- **Secure**: Deno's secure-by-default runtime
- **Minimal**: ~720 LOC core engine + ~310 LOC Powers, JSON API

## 🚀 Quick Start

### Install from JSR

```ts
// deno.json
{
  "imports": {
    "@wxt/my-search-engine": "jsr:@wxt/my-search-engine@^0.1.0"
  }
}
```

```ts
import { SemanticEngine, HybridSearch } from "@wxt/my-search-engine";

const engine = SemanticEngine.getInstance();
engine.use(HybridSearch());

await engine.add([
  { id: "1", content: "Deno is a secure runtime for JS and TS" },
  { id: "2", content: "Fresh is a modern web framework for Deno" },
]);

const results = await engine.search("secure typescript runtime");
console.log(results);
```

### Run the Demo Server

### Prerequisites

- [Deno](https://deno.land/) 2.0 or later

### Installation

```bash
# Clone the repository
git clone https://github.com/thewizster/MySSE.git
cd MySSE

# Start the development server
deno task dev
```

The server will start at `http://localhost:8000`.

## 📖 API Reference

### Add Documents

```bash
POST /api/add
Content-Type: application/json

# Single document
{"id": "doc1", "content": "Your document text", "metadata": {"source": "example"}}

# Multiple documents
[
  {"id": "doc1", "content": "First document text"},
  {"id": "doc2", "content": "Second document text"}
]
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/add \
  -H "Content-Type: application/json" \
  -d '[
    {"id":"1", "content":"Deno is a secure runtime for JavaScript and TypeScript", "metadata":{"source":"docs"}},
    {"id":"2", "content":"Fresh is a modern web framework for Deno", "metadata":{"source":"docs"}},
    {"id":"3", "content":"Transformers.js runs ML models in the browser", "metadata":{"source":"docs"}}
  ]'
```

### Search Documents

```bash
GET /api/search?q=<query>&k=<top_k>
```

**Parameters:**

- `q` (required): The search query
- `k` (optional): Number of results to return (default: 10, max: 100)

**Example:**

```bash
curl "http://localhost:8000/api/search?q=secure+typescript+runtime&k=5"
```

![Search Results](https://github.com/user-attachments/assets/3ba82fe8-bd61-481a-ab7a-759f38c6988f)

### Check Status

```bash
GET /api/status
```

Returns the current engine status and document count.

### Clear Index

```bash
DELETE /api/clear
```

Removes all documents from the index.

## 🏗️ Architecture

```
MySSE/
├── lib/
│   ├── hnsw.ts               # HNSW approximate nearest-neighbor index (~210 LOC)
│   ├── semantic-engine.ts     # Core semantic search engine (~510 LOC)
│   └── powers/                # Extensibility plugins
│       ├── cache.ts           # QueryCache — search result caching
│       ├── embedding-swap.ts  # EmbeddingSwap — hot-swap embedding models
│       ├── hybrid-search.ts   # HybridSearch — BM25 + semantic RRF fusion
│       └── metadata-filter.ts # MetadataFilter — filter results by metadata
├── tests/
│   ├── hnsw_test.ts                   # HNSW unit tests (19 tests)
│   ├── semantic-engine_test.ts        # Engine unit tests (7 tests)
│   ├── semantic-engine-ann_test.ts    # ANN integration tests (8 tests)
│   └── powers_test.ts                # Powers unit tests (20 tests)
├── main.ts                   # HTTP server with routing & UI
├── deno.json                 # Deno configuration
└── README.md
```

## 🧠 How It Works

1. **Document Ingestion**: Documents are embedded into 384-dimensional vectors
2. **Normalization**: All vectors are pre-normalized to unit length
3. **Storage**: Embeddings stored as `Float32Array` for cache-friendly access
4. **Indexing**: Vectors are inserted into an HNSW graph (built incrementally on
   `add()`)
5. **Search**: Under 2000 docs → exact brute-force dot product; above → HNSW
   approximate search with O(log n) query time

### HNSW Index

The HNSW implementation follows the original 2016 paper:

- **Level multiplier**: `mL = 1/ln(M)` — ~94% of nodes on layer 0
- **Two-phase insert**: greedy walk (ef = 1) on upper layers, then
  efConstruction-width search + bidirectional connect on lower layers
- **Neighbor shrinkage**: connections pruned when exceeding M_max (2·M on
  layer 0)
- **Configurable**: M, efConstruction, and efSearch exposed as optional
  parameters

### Embedding Models

The engine uses a pluggable embedding interface:

- **SimpleEmbeddingModel** (default): Hash-based embeddings for zero-dependency
  operation
- **TransformersJsEmbedding** (production): Swap in `@huggingface/transformers`
  for state-of-the-art embeddings

You can swap the embedding model at runtime using the **EmbeddingSwap** Power
(see Powers below) or by uncommenting the `TransformersJsEmbedding` class in
[lib/semantic-engine.ts](lib/semantic-engine.ts).

## ⚡ Powers (Plugin System)

Powers are MySSE's extensibility mechanism. A Power is a plain object with
lifecycle hooks that the engine calls at key points in the search pipeline.
Register a Power with `engine.use()`, remove it with `engine.eject()`. When no
Powers are registered there is zero overhead.

### How It Works

The engine runs each hook in registration order at the appropriate point:

| Hook            | When it runs                          | Typical use                          |
| --------------- | ------------------------------------- | ------------------------------------ |
| `beforeAdd`     | Before documents are embedded         | Transform or enrich documents        |
| `afterAdd`      | After documents are stored and indexed| Build auxiliary indexes (e.g. BM25)  |
| `beforeSearch`  | Before the core search runs           | Rewrite queries, return cached results |
| `afterSearch`   | After results are returned            | Re-rank, filter, or fuse results     |
| `embed`         | Replaces the default embedding model  | Plug in a real ML model at runtime   |
| `onDelete`      | After a document is deleted           | Clean up auxiliary state             |
| `onClear`       | After the entire index is cleared     | Reset auxiliary state                |

### Built-in Powers

MySSE ships with four ready-to-use Powers in `lib/powers/`:

#### QueryCache

Caches search results by query string. Repeated identical queries are served
instantly from memory.

```ts
import { QueryCache } from "./lib/powers/cache.ts";

engine.use(QueryCache());                     // defaults: 100 entries, 60 s TTL
engine.use(QueryCache({ maxSize: 200, ttl: 30_000 }));  // custom settings
```

#### HybridSearch

Combines dense semantic retrieval with sparse BM25 keyword retrieval using
Reciprocal Rank Fusion (RRF). This gives you the best of both worlds — semantic
understanding *and* exact keyword matching.

```ts
import { HybridSearch } from "./lib/powers/hybrid-search.ts";

engine.use(HybridSearch());                   // 50/50 blend (recommended)
engine.use(HybridSearch({ alpha: 0.7 }));     // 70% semantic, 30% keyword
```

| `alpha` value | Behaviour                       |
| ------------- | ------------------------------- |
| `1.0`         | Pure semantic (dense only)      |
| `0.5`         | Equal blend (default)           |
| `0.0`         | Pure keyword (BM25 only)        |

#### MetadataFilter

Filters search results by document metadata using a predicate function.

```ts
import { MetadataFilter } from "./lib/powers/metadata-filter.ts";

// Only return published documents
engine.use(MetadataFilter((meta) => meta.published === true));

// Only return documents from a specific source
engine.use(MetadataFilter((meta) => meta.source === "docs"));
```

#### EmbeddingSwap

Hot-swaps the embedding model at runtime without restarting the engine or
changing constructor options.

```ts
import { EmbeddingSwap } from "./lib/powers/embedding-swap.ts";

engine.use(EmbeddingSwap(async (texts) => {
  // Call your ML model, external API, etc.
  return texts.map(() => new Float32Array(384).fill(0.1));
}));
```

### Combining Powers

Powers compose naturally. Register as many as you need — they run in order:

```ts
import { QueryCache } from "./lib/powers/cache.ts";
import { HybridSearch } from "./lib/powers/hybrid-search.ts";
import { MetadataFilter } from "./lib/powers/metadata-filter.ts";

engine.use(QueryCache({ ttl: 30_000 }));
engine.use(HybridSearch({ alpha: 0.6 }));
engine.use(MetadataFilter((meta) => meta.published === true));
```

With this setup, each search: checks the cache → runs semantic + BM25 fusion →
filters by metadata → caches the result for next time.

### Managing Powers at Runtime

```ts
// Register a Power
engine.use(QueryCache());

// List registered Powers
console.log(engine.powers);   // ["QueryCache"]

// Remove a Power by name
engine.eject("QueryCache");   // returns true if found and removed

// The /api/status endpoint also reports active Powers
curl http://localhost:8000/api/status
// { "status": "healthy", "documents": 42, "powers": ["QueryCache"], ... }
```

### Writing a Custom Power

A Power is any object that satisfies the `Power` interface. At minimum it needs
a `name`; every hook is optional — implement only what you need.

```ts
import type { Power } from "./lib/semantic-engine.ts";

const logger: Power = {
  name: "Logger",
  beforeSearch(ctx) {
    console.log(`Searching for: ${ctx.query}`);
    return ctx;
  },
  afterSearch(results, query) {
    console.log(`Found ${results.length} results for "${query}"`);
    return results;
  },
};

engine.use(logger);
```

### Persistence (toJSON / fromJSON)

The engine supports exporting and importing its full state for persistence
(e.g. to Deno KV or a JSON file). The HNSW index is rebuilt automatically on
import, and `afterAdd` hooks fire so Powers like HybridSearch can rebuild
auxiliary state.

```ts
// Export
const snapshot = engine.toJSON();
await Deno.writeTextFile("index.json", JSON.stringify(snapshot));

// Import
const data = JSON.parse(await Deno.readTextFile("index.json"));
await engine.fromJSON(data);
```

### Performance

| Metric                 | Value                                 |
| ---------------------- | ------------------------------------- |
| Brute-force (10k docs) | ~5 ms per query                       |
| HNSW (10k docs)        | ~1.5 ms per query (4–6× faster)       |
| HNSW recall@10         | ≥ 92% (typically 95–97%)              |
| Index build (10k docs) | ~20 s (one-time, incremental on add)  |
| Memory per vector      | ~1.5 KB (384 × 4 bytes + graph edges) |

## 🔧 Tasks

```bash
deno task dev      # Start development server with hot reload
deno task start    # Start production server
deno task check    # Run format, lint, and type checks
deno task test     # Run tests
```

## 🔮 Future Extensions

- **Transformers.js**: Enable real ML embeddings with
  `@huggingface/transformers` (or use the EmbeddingSwap Power today)
- **WebGPU Acceleration**: Automatic GPU acceleration when available
- **Multi-Modal**: Swap to CLIP model for image+text embeddings via
  EmbeddingSwap
- **Quantization**: Enable `quantized: true` for smaller model footprint
- **Persistence**: Expand `toJSON` / `fromJSON` with streaming and Deno KV
  adapters
- **Community Powers**: Build and share your own Powers — logging,
  rate-limiting, A/B testing, result enrichment, and more

## 📚 Getting Started Guide

See [GETTING-STARTED.md](GETTING-STARTED.md) for a beginner-friendly walkthrough
on adding semantic search to your project with MySSE.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details
