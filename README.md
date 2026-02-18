# MySSE - In-Memory Semantic Search Engine

A fully in-RAM semantic search engine built with **Deno** in pure TypeScript —
minimalist, elegant, blazing-fast, and built for 2026+.

![MySSE Home Page](https://github.com/user-attachments/assets/0cc9ec0e-ac94-4cd3-a0ea-11f95c89f515)

## ✨ Features

- **100% In-Memory**: No external DB, no disk I/O during queries
- **AI-Ready**: Pluggable embedding interface (simple hash-based included,
  Transformers.js ready)
- **Blazing Fast**: Pre-normalized vectors → cosine similarity as simple dot
  product
- **Type-Safe**: Pure TypeScript end-to-end
- **Secure**: Deno's secure-by-default runtime
- **Minimal**: ~200 LOC core, zero external dependencies

## 🚀 Quick Start

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
│   └── semantic-engine.ts    # Core semantic search engine (~200 LOC)
├── tests/
│   └── semantic-engine_test.ts
├── main.ts                   # HTTP server with routing
├── deno.json                 # Deno configuration
└── README.md
```

## 🧠 How It Works

1. **Document Ingestion**: Documents are embedded into 384-dimensional vectors
2. **Normalization**: All vectors are pre-normalized to unit length
3. **Storage**: Embeddings stored as `Float32Array` for cache-friendly access
4. **Search**: Cosine similarity reduces to a simple dot product (extremely
   fast)

### Embedding Models

The engine uses a pluggable embedding interface:

- **SimpleEmbeddingModel** (default): Hash-based embeddings for zero-dependency
  operation
- **TransformersJsEmbedding** (production): Swap in `@huggingface/transformers`
  for state-of-the-art embeddings

To enable Transformers.js embeddings, uncomment the `TransformersJsEmbedding`
class in `lib/semantic-engine.ts` and update the imports.

### Performance

- Brute-force search on 10k documents ≈ 4 million operations → **< 5ms** on
  modern hardware
- Pre-normalized vectors eliminate magnitude calculations
- `Float32Array` + V8 turbo = near-native speed

## 🔧 Tasks

```bash
deno task dev      # Start development server with hot reload
deno task start    # Start production server
deno task check    # Run format, lint, and type checks
deno task test     # Run tests
```

## 🔮 Future Extensions

- **Transformers.js**: Enable real ML embeddings with
  `@huggingface/transformers`
- **WebGPU Acceleration**: Automatic GPU acceleration when available
- **Multi-Modal**: Swap to CLIP model for image+text embeddings
- **Quantization**: Enable `quantized: true` for smaller model footprint
- **Persistence**: Serialize to Deno KV or JSON file
- **HNSW Index**: Drop-in pure-TS HNSW for >50k documents

## 📄 License

MIT License - see [LICENSE](LICENSE) for details
