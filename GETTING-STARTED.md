# Getting Started with MySSE

Welcome! This guide will walk you through adding semantic search to your project
using MySSE. No machine-learning background required — if you can write a
`fetch()` call, you can use MySSE.

## What is Semantic Search?

Traditional search matches **exact words**. If you search for "car", you won't
find a document that says "automobile." Semantic search understands **meaning**,
so "car" and "automobile" are treated as related.

MySSE gives you this superpower in a single Deno server with zero external
services — everything runs in memory, on your machine.

---

## 1. Install Deno

MySSE runs on Deno. If you don't have it yet:

```bash
# macOS / Linux
curl -fsSL https://deno.land/install.sh | sh

# Windows (PowerShell)
irm https://deno.land/install.ps1 | iex
```

Verify it works:

```bash
deno --version
# You need Deno 2.0 or later
```

---

## 2. Clone and Start MySSE

```bash
git clone https://github.com/thewizster/MySSE.git
cd MySSE
deno task dev
```

You should see:

```
🚀 MySSE starting on http://localhost:8000
```

Open `http://localhost:8000` in your browser — you'll see the search UI. It's
empty right now because we haven't added any documents yet. Let's fix that.

---

## 3. Add Your First Documents

Documents are the things you want to search through — product descriptions, help
articles, notes, whatever you like. Each document needs:

- **`content`** (required): the text to search
- **`id`** (optional): a unique identifier (auto-generated if you skip it)
- **`metadata`** (optional): any extra info you want to attach

Open a new terminal and run:

```bash
curl -X POST http://localhost:8000/api/add \
  -H "Content-Type: application/json" \
  -d '[
    {"id": "1", "content": "How to reset your password"},
    {"id": "2", "content": "Changing your account email address"},
    {"id": "3", "content": "Setting up two-factor authentication"},
    {"id": "4", "content": "Deleting your account permanently"},
    {"id": "5", "content": "Updating your billing and payment info"}
  ]'
```

You'll get back:

```json
{
  "success": true,
  "added": 5,
  "total": 5,
  "ids": ["1", "2", "3", "4", "5"]
}
```

---

## 4. Search!

Now search for something — notice you don't need to use the exact words from
your documents:

```bash
curl "http://localhost:8000/api/search?q=forgot+my+login+credentials&k=3"
```

Even though no document says "forgot my login credentials", MySSE returns
account-related articles ranked by relevance — password reset and email changes
near the top, billing info further down.

The response looks like:

```json
{
  "query": "forgot my login credentials",
  "results": [
    {
      "id": "2",
      "content": "Changing your account email address",
      "score": 0.024
    },
    {
      "id": "1",
      "content": "How to reset your password",
      "score": 0.014
    },
    {
      "id": "5",
      "content": "Updating your billing and payment info",
      "score": 0.006
    }
  ],
  "count": 3,
  "total_documents": 5
}
```

### A note about scores

The **ranking order** is what matters — the result at the top is the best match.

The absolute score values look low (0.02 instead of 0.9) because MySSE ships
with a lightweight **hash-based embedding model** that has zero external
dependencies. It's great for getting started and for testing, but it doesn't
capture deep semantic meaning the way a real ML model does.

When you swap in a production embedding model like Transformers.js (see the
commented-out `TransformersJsEmbedding` class in `lib/semantic-engine.ts`),
you'll see scores in the 0.6–0.95 range and much stronger semantic matching —
"forgot my login credentials" will rank "How to reset your password" clearly
first.

**Bottom line**: use the built-in embeddings for development and testing. Plug in
a real model when you're ready for production.

You can also try searching from the web UI at `http://localhost:8000`.

---

## 5. Use MySSE from Your Own Code

MySSE is a standard HTTP API, so you can call it from any language. Here are
some examples.

### JavaScript / TypeScript (browser or Deno)

```ts
// Add documents
await fetch("http://localhost:8000/api/add", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify([
    { id: "faq-1", content: "How do I cancel my subscription?" },
    { id: "faq-2", content: "Where can I download my invoice?" },
  ]),
});

// Search
const res = await fetch(
  "http://localhost:8000/api/search?q=stop+paying&k=5"
);
const data = await res.json();

for (const result of data.results) {
  console.log(`${result.id}: ${result.content} (${(result.score * 100).toFixed(1)}%)`);
}
```

### Python

```python
import requests

# Add documents
requests.post("http://localhost:8000/api/add", json=[
    {"id": "1", "content": "How to reset your password"},
    {"id": "2", "content": "Changing your account email address"},
])

# Search
response = requests.get("http://localhost:8000/api/search", params={
    "q": "forgot my password",
    "k": 3,
})

for result in response.json()["results"]:
    print(f"{result['id']}: {result['content']} ({result['score']:.0%})")
```

---

## 6. Useful API Endpoints

| Endpoint             | Method   | What it does                          |
| -------------------- | -------- | ------------------------------------- |
| `/api/add`           | `POST`   | Add one or more documents             |
| `/api/search?q=...`  | `GET`    | Search documents by meaning           |
| `/api/status`        | `GET`    | Check health and document count       |
| `/api/clear`         | `DELETE` | Remove all documents from the index   |

---

## 7. How Search Scales Automatically

You don't need to configure anything — MySSE picks the best strategy for you:

| Document Count | Search Method     | What It Means                           |
| -------------- | ----------------- | --------------------------------------- |
| ≤ 2 000        | **Brute-force**   | Compares your query to every document. Perfect accuracy. |
| > 2 000        | **HNSW (approximate)** | Uses a graph-based index for much faster search. Accuracy stays above 92%. |

The switch happens automatically. If you add your 2 001st document, the next
search uses the fast path. If you delete back down to 2 000, it switches back to
exact search. You never have to think about it.

### Want to Tune It?

If you're importing MySSE as a library (not using the HTTP server), you can
customize the behavior:

```ts
import { SemanticEngine } from "./lib/semantic-engine.ts";

const engine = SemanticEngine.getInstance({
  useANN: true,         // set to false to always use brute-force
  annThreshold: 5000,   // switch to HNSW above this many documents
  m: 16,                // HNSW connections per node (higher = better recall, more RAM)
  efSearch: 128,        // HNSW search beam width (higher = better recall, slower)
});
```

For most users, the defaults work great.

---

## 8. Persisting Your Index

MySSE is 100% in-memory, so your documents disappear when the server restarts.
To save and restore them, use the export/import feature:

```ts
import { engine } from "./lib/semantic-engine.ts";

// Save to a file
const data = engine.toJSON();
await Deno.writeTextFile("index.json", JSON.stringify(data));

// Load it back later
const saved = JSON.parse(await Deno.readTextFile("index.json"));
engine.fromJSON(saved);
```

---

## 9. Running Tests

MySSE comes with a full test suite — 33 tests covering the search engine, the
HNSW index, recall quality, and latency benchmarks:

```bash
deno task test
```

---

## 10. What's Next?

Now that you have semantic search running, here are some ideas:

- **Build a help center**: Add your FAQ articles, let users search by question
- **Power a chatbot**: Use MySSE to find relevant context before sending to an
  LLM
- **Search your notes**: Import your markdown files and find things by meaning
- **Product search**: Let customers describe what they want instead of filtering
  by category

The API is simple on purpose. Start small, add documents, search them. That's
it.

---

*© 2026 Raymond Brady. All Rights Reserved. The LUMOS Initiative. Built with
❤️ in Texas.*
