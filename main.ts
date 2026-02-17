// main.ts
// Deno native HTTP server with minimal routing

import { engine } from "./lib/semantic-engine.ts";

const PORT = parseInt(Deno.env.get("PORT") ?? "8000");

// Simple router
async function handleRequest(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const path = url.pathname;
  const method = req.method;

  // CORS headers for API endpoints
  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };

  // Handle CORS preflight
  if (method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // API Routes
    if (path === "/api/add" && method === "POST") {
      return await handleAdd(req, corsHeaders);
    }

    if (path === "/api/search" && method === "GET") {
      return await handleSearch(url, corsHeaders);
    }

    if (path === "/api/status" && method === "GET") {
      return handleStatus(corsHeaders);
    }

    if (path === "/api/clear" && method === "DELETE") {
      return handleClear(corsHeaders);
    }

    // Serve home page
    if (path === "/" && method === "GET") {
      return serveHomePage();
    }

    // 404 for unknown routes
    return new Response(JSON.stringify({ error: "Not found" }), {
      status: 404,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Request error:", error);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
}

// POST /api/add - Add documents
async function handleAdd(
  req: Request,
  corsHeaders: Record<string, string>,
): Promise<Response> {
  const body = await req.json();
  const documents = Array.isArray(body) ? body : [body];

  const validDocs = documents.map((d, i) => ({
    id: d.id ?? `doc-${Date.now()}-${i}`,
    content: String(d.content ?? ""),
    metadata: d.metadata ?? {},
  }));

  const docsWithContent = validDocs.filter((d) => d.content.trim().length > 0);

  if (docsWithContent.length === 0) {
    return Response.json(
      {
        error:
          "No valid documents provided. Each document needs 'content' field.",
      },
      { status: 400, headers: corsHeaders },
    );
  }

  await engine.add(docsWithContent);

  return Response.json({
    success: true,
    added: docsWithContent.length,
    total: engine.size,
    ids: docsWithContent.map((d) => d.id),
  }, { headers: corsHeaders });
}

// GET /api/search - Search documents
async function handleSearch(
  url: URL,
  corsHeaders: Record<string, string>,
): Promise<Response> {
  const query = url.searchParams.get("q");
  const topK = parseInt(url.searchParams.get("k") ?? "10", 10);

  if (!query || query.trim().length === 0) {
    return Response.json(
      { error: "Missing or empty query parameter '?q='" },
      { status: 400, headers: corsHeaders },
    );
  }

  if (isNaN(topK) || topK < 1 || topK > 100) {
    return Response.json(
      { error: "Parameter 'k' must be a number between 1 and 100" },
      { status: 400, headers: corsHeaders },
    );
  }

  const results = await engine.search(query, topK);

  return Response.json({
    query,
    results,
    count: results.length,
    total_documents: engine.size,
  }, { headers: corsHeaders });
}

// GET /api/status - Get engine status
function handleStatus(corsHeaders: Record<string, string>): Response {
  return Response.json({
    status: "healthy",
    documents: engine.size,
    timestamp: new Date().toISOString(),
  }, { headers: corsHeaders });
}

// DELETE /api/clear - Clear all documents
function handleClear(corsHeaders: Record<string, string>): Response {
  const previousSize = engine.size;
  engine.clear();
  return Response.json({
    success: true,
    cleared: previousSize,
    total: engine.size,
  }, { headers: corsHeaders });
}

// Serve home page with embedded search UI
function serveHomePage(): Response {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MySSE - Semantic Search Engine</title>
  <style>
    :root {
      --bg: #0a0a0a;
      --fg: #fafafa;
      --accent: #6366f1;
      --accent-hover: #818cf8;
      --border: #27272a;
      --muted: #71717a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--fg);
      min-height: 100vh;
      line-height: 1.6;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      padding: 2rem;
    }
    header {
      text-align: center;
      margin-bottom: 3rem;
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .tagline {
      color: var(--muted);
      margin-top: 0.5rem;
    }
    .search-container {
      background: #18181b;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 2rem;
    }
    .search-box {
      display: flex;
      gap: 0.75rem;
    }
    .search-input {
      flex: 1;
      padding: 0.875rem 1rem;
      font-size: 1rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--fg);
      outline: none;
      transition: border-color 0.2s;
    }
    .search-input:focus {
      border-color: var(--accent);
    }
    .search-input::placeholder {
      color: var(--muted);
    }
    .search-btn {
      padding: 0.875rem 1.5rem;
      font-size: 1rem;
      font-weight: 500;
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .search-btn:hover:not(:disabled) {
      background: var(--accent-hover);
    }
    .search-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .results {
      margin-top: 1.5rem;
    }
    .result-item {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 0.75rem;
      transition: border-color 0.2s;
    }
    .result-item:hover {
      border-color: #3f3f46;
    }
    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    .result-id {
      font-family: monospace;
      font-size: 0.75rem;
      color: var(--muted);
    }
    .result-score {
      font-family: monospace;
      font-size: 0.75rem;
      color: var(--accent);
      background: rgba(99, 102, 241, 0.1);
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
    }
    .result-content {
      font-size: 0.9375rem;
      line-height: 1.6;
      color: #d4d4d8;
    }
    .error {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #f87171;
      padding: 0.75rem 1rem;
      border-radius: 8px;
      margin-top: 1rem;
    }
    .no-results, .loading {
      text-align: center;
      color: var(--muted);
      padding: 2rem;
    }
    .hint {
      font-size: 0.875rem;
      color: var(--muted);
      margin-top: 1rem;
      text-align: center;
    }
    .hint code {
      background: var(--border);
      padding: 0.125rem 0.375rem;
      border-radius: 4px;
      font-size: 0.8125rem;
    }
    .features {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-top: 3rem;
    }
    .feature {
      background: #18181b;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
    }
    .feature h3 {
      font-size: 1rem;
      margin-bottom: 0.5rem;
    }
    .feature p {
      color: var(--muted);
      font-size: 0.875rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🔍 MySSE</h1>
      <p class="tagline">In-Memory Semantic Search Engine</p>
    </header>

    <div class="search-container">
      <div class="search-box">
        <input type="text" class="search-input" id="searchInput" placeholder="Search for anything...">
        <button class="search-btn" id="searchBtn">Search</button>
      </div>
      <div id="results"></div>
      <p class="hint" id="hint">Add documents via <code>POST /api/add</code>, then search here</p>
    </div>

    <div class="features">
      <div class="feature">
        <h3>⚡ Blazing Fast</h3>
        <p>100% in-memory search with pre-normalized embeddings</p>
      </div>
      <div class="feature">
        <h3>🧠 AI-Powered</h3>
        <p>Transformers.js embeddings with WebGPU acceleration</p>
      </div>
      <div class="feature">
        <h3>🔒 Secure</h3>
        <p>Deno's secure-by-default runtime with minimal permissions</p>
      </div>
      <div class="feature">
        <h3>📦 Minimal</h3>
        <p>Zero external databases, pure TypeScript, ~200 LOC core</p>
      </div>
    </div>
  </div>

  <script>
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsDiv = document.getElementById('results');
    const hint = document.getElementById('hint');

    async function search() {
      const query = searchInput.value.trim();
      if (!query) return;

      searchBtn.disabled = true;
      searchBtn.textContent = 'Searching...';
      resultsDiv.innerHTML = '<div class="loading">🔍 Searching...</div>';
      hint.style.display = 'none';

      try {
        const response = await fetch('/api/search?q=' + encodeURIComponent(query) + '&k=10');
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Search failed');
        }

        if (data.results.length === 0) {
          resultsDiv.innerHTML = '<div class="no-results">No results found. Try adding some documents first!</div>';
        } else {
          resultsDiv.innerHTML = data.results.map(r => 
            '<div class="result-item">' +
              '<div class="result-header">' +
                '<span class="result-id">' + escapeHtml(r.id) + '</span>' +
                '<span class="result-score">' + (r.score * 100).toFixed(1) + '% match</span>' +
              '</div>' +
              '<div class="result-content">' + escapeHtml(r.content) + '</div>' +
            '</div>'
          ).join('');
        }
      } catch (e) {
        resultsDiv.innerHTML = '<div class="error">' + escapeHtml(e.message) + '</div>';
      } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = 'Search';
      }
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    searchBtn.addEventListener('click', search);
    searchInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') search();
    });
  </script>
</body>
</html>`;

  return new Response(html, {
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}

// Start the server
console.log(`🚀 MySSE starting on http://localhost:${PORT}`);
Deno.serve({ port: PORT }, handleRequest);
