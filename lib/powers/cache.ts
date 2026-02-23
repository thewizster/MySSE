// lib/powers/cache.ts
// QueryCache Power — short-circuits search on cache hits via beforeSearch,
// stores results via afterSearch, and invalidates on clear.

import type { Power, SearchResult } from "../semantic-engine.ts";

interface CacheEntry {
  results: SearchResult[];
  expires: number;
}

export interface QueryCacheOptions {
  /** Maximum number of cached queries (default: 100). FIFO eviction when full. */
  maxSize?: number;
  /** Time-to-live in milliseconds (default: 60_000 = 1 minute). */
  ttl?: number;
}

/**
 * QueryCache Power — caches `search()` results by query string.
 *
 * - `beforeSearch`: returns cached results as `shortCircuit` on a cache hit.
 * - `afterSearch`: stores fresh results in the cache.
 * - `onClear`: invalidates the entire cache.
 *
 * @example
 * ```ts
 * import { QueryCache } from "./lib/powers/cache.ts";
 * engine.use(QueryCache({ maxSize: 200, ttl: 30_000 }));
 * ```
 */
export function QueryCache(opts: QueryCacheOptions = {}): Power {
  const maxSize = opts.maxSize ?? 100;
  const ttl = opts.ttl ?? 60_000;
  const cache = new Map<string, CacheEntry>();

  return {
    name: "QueryCache",

    beforeSearch(ctx) {
      const entry = cache.get(ctx.query);
      if (entry && Date.now() < entry.expires) {
        return { ...ctx, shortCircuit: entry.results };
      }
      return ctx;
    },

    afterSearch(results, query) {
      // Evict oldest entry when the cache is full (insertion-order eviction)
      if (cache.size >= maxSize) {
        const oldest = cache.keys().next().value as string;
        cache.delete(oldest);
      }
      cache.set(query, { results, expires: Date.now() + ttl });
      return results;
    },

    onClear() {
      cache.clear();
    },
  };
}
