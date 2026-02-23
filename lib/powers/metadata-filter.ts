// lib/powers/metadata-filter.ts
// MetadataFilter Power — removes results whose metadata does not satisfy a
// caller-supplied predicate, via the afterSearch hook.

import type { Power } from "../semantic-engine.ts";

/**
 * MetadataFilter Power — filters search results by document metadata.
 *
 * - `afterSearch`: removes results where `predicate(result.metadata)` is false.
 *
 * @param predicate A function that receives each result's metadata object and
 *   returns `true` to keep the result or `false` to discard it.
 *
 * @example
 * ```ts
 * import { MetadataFilter } from "./lib/powers/metadata-filter.ts";
 * // Only return documents tagged as published
 * engine.use(MetadataFilter((meta) => meta.published === true));
 * ```
 */
export function MetadataFilter(
  predicate: (meta: Record<string, unknown>) => boolean,
): Power {
  return {
    name: "MetadataFilter",

    afterSearch(results) {
      return results.filter((r) => predicate(r.metadata ?? {}));
    },
  };
}
