// include/strata/program/split_rules.hpp - rules for the automatic layer split that do not need a GPU to check.
//
// A stage that owns one or two layers holds a handful of expert slots (a 12 GB 3060 with one layer held 512), and
// the prompt chunk of the whole pipeline is clamped to what the smallest cache can lend (#448): 576 -> 347 tok/s in
// #1616.  So the search first looks for a placement in which every stage owns at least kMinStageLayers layers, and
// only falls back to the unrestricted best when no such placement can start the prompt path.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::program::split_rules {

constexpr int64_t kMinStageLayers = 3;

/// The restriction only makes sense when the model has the layers for it.
inline bool restriction_possible(int64_t n_layers, int stages) {
    return stages >= 2 && n_layers >= kMinStageLayers * stages;
}

/// cuts: the first layer of each later stage (stages - 1 of them); n_layers: the model's layers.
/// Returns the size of the smallest stage.
inline int64_t smallest_stage(const std::vector<int64_t>& cuts, int64_t n_layers) {
    int64_t lo = n_layers, prev = 0;
    for (const int64_t c : cuts) {
        if (c - prev < lo) lo = c - prev;
        prev = c;
    }
    if (n_layers - prev < lo) lo = n_layers - prev;
    return lo;
}

/// True when some stage owns fewer than kMinStageLayers layers.
inline bool has_tiny_stage(const std::vector<int64_t>& cuts, int64_t n_layers) {
    return smallest_stage(cuts, n_layers) < kMinStageLayers;
}

/// extras/split-prefill (opt-in, STRATA_SPLIT_PF_CHUNKS=k): the chunk a layer split reads a `tokens`-token request
/// segment in, so the stages overlap.  The prompt path pipelines a split over chunks (Prefill::run_impl: stage s+1
/// reads chunk c while stage s reads c+1), but the auto chunk is the largest one the caches can lend, so a prompt up
/// to that size is ONE chunk and the stages run one after the other.  Here the segment is cut into
/// n = min(k, tokens / min_chunk) chunks of equal size (ceil(tokens / n), 256-aligned): no chunk below `min_chunk`
/// (and no short tail that streams every expert again), never above `max_chunk`.  Above k * max_chunk tokens the
/// chunks are max_chunk's, as without it.  k <= 1, or a segment too short for two chunks: `max_chunk` unchanged.
inline int64_t pipeline_chunk(int64_t tokens, int64_t max_chunk, int64_t k, int64_t min_chunk) {
    if (k <= 1 || tokens <= 0 || max_chunk <= 0) return max_chunk;
    const int64_t lo = min_chunk < 256 ? 256 : min_chunk;
    int64_t n = tokens / lo;
    if (n > k) n = k;
    if (n <= 1) return max_chunk;
    int64_t c = (tokens + n - 1) / n;
    c = (c + 255) / 256 * 256;
    return c < max_chunk ? c : max_chunk;
}

}  // namespace strata::program::split_rules
