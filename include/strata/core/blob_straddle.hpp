// include/strata/core/blob_straddle.hpp - strata-nvfp4 (fork issue #20): the places of a RAM copy that cross the edge
// of a page-locked registration, and their own copies.
//
// FileExpertSource's path 2 registers its RAM copy slice by slice on page boundaries, so a few experts start in one
// registration and end in the next (or past the last one, when the driver refused a slice or the cap stopped them).
// pinned() already counts them as not pinned (the CPU computes them), but a host-to-device copy of such a blob (the
// adaptive tier's swaps, the cache refills, a lent slot's refill) is one cudaMemcpyAsync across a registration's edge,
// which CUDA refuses (cudaErrorInvalidValue: "an adaptive refill failed", 0.2.0-beta-nvfp4.3 with a 64 GiB Mix pack
// copy).  The source therefore gives each such PLACE its own page-locked copy.  The copy belongs to the place, not to
// the expert: the adaptive tier moves places between experts (exchange_cache_complement), so a lookup goes through the
// expert's current offset and every write into the place is written into its copy too.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::core {

/// The indexes `i` whose blob [offsets[i], offsets[i] + bytes_of(i)) overlaps a range of `ranges` ([start, end), sorted
/// by start, not overlapping) without lying inside one.  A blob inside one range, or outside every range (plain
/// pageable memory: a legal copy source, and the unpinned tail of a capped path 2 can be tens of GiB), is not listed.
/// An offset equal to `none` (no blob) is skipped.
template <class BytesOf>
std::vector<size_t> straddling_blobs(const std::vector<uint64_t>& offsets, uint64_t none,
                                     const std::vector<std::pair<uint64_t, uint64_t>>& ranges, BytesOf bytes_of) {
    std::vector<size_t> out;
    if (ranges.empty()) return out;
    for (size_t i = 0; i < offsets.size(); ++i) {
        const uint64_t off = offsets[i];
        if (off == none) continue;
        const uint64_t n = (uint64_t) bytes_of(i);
        if (n == 0) continue;
        // lo = the ranges starting at or before `off`; ranges[lo - 1] may hold it, ranges[lo] may start inside it
        size_t lo = 0, hi = ranges.size();
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (ranges[mid].first <= off) lo = mid + 1;
            else hi = mid;
        }
        const bool in_prev = lo > 0 && off < ranges[lo - 1].second;
        if (in_prev && n <= ranges[lo - 1].second - off) continue;   // inside one registration
        const bool into_next = lo < ranges.size() && ranges[lo].first - off < n;
        if (in_prev || into_next) out.push_back(i);
    }
    return out;
}

/// A place's own copy, `copies` = (offset, copy) sorted by offset; nullptr when the place at `offset` has none.
inline uint8_t* straddle_copy_at(const std::vector<std::pair<uint64_t, uint8_t*>>& copies, uint64_t offset) {
    const auto it = std::lower_bound(copies.begin(), copies.end(), offset,
                                     [](const std::pair<uint64_t, uint8_t*>& e, uint64_t v) { return e.first < v; });
    return it != copies.end() && it->first == offset ? it->second : nullptr;
}

/// A write of `n` bytes into the place at `offset` of the RAM copy: the same bytes into that place's own copy, if any
/// (true when there was one).
inline bool straddle_mirror(const std::vector<std::pair<uint64_t, uint8_t*>>& copies, uint64_t offset, const void* src,
                            size_t n) {
    uint8_t* c = straddle_copy_at(copies, offset);
    if (c != nullptr) std::memcpy(c, src, n);
    return c != nullptr;
}

}  // namespace strata::core
