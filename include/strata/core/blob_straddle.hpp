// include/strata/core/blob_straddle.hpp - strata-nvfp4 (fork issue #20): the expert blobs of a RAM copy that no single
// page-locked registration holds whole.
//
// FileExpertSource's path 2 registers its RAM copy slice by slice on page boundaries, so a few experts start in one
// registration and end in the next.  pinned() already counts them as not pinned (the CPU computes them), but a
// host-to-device copy of such a blob (the adaptive tier's swaps, the cache refills, a lent slot's refill) is one
// cudaMemcpyAsync over two registrations, which CUDA refuses (cudaErrorInvalidValue: "an adaptive refill failed",
// 0.2.0-beta-nvfp4.3 with a 64 GiB Mix pack copy).  The source therefore gives each such blob its own page-locked copy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

/// The indexes `i` whose blob [offsets[i], offsets[i] + bytes_of(i)) lies inside no single range of `ranges`
/// ([start, end), sorted by start, not overlapping).  An offset equal to `none` (no blob) is skipped.
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
        // the last range starting at or before `off`
        size_t lo = 0, hi = ranges.size();
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (ranges[mid].first <= off) lo = mid + 1;
            else hi = mid;
        }
        const bool whole = lo > 0 && off < ranges[lo - 1].second && n <= ranges[lo - 1].second - off;
        if (!whole) out.push_back(i);
    }
    return out;
}

}  // namespace strata::core
