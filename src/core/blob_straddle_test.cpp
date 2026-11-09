// src/core/blob_straddle_test.cpp - strata-nvfp4 (fork issue #20): straddling_blobs finds the blobs that cross a
// registration's edge, and a place's own copy follows the place, not the expert (host only, no GPU).
#include "strata/core/blob_straddle.hpp"

#include <cstdio>

using strata::core::straddle_copy_at;
using strata::core::straddle_mirror;
using strata::core::straddling_blobs;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const uint64_t none = ~0ull;
    auto bytes = [](size_t) { return (uint64_t) 100; };
    // two registrations [0, 4096) and [4096, 8192): a blob at 4000 runs into the second
    const std::vector<std::pair<uint64_t, uint64_t>> two = {{0, 4096}, {4096, 8192}};
    {
        const std::vector<uint64_t> off = {0, 3996, 4000, 4096, 8092, none};
        const auto s = straddling_blobs(off, none, two, bytes);
        CHECK(s.size() == 1 && s[0] == 2);   // 3996 ends at 4096 exactly (inside), 8092 ends at 8192 (inside)
    }
    {   // registrations with a gap: a blob across an edge into the gap or past the last range is listed; a blob wholly
        // in the gap or past the last range is plain pageable memory (a legal copy source) and is not
        const std::vector<std::pair<uint64_t, uint64_t>> gap = {{0, 1000}, {2000, 3000}};
        const std::vector<uint64_t> off = {950, 1200, 2000, 2950, 3100, 1950};
        const auto s = straddling_blobs(off, none, gap, bytes);
        CHECK(s.size() == 3 && s[0] == 0 && s[1] == 3 && s[2] == 5);   // 1950 starts in the gap, ends in [2000, 3000)
    }
    {   // a capped path 2: the unregistered tail past the last slice copies nothing but the one blob across its edge
        const std::vector<uint64_t> off = {8150, 9000, 20000, 4000};
        const auto s = straddling_blobs(off, none, two, bytes);
        CHECK(s.size() == 2 && s[0] == 0 && s[1] == 3);
    }
    {   // the copy belongs to the place: after an exchange moves the place to another expert, a lookup by that expert's
        // offset finds the copy, a write into the place reaches it, and the expert that left finds none
        uint8_t a[4] = {1, 1, 1, 1}, b[4] = {2, 2, 2, 2};
        const std::vector<std::pair<uint64_t, uint8_t*>> copies = {{3996, a}, {8150, b}};
        std::vector<uint64_t> offsets = {3996, none};   // expert 0 holds the place at 3996, expert 1 is in VRAM
        CHECK(straddle_copy_at(copies, offsets[0]) == a);
        CHECK(straddle_copy_at(copies, 4000) == nullptr && straddle_copy_at(copies, 0) == nullptr);
        CHECK(straddle_copy_at(copies, 9000) == nullptr);
        const uint8_t bytes1[4] = {7, 8, 9, 10};   // the swap: expert 1's bytes into expert 0's place, offsets moved
        straddle_mirror(copies, offsets[0], bytes1, 4);
        std::swap(offsets[0], offsets[1]);
        CHECK(straddle_copy_at(copies, offsets[1]) == a && a[0] == 7 && a[3] == 10);
        CHECK(offsets[0] == none);
        straddle_mirror(copies, 5000, bytes1, 4);   // a place with no copy: nothing written
        CHECK(b[0] == 2);
        CHECK(straddle_mirror(copies, 3996, bytes1, 4) && !straddle_mirror(copies, 5000, bytes1, 4));
    }
    {   // no registrations (not path 2): nothing to copy
        const std::vector<uint64_t> off = {0, 4000};
        CHECK(straddling_blobs(off, none, {}, bytes).empty());
    }
    {   // per-layer sizes: index 1's blob is larger
        const std::vector<uint64_t> off = {0, 4000};
        auto by = [](size_t i) { return (uint64_t) (i == 1 ? 50 : 4096); };
        const auto s = straddling_blobs(off, none, two, by);
        CHECK(s.empty());   // 0 + 4096 = 4096 fits the first, 4000 + 50 fits the first
    }
    if (fails == 0) std::printf("blob_straddle_test: OK\n");
    return fails == 0 ? 0 : 1;
}
