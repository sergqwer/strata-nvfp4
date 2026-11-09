// src/core/blob_straddle_test.cpp - strata-nvfp4 (fork issue #20): straddling_blobs finds the blobs no registration
// holds whole (host only, no GPU).
#include "strata/core/blob_straddle.hpp"

#include <cstdio>

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
    {   // a blob past the last registration and one in a gap between registrations
        const std::vector<std::pair<uint64_t, uint64_t>> gap = {{0, 1000}, {2000, 3000}};
        const std::vector<uint64_t> off = {950, 1200, 2000, 2950, 3100};
        const auto s = straddling_blobs(off, none, gap, bytes);
        CHECK(s.size() == 4 && s[0] == 0 && s[1] == 1 && s[2] == 3 && s[3] == 4);
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
