// The auto layer split's stage-size rule (#1616): no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/split_rules.hpp"

using namespace strata::program::split_rules;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the reporter's 4-GPU pick on a 48-layer model: layers 0-1, 2, 3-19, 20-47 -> cuts 2,3,20
    CHECK(smallest_stage({2, 3, 20}, 48) == 1);
    CHECK(has_tiny_stage({2, 3, 20}, 48));
    // the manual split that fixed it: 4,8,24 is not a cut list of 4 stages; as cuts 4,8,24 -> 4,4,16,24
    CHECK(smallest_stage({4, 8, 24}, 48) == 4);
    CHECK(!has_tiny_stage({4, 8, 24}, 48));
    // identical cards: the balanced split has no tiny stage, so the rule never fires and the pick stays
    CHECK(!has_tiny_stage({12, 24, 36}, 48));
    CHECK(!has_tiny_stage({24}, 48));
    CHECK(has_tiny_stage({47}, 48));
    CHECK(has_tiny_stage({2}, 48));
    CHECK(!has_tiny_stage({3}, 48));
    CHECK(smallest_stage({}, 48) == 48);
    // the restriction needs the layers for it
    CHECK(restriction_possible(48, 4));
    CHECK(restriction_possible(12, 4));
    CHECK(!restriction_possible(11, 4));
    CHECK(!restriction_possible(48, 1));
    // extras/split-prefill: STRATA_SPLIT_PF_CHUNKS=k cuts a segment into equal chunks of at least min_chunk
    CHECK(pipeline_chunk(32037, 32768, 8, 2048) == 4096);    // 8 chunks: 7 x 4096 + 3365
    CHECK(pipeline_chunk(32037, 31232, 4, 2048) == 8192);    // 4 chunks: 3 x 8192 + 7461 (no 805-token tail)
    CHECK(pipeline_chunk(32037, 31232, 2, 2048) == 16128);   // 2 chunks of ~16K
    CHECK(pipeline_chunk(5000, 32768, 8, 2048) == 2560);     // only 2 chunks fit the 2048 floor
    CHECK(pipeline_chunk(2048, 32768, 8, 2048) == 32768);    // one chunk anyway: unchanged
    CHECK(pipeline_chunk(200000, 32768, 4, 2048) == 32768);  // above k x max: the configured chunk
    CHECK(pipeline_chunk(32037, 32768, 0, 2048) == 32768);   // off
    CHECK(pipeline_chunk(32037, 32768, 1, 2048) == 32768);
    CHECK(pipeline_chunk(32037, 32768, 64, 100) == 512);     // a floor under 256 is 256; k caps the count: 62 x 512 + 293
    if (failed == 0) std::printf("split_rules_test: ok\n");
    return failed ? 1 : 0;
}
