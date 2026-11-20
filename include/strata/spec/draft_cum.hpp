// include/strata/spec/draft_cum.hpp - the cumulative MTP draft-length rule (--mtp-cum), ported from Basalt
// (include/basalt/core/draft_cum.hpp, MIT, (c) 2026 Jesús Gulfo, github.com/jesdga95/basalt).
//
// The MTP chain keeps draft j while the estimated chance that drafts 0..j are ALL accepted stays at or above the
// threshold: the product over i <= j of acc(p_i), where acc maps a draft probability to the observed acceptance rate of
// drafts in that probability bin.  A middling first draft is cheap to verify and still often right, so it is proposed;
// a long run of merely high probabilities stops once the joint chance decays.  The drafter's chain (MtpDrafter::draft)
// and the window rule (window_rule.hpp) evaluate the same products in the same order with the same threshold, so they
// agree exactly.  Only what is proposed changes: the verify window decides every token.
#pragma once

#include <algorithm>
#include <cmath>

namespace strata::spec {

/// The acceptance rate of a draft by its probability bin (Basalt's table).
inline float draft_cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}

/// Whether draft j stays in the window: prod_{i <= j} acc(probs[i]) >= th.
inline bool draft_cum_keep(const float* probs, int j, float th) {
    float c = 1.0f;
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);
    return c >= th;
}

/// Per-request threshold.  It falls from the base toward `lo` while the request's drafts are accepted more often than
/// the bin table predicts (long windows pay then) and returns to the base when they are not.  The observed and expected
/// accepted counts decay per window; their ratio sets the threshold along a linear ramp.  A window's threshold is fixed
/// before its drafts are made and shared by the chain and the window rule.
struct DraftCumAdapt {
    float base = 0.35f, lo = 0.25f;
    double obs = 0.0, exp = 0.0;
    float cur = 0.35f;

    void reset(float b, float l) { base = b; lo = l; obs = exp = 0.0; cur = b; }
    float th() const { return cur; }

    /// One verified window of the MTP chain: `n` drafts were offered with probabilities probs[0..n), `acc` accepted.
    void observe(const float* probs, int n, int acc) {
        if (n <= 0) return;
        double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) { c *= (double) draft_cum_acc(probs[j]); e += c; }
        constexpr double kDecay = 0.97, kRlo = 0.9, kRhi = 1.1;
        obs = obs * kDecay + (double) acc;
        exp = exp * kDecay + e;
        if (exp < 0.5) return;
        const double t = std::fmin(std::fmax((obs / exp - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float) t * (std::min(lo, base) - base);
    }
};

}  // namespace strata::spec
