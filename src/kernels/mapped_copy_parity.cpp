// src/kernels/mapped_copy_parity.cpp - the copies out of mapped host memory land the same bytes, nothing more.
//
//     build/mapped_copy_parity
//
// fetch_blobs (the PCIe share of a layer's missed experts, staged into VRAM), copy_rows_from_mapped (the CPU pool's
// rows of a verify window, the GPU's own rows zeroed) and copy_i32_from_mapped (plans, steps, positions) against the
// host's bytes: every source alignment a blob can have in the arena (16 B steps through a 128 B line), blob sizes
// below, at and past a chunk, counts of 0 to the capacity, and a guard past the copied range that must stay untouched.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    ++failures;
    return false;
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::printf("no CUDA device\n");
        return 77;
    }
    const size_t host_bytes = 48ull << 20;
    uint8_t* host = nullptr;
    uint8_t* alias = nullptr;
    if (!ck(cudaHostAlloc((void**) &host, host_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc") ||
        !ck(cudaHostGetDevicePointer((void**) &alias, host, 0), "cudaHostGetDevicePointer"))
        return 1;
    std::mt19937_64 rng(20261007);
    for (size_t i = 0; i < host_bytes; i += 8) {
        const uint64_t v = rng();
        std::memcpy(host + i, &v, 8);
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    constexpr uint8_t kGuard = 0xA5;

    // ---- fetch_blobs: n blobs of `bb` bytes from scattered 16 B-aligned sources into dst + k * bb
    {
        constexpr int cap = 16;
        const int64_t sizes[] = {16, 48, 112, 128, 144, 16400, 65536 + 16, 2764816, 3584016};
        const int counts[] = {0, 1, 2, 5, 16};
        unsigned long long* d_ptr = nullptr;
        int32_t* d_n = nullptr;
        uint8_t* d_dst = nullptr;
        const size_t dst_bytes = (size_t) cap * 3584016 + 4096;
        ck(cudaMalloc(&d_ptr, cap * sizeof(unsigned long long)), "malloc ptr");
        ck(cudaMalloc(&d_n, sizeof(int32_t)), "malloc n");
        ck(cudaMalloc(&d_dst, dst_bytes), "malloc dst");
        std::vector<uint8_t> got(dst_bytes);
        int cases = 0;
        for (int64_t bb : sizes) {
            for (int n : counts) {
                // sources: distinct slots of the host buffer, each shifted by 16 * (k % 8) bytes, so the blobs start at
                // every 16 B offset within a 128 B line (the arena's blobs are k * blob_bytes from a 2 MB-aligned base)
                std::vector<unsigned long long> ptr(cap, 0);
                std::vector<size_t> off(cap, 0);
                const size_t slot = ((size_t) bb + 128 + 4095) / 4096 * 4096;
                const size_t slots = host_bytes / slot;
                for (int k = 0; k < n; ++k) {
                    off[(size_t) k] = (size_t) (rng() % slots) * slot + 16 * (size_t) (k % 8);
                    ptr[(size_t) k] = (unsigned long long) (alias + off[(size_t) k]);
                }
                ck(cudaMemcpy(d_ptr, ptr.data(), cap * sizeof(unsigned long long), cudaMemcpyHostToDevice), "ptr");
                ck(cudaMemcpy(d_n, &n, sizeof(int32_t), cudaMemcpyHostToDevice), "n");
                ck(cudaMemset(d_dst, kGuard, dst_bytes), "memset");
                strata::kernels::fetch_blobs(d_ptr, d_n, d_dst, bb, cap, s);
                ck(cudaStreamSynchronize(s), "fetch_blobs");
                ck(cudaMemcpy(got.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "back");
                bool ok = true;
                for (int k = 0; k < n && ok; ++k)
                    ok = std::memcmp(got.data() + (size_t) k * (size_t) bb, host + off[(size_t) k], (size_t) bb) == 0;
                for (size_t i = (size_t) n * (size_t) bb; i < dst_bytes && ok; ++i) ok = got[i] == kGuard;
                if (!ok) {
                    std::printf("fetch_blobs: blob %lld B, %d blobs: wrong bytes\n", (long long) bb, n);
                    ++failures;
                }
                ++cases;
            }
        }
        std::printf("fetch_blobs: %d cases\n", cases);
        // STRATA_ADAPT_FETCH: dst2[k] != 0 sends blob k to that address (a cache slot) instead of dst + k * bb, and
        // rebase_ptrs points the group there; the staging place it skipped stays untouched
        {
            unsigned long long* d_dst2 = nullptr;
            uint8_t* d_slots = nullptr;
            ck(cudaMalloc(&d_dst2, cap * sizeof(unsigned long long)), "malloc dst2");
            ck(cudaMalloc(&d_slots, dst_bytes), "malloc slots");
            std::vector<uint8_t> got2(dst_bytes);
            int cases2 = 0;
            for (int64_t bb : {(int64_t) 144, (int64_t) 2764816, (int64_t) 3584016}) {
                for (int n : {1, 5, 16}) {
                    std::vector<unsigned long long> ptr(cap, 0), dst2(cap, 0);
                    std::vector<size_t> off(cap, 0);
                    const size_t slot = ((size_t) bb + 128 + 4095) / 4096 * 4096;
                    const size_t slots = host_bytes / slot;
                    for (int k = 0; k < n; ++k) {
                        off[(size_t) k] = (size_t) (rng() % slots) * slot + 16 * (size_t) (k % 8);
                        ptr[(size_t) k] = (unsigned long long) (alias + off[(size_t) k]);
                        if (k % 2 == 0) dst2[(size_t) k] = (unsigned long long) (d_slots + (size_t) (cap - 1 - k) * (size_t) bb);
                    }
                    ck(cudaMemcpy(d_ptr, ptr.data(), cap * sizeof(unsigned long long), cudaMemcpyHostToDevice), "ptr");
                    ck(cudaMemcpy(d_dst2, dst2.data(), cap * sizeof(unsigned long long), cudaMemcpyHostToDevice), "dst2");
                    ck(cudaMemcpy(d_n, &n, sizeof(int32_t), cudaMemcpyHostToDevice), "n");
                    ck(cudaMemset(d_dst, kGuard, dst_bytes), "memset");
                    ck(cudaMemset(d_slots, kGuard, dst_bytes), "memset slots");
                    strata::kernels::fetch_blobs(d_ptr, d_n, d_dst, bb, cap, s, d_dst2);
                    strata::kernels::rebase_ptrs(d_ptr, d_n, d_dst, bb, s, d_dst2);
                    ck(cudaStreamSynchronize(s), "fetch_blobs dst2");
                    ck(cudaMemcpy(got.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "back");
                    ck(cudaMemcpy(got2.data(), d_slots, dst_bytes, cudaMemcpyDeviceToHost), "back slots");
                    std::vector<unsigned long long> rp(cap, 0);
                    ck(cudaMemcpy(rp.data(), d_ptr, cap * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "back ptr");
                    bool ok = true;
                    for (int k = 0; k < n && ok; ++k) {
                        const size_t at = (size_t) (k % 2 == 0 ? cap - 1 - k : k) * (size_t) bb;
                        const uint8_t* where = (k % 2 == 0 ? got2.data() : got.data()) + at;
                        const uint8_t* skipped = (k % 2 == 0 ? got.data() + (size_t) k * (size_t) bb : nullptr);
                        ok = std::memcmp(where, host + off[(size_t) k], (size_t) bb) == 0 &&
                             rp[(size_t) k] == (k % 2 == 0 ? dst2[(size_t) k]
                                                           : (unsigned long long) (d_dst + (size_t) k * (size_t) bb));
                        for (size_t i = 0; skipped != nullptr && i < (size_t) bb && ok; ++i) ok = skipped[i] == kGuard;
                    }
                    if (!ok) {
                        std::printf("fetch_blobs dst2: blob %lld B, %d blobs: wrong bytes or pointers\n", (long long) bb, n);
                        ++failures;
                    }
                    ++cases2;
                }
            }
            std::printf("fetch_blobs into slots (dst2): %d cases\n", cases2);
            cudaFree(d_dst2);
            cudaFree(d_slots);
        }
        cudaFree(d_ptr);
        cudaFree(d_n);
        cudaFree(d_dst);
    }

    // ---- copy_rows_from_mapped: rows x width floats, the listed hit rows +0.0, the rest copied
    {
        const int64_t row_counts[] = {1, 10, 40, 60};
        const int64_t widths[] = {4, 2560, 2564, 4096};
        int32_t* d_hit = nullptr;
        int32_t* d_count = nullptr;
        float* d_dst = nullptr;
        const size_t dst_floats = 60 * 4096 + 1024;
        ck(cudaMalloc(&d_hit, 64 * sizeof(int32_t)), "malloc hit");
        ck(cudaMalloc(&d_count, sizeof(int32_t)), "malloc count");
        ck(cudaMalloc(&d_dst, dst_floats * sizeof(float)), "malloc dst");
        std::vector<float> got(dst_floats);
        int cases = 0;
        for (int64_t rows : row_counts) {
            for (int64_t w : widths) {
                for (int every : {0, 3}) {   // no hit rows, a third of them
                    std::vector<int32_t> hit;
                    if (every > 0)
                        for (int64_t r = 1; r < rows; r += every) hit.push_back((int32_t) r);
                    const int32_t nh = (int32_t) hit.size();
                    if (nh > 0) ck(cudaMemcpy(d_hit, hit.data(), (size_t) nh * 4, cudaMemcpyHostToDevice), "hit");
                    ck(cudaMemcpy(d_count, &nh, 4, cudaMemcpyHostToDevice), "count");
                    ck(cudaMemset(d_dst, kGuard, dst_floats * sizeof(float)), "memset");
                    const size_t off = (size_t) (rng() % 1024) * 16;   // 16 B aligned, as the kernel requires
                    strata::kernels::copy_rows_from_mapped(d_dst, (const float*) (alias + off), rows, w, d_hit, d_count, s);
                    ck(cudaStreamSynchronize(s), "copy_rows_from_mapped");
                    ck(cudaMemcpy(got.data(), d_dst, dst_floats * sizeof(float), cudaMemcpyDeviceToHost), "back");
                    bool ok = true;
                    for (int64_t r = 0; r < rows && ok; ++r) {
                        bool is_hit = false;
                        for (int32_t h : hit) is_hit |= h == r;
                        const uint8_t* g = (const uint8_t*) (got.data() + r * w);
                        if (is_hit) {
                            for (int64_t i = 0; i < w * 4 && ok; ++i) ok = g[i] == 0;
                        } else {
                            ok = std::memcmp(g, host + off + (size_t) (r * w) * 4, (size_t) w * 4) == 0;
                        }
                    }
                    const uint8_t* gb = (const uint8_t*) got.data();
                    for (size_t i = (size_t) (rows * w) * 4; i < dst_floats * sizeof(float) && ok; ++i) ok = gb[i] == kGuard;
                    if (!ok) {
                        std::printf("copy_rows_from_mapped: %lld rows x %lld, %d hit rows: wrong bytes\n", (long long) rows,
                                    (long long) w, (int) nh);
                        ++failures;
                    }
                    ++cases;
                }
            }
        }
        std::printf("copy_rows_from_mapped: %d cases\n", cases);
        cudaFree(d_hit);
        cudaFree(d_count);
        cudaFree(d_dst);
    }

    // ---- copy_i32_from_mapped: n words from 4 B-aligned sources
    {
        const int64_t ns[] = {1, 5, 127, 128, 129, 488, 511, 512, 513, 1000, 3001};
        int32_t* d_dst = nullptr;
        const size_t dst_words = 4096;
        ck(cudaMalloc(&d_dst, dst_words * 4), "malloc dst");
        std::vector<int32_t> got(dst_words);
        int cases = 0;
        for (int64_t n : ns) {
            for (int shift = 0; shift < 4; ++shift) {
                const size_t off = (size_t) (rng() % 4096) * 16 + 4 * (size_t) shift;
                ck(cudaMemset(d_dst, kGuard, dst_words * 4), "memset");
                strata::kernels::copy_i32_from_mapped(d_dst, (const int32_t*) (alias + off), n, s);
                ck(cudaStreamSynchronize(s), "copy_i32_from_mapped");
                ck(cudaMemcpy(got.data(), d_dst, dst_words * 4, cudaMemcpyDeviceToHost), "back");
                bool ok = std::memcmp(got.data(), host + off, (size_t) n * 4) == 0;
                const uint8_t* gb = (const uint8_t*) got.data();
                for (size_t i = (size_t) n * 4; i < dst_words * 4 && ok; ++i) ok = gb[i] == kGuard;
                if (!ok) {
                    std::printf("copy_i32_from_mapped: %lld words at +%zu: wrong bytes\n", (long long) n, off);
                    ++failures;
                }
                ++cases;
            }
        }
        std::printf("copy_i32_from_mapped: %d cases\n", cases);
        cudaFree(d_dst);
    }

    cudaStreamDestroy(s);
    cudaFreeHost(host);
    std::printf(failures == 0 ? "mapped_copy_parity: PASS\n" : "mapped_copy_parity: %d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
