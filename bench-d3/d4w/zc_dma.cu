// zc_dma.cu (D4 microbench): the PCIe share's staging copy at the engine's sizes, zero-copy (fetch_blobs_kernel, the
// engine's code) vs the copy engine (cudaMemcpyAsync per blob), with and without the CPU streaming DRAM and with and
// without a background copy-engine stream (the adaptive tier's refills / KV streaming).
// Each op: n blobs of 2.76 MB from scattered 16 B-aligned sources in a 2 GiB pinned arena (a fresh source each rep).
// Prints per case the median us per op, GB/s, and the CPU threads' read rate meanwhile.
// usage: zc_dma.exe [cpu_threads=15] [ms_per_case=400]
#include <cuda_runtime.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int kFetchU = 4;   // = verify_kernels.cu
__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per, const unsigned long long* __restrict__ dst2) {
    const long long ce = (long long) blockDim.x * kFetchU;
    const long long cpb = (per + ce - 1) / ce;
    const long long total = (long long) *n * cpb;
    for (long long c = blockIdx.x; c < total; c += gridDim.x) {
        const long long k = c / cpb, ci = c - k * cpb;
        const uint4* s = (const uint4*) src[k];
        uint4* d = dst2 != nullptr && dst2[k] != 0 ? (uint4*) dst2[k] : dst + k * per;
        const int h = (int) ((8 - (((unsigned long long) s >> 4) & 7)) & 7);
        if (ci == 0 && (long long) threadIdx.x < h && (long long) threadIdx.x < per) d[threadIdx.x] = s[threadIdx.x];
        const long long o0 = ci * ce + h + threadIdx.x;
        uint4 v[kFetchU];
#pragma unroll
        for (int j = 0; j < kFetchU; ++j) { const long long o = o0 + (long long) j * blockDim.x; if (o < per) v[j] = s[o]; }
#pragma unroll
        for (int j = 0; j < kFetchU; ++j) { const long long o = o0 + (long long) j * blockDim.x; if (o < per) d[o] = v[j]; }
    }
}

int main(int argc, char** argv) {
    const int nthr = argc > 1 ? std::atoi(argv[1]) : 15;
    const int case_ms = argc > 2 ? std::atoi(argv[2]) : 400;
    const size_t blob = 2760000 / 16 * 16;            // the pack's largest blob, 2.76 MB
    const size_t arena = 2ull << 30;
    uint8_t* h = nullptr;
    CK(cudaHostAlloc((void**) &h, arena, cudaHostAllocMapped));
    std::memset(h, 7, arena);
    uint8_t* hd = nullptr;
    CK(cudaHostGetDevicePointer((void**) &hd, h, 0));
    uint8_t *stage, *bg_dst;
    CK(cudaMalloc(&stage, 8 * blob));
    CK(cudaMalloc(&bg_dst, 4 * blob));
    unsigned long long* d_src; int32_t* d_n;
    CK(cudaMalloc(&d_src, 64 * 8));
    CK(cudaMalloc(&d_n, 4));
    cudaStream_t s, sbg;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sbg, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // CPU DRAM load: AVX2 streaming reads over a private 1 GiB buffer per run (not the arena)
    const size_t cb = 1ull << 30;
    std::vector<uint8_t> cbuf(cb, 1);
    // sources: slot r of the arena (fresh each rep), +16*r misalignment like the arena's blobs
    const size_t nslots = (arena - 64 * 16) / blob - 1;
    size_t rr = 0;
    auto run_case = [&](const char* name, int nb, int mode /*0 zc, 1 dma*/, int blocks, bool load, bool bg) {
        std::atomic<bool> stop{false};
        std::atomic<long long> cbytes{0};
        std::vector<std::thread> th;
        if (load)
            for (int t = 0; t < nthr; ++t) th.emplace_back([&, t] {
                const __m256i* p = (const __m256i*) (cbuf.data() + (cb / nthr) * t);
                const size_t n = cb / nthr / 32;
                __m256i acc = _mm256_setzero_si256();
                while (!stop.load(std::memory_order_relaxed)) {
                    for (size_t i = 0; i + 4 <= n; i += 4)
                        acc = _mm256_xor_si256(acc, _mm256_xor_si256(_mm256_xor_si256(_mm256_load_si256(p + i), _mm256_load_si256(p + i + 1)),
                                                                     _mm256_xor_si256(_mm256_load_si256(p + i + 2), _mm256_load_si256(p + i + 3))));
                    cbytes += (long long) n * 32;
                }
                if (_mm256_testz_si256(acc, acc) == 2) std::printf("x");
            });
        std::atomic<bool> bstop{false};
        std::thread bth;
        if (bg) bth = std::thread([&] {   // a refill-like H2D stream: 2.76 MB copies back to back, ~4 in flight
            size_t q = nslots / 2;
            while (!bstop.load()) {
                for (int i = 0; i < 4; ++i) { cudaMemcpyAsync(bg_dst + (size_t) i * blob, h + (q % nslots) * blob, blob, cudaMemcpyHostToDevice, sbg); q += 7; }
                cudaStreamSynchronize(sbg);
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const long long b0 = cbytes.load();
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> us;
        unsigned long long src[64];
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < case_ms) {
            for (int k = 0; k < nb; ++k) { src[k] = (unsigned long long) ((mode == 0 ? hd : h) + (rr % nslots) * blob + 16 * (rr % 7 + 1)); rr += 13; }
            if (mode == 0) { CK(cudaMemcpyAsync(d_src, src, nb * 8, cudaMemcpyHostToDevice, s)); CK(cudaMemcpyAsync(d_n, &nb, 4, cudaMemcpyHostToDevice, s)); }
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(e0, s));
            if (mode == 0) fetch_blobs_kernel<<<blocks, 256, 0, s>>>(d_src, d_n, (uint4*) stage, (long long) (blob / 16), nullptr);
            else for (int k = 0; k < nb; ++k) CK(cudaMemcpyAsync(stage + (size_t) k * blob, (const void*) src[k], blob, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            us.push_back(ms * 1000.0f);
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const long long b1 = cbytes.load();
        stop = true; for (auto& t : th) t.join();
        bstop = true; if (bth.joinable()) bth.join();
        std::sort(us.begin(), us.end());
        const float med = us[us.size() / 2], p90 = us[us.size() * 9 / 10];
        std::printf("%-10s n=%d load=%-3s bg=%d: median %7.1f us (p90 %7.1f) = %5.1f GB/s | CPU %5.1f GB/s | %zu reps\n", name, nb,
                    load ? "yes" : "no", (int) bg, med, p90, nb * blob / (med * 1e-6) / 1e9, load ? (b1 - b0) / sec / 1e9 : 0.0, us.size());
        std::fflush(stdout);
    };
    // CPU alone for reference
    {
        std::atomic<bool> stop{false}; std::atomic<long long> cbytes{0}; std::vector<std::thread> th;
        for (int t = 0; t < nthr; ++t) th.emplace_back([&, t] {
            const __m256i* p = (const __m256i*) (cbuf.data() + (cb / nthr) * t); const size_t n = cb / nthr / 32; __m256i acc = _mm256_setzero_si256();
            while (!stop.load(std::memory_order_relaxed)) { for (size_t i = 0; i < n; ++i) acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i)); cbytes += (long long) n * 32; }
            if (_mm256_testz_si256(acc, acc) == 2) std::printf("x");
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const long long b0 = cbytes.load(); const auto t0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(case_ms));
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const long long b1 = cbytes.load(); stop = true; for (auto& t : th) t.join();
        std::printf("CPU alone, %d threads: %.1f GB/s\n", nthr, (b1 - b0) / sec / 1e9);
    }
    for (int round = 0; round < 2; ++round)
        for (bool load : {false, true})
            for (bool bg : {false, true})
                for (int nb : {1, 2, 3}) {
                    run_case("zc16", nb, 0, 16, load, bg);
                    run_case("zc64", nb, 0, 64, load, bg);
                    run_case("zc170", nb, 0, 170, load, bg);
                    run_case("dma", nb, 1, 0, load, bg);
                }
    return 0;
}
