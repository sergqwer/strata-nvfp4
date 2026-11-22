// tools/vision/own/engine_reader.cpp - see engine_reader.h
#include "engine_reader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace strata_vit {

bool EngineReader::open(const std::string& mmproj, int max_tokens, int min_tokens, std::string& err) {
    Gguf gg;
    if (!gg.open(mmproj, err) || !vit_geometry_from_gguf(gg, geo_, err) || !vit_geometry_supported(geo_, err)) return false;
    if (geo_.head_dim > 96) { err = "a head size above 96 is not handled by the GPU attention"; return false; }
    if (max_tokens > 0 && min_tokens > max_tokens) { err = "--vision-min-tokens is above --vision-max-tokens"; return false; }
    mmproj_ = mmproj;
    max_tokens_ = max_tokens;
    min_tokens_ = min_tokens;
    opened_ = true;
    return true;
}

bool EngineReader::stage(std::string& err) {
    if (staged_) return true;
    if (!opened_) { err = "no vision encoder file"; return false; }
    Gguf gg;
    if (!gg.open(mmproj_, err)) return false;
    std::unique_ptr<VitGpu> g(new VitGpu());
    if (!g->stage(gg, geo_, 0, err)) return false;
    gpu_ = std::move(g);
    staged_ = true;
    return true;
}

bool EngineReader::prepare(const std::string& image, PreparedImage& prep, std::string& err) const {
    ImageU8 src;
    if (image == "@warm") {   // the engine's warm-up picture (--vision-gpu-preload): grey, 256 x 256
        src.w = src.h = 256;
        src.rgb.assign((size_t) 256 * 256 * 3, 128);
    } else if (!image_load_file(image, src, err)) {
        err = "cannot read the image " + image;
        return false;
    }
    if (!prepare_image(src, geo_, min_tokens_, max_tokens_, prep, err)) { err = "the image could not be preprocessed (" + err + ")"; return false; }
    const int n_expect = (prep.w / geo_.align()) * (prep.h / geo_.align());
    if (max_tokens_ > 0 && n_expect > max_tokens_ + max_tokens_ / 8) {
        err = "the picture became " + std::to_string(n_expect) + " image tokens, over the limit of " + std::to_string(max_tokens_) +
              " (vision.max_tokens); send a smaller picture or raise the limit";
        return false;
    }
    return true;
}

size_t EngineReader::need_for(const PreparedImage& prep) {
    if (!gpu_) return 0;
    return gpu_->arena_need((prep.w / geo_.patch) * (prep.h / geo_.patch));
}

bool EngineReader::encode(void* base, size_t bytes, const PreparedImage& prep, const std::string& out, int& n, int& nx, int& ny,
                          EngineReaderTimes& t, std::string& err) {
    using clock = std::chrono::steady_clock;
    if (!gpu_) { err = "the vision weights are not loaded"; return false; }
    const auto t0 = clock::now();
    if (!gpu_->bind(base, bytes, err)) return false;
    const auto t1 = clock::now();
    std::vector<float> rows;
    const bool ok = gpu_->encode(prep, rows, nx, ny, err);
    gpu_->unbind();
    const auto t2 = clock::now();
    t.upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    t.encode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (!ok) return false;
    n = nx * ny;
    if (out == "-") return true;   // the warm-up: nothing to write
    FILE* f = std::fopen(out.c_str(), "wb");
    if (!f) { err = "cannot write " + out; return false; }
    const int32_t hdr[5] = {0x31455653, n, nx, ny, geo_.out_dim};
    const bool wr = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                    std::fwrite(rows.data(), sizeof(float) * (size_t) geo_.out_dim, (size_t) n, f) == (size_t) n;
    std::fclose(f);
    if (!wr) { err = "short write to " + out; return false; }
    t.write_ms = std::chrono::duration<double, std::milli>(clock::now() - t2).count();
    return true;
}

}  // namespace strata_vit
