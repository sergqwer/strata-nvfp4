// tools/vision/own/engine_reader.h - the picture reader inside the engine (strata --vision-gpu <mmproj>): our own tower on the engine's GPU, in
// device memory the engine lends it for one picture (its borrowed expert slots), with the weights kept in page-locked RAM between pictures.
//
//   EngineReader r;  r.open(mmproj, max_tokens, min_tokens, err);   // the file's header and geometry (cheap)
//   r.stage(err);                                                   // the weights into pinned RAM (once: at the start or the first picture)
//   r.prepare(image, prep, err);  need = r.need_for(prep);          // the picture, and the room it needs
//   r.encode(base, bytes, prep, out, n, nx, ny, t, err);            // upload, encode, write the SVE1 file; the room is the caller's again after
// The output is the reader program's: int32 {'SVE1', n_tokens, nx, ny, n_embd} then float32 [n_tokens][n_embd].
#pragma once

#include "gguf_lite.h"
#include "image_prep.h"
#include "vit_geometry.h"
#include "vit_gpu.h"

#include <memory>
#include <string>

namespace strata_vit {

struct EngineReaderTimes {
    double upload_ms = 0, encode_ms = 0, write_ms = 0;
};

class EngineReader {
public:
    bool open(const std::string& mmproj, int max_tokens, int min_tokens, std::string& err);
    bool opened() const { return opened_; }
    bool staged() const { return staged_; }
    bool stage(std::string& err);
    size_t staged_bytes() const { return gpu_ ? gpu_->staged_bytes() : 0; }
    bool staged_pinned() const { return gpu_ && gpu_->staged_pinned(); }
    int out_dim() const { return geo_.out_dim; }
    // read and prepare a picture; refuses one over the limit (by more than an eighth, as the reader program)
    bool prepare(const std::string& image, PreparedImage& prep, std::string& err) const;
    size_t need_for(const PreparedImage& prep);
    bool encode(void* base, size_t bytes, const PreparedImage& prep, const std::string& out, int& n, int& nx, int& ny, EngineReaderTimes& t,
                std::string& err);

private:
    std::string mmproj_;
    int max_tokens_ = 0, min_tokens_ = 0;
    bool opened_ = false, staged_ = false;
    VitGeometry geo_;
    std::unique_ptr<VitGpu> gpu_;
};

}  // namespace strata_vit
