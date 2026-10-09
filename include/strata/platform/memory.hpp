// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// #243, Windows: the GPU's shared (non-local) memory budget and this process's use of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) for the adapter whose LUID is the
/// 8 bytes at `luid` (cudaDeviceProp::luid).  Page-locked host memory the GPU maps is charged there.  False (and
/// `why` says so) when the query is not possible - always elsewhere than Windows.
bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// The fork's auto low-RAM rule (generate.cpp; setup.py's NVFP4_ALL_RAM_GB is the same number): on below 92 GiB
/// installed (total_physical_memory), so a 96 GB PC (93.4-95.6 GiB listed) keeps every expert in RAM and a 64 GB one
/// does not.  Measured with 88 GiB left to the engine (ram88): the arena 13.1 ms a round, 21 GiB free; low-RAM 20.3.
constexpr uint64_t kLowRamBelowGib = 92;
inline bool low_ram_auto(uint64_t installed) { return installed > 0 && installed < (kLowRamBelowGib << 30); }

/// Windows' page files.  The commit limit is RAM + page files, and under WDDM the GPU's allocations are charged to it
/// too (~30 GiB on a 32 GB card): the NVFP4 models commit ~100 GiB.  Sizes in Windows' dialog units ("MB" = MiB), so
/// a file typed as 64000 counts 64000.  Only what is there for sure counts: a page file Windows grows on demand
/// (system-managed, or an initial size below the maximum) may not grow in time while WDDM charges the VRAM (upstream
/// issue #60: "System managed" and 4096-32768 MB still failed, a fixed 64 GB worked).  Below kPageFileWarnMb in all,
/// the engine and setup warn; kPageFileAdviseMb is what they advise, as a fixed size (initial = maximum).
constexpr uint64_t kPageFileWarnMb = 60000, kPageFileAdviseMb = 64000;

/// One PagingFiles entry ("C:\pagefile.sys 16000 64000", "C:\pagefile.sys 0 0" or "?:\pagefile.sys"): the drive
/// ('?' = every drive automatic) and its sizes, -1 when Windows manages the size.  False for an empty line.
inline bool parse_paging_file(const std::string& line, char& drive, int64_t& initial_mb, int64_t& max_mb) {
    size_t a = line.find_first_not_of(' ');
    if (a == std::string::npos || line.size() < a + 2 || line[a + 1] != ':') return false;
    drive = line[a] >= 'a' && line[a] <= 'z' ? (char) (line[a] - 'a' + 'A') : line[a];
    initial_mb = max_mb = -1;
    const size_t sp = line.find(' ', a);
    if (sp != std::string::npos) {
        long long i = -1, m = -1;
        if (std::sscanf(line.c_str() + sp, "%lld %lld", &i, &m) == 2 && m > 0) {
            initial_mb = i;
            max_mb = m;
        }
    }
    return true;
}

/// What one page file has for sure (MB): its size now, or its configured initial size when that is larger (a size
/// raised in the dialog), the initial size no more than the file is now plus the volume's free space.  Never the
/// maximum, nor what a system-managed file (`initial_mb` -1) may grow to.
inline uint64_t page_file_guaranteed_mb(int64_t initial_mb, uint64_t free_mb, uint64_t current_mb) {
    uint64_t want = initial_mb > 0 ? (uint64_t) initial_mb : 0;
    if (want > current_mb + free_mb) want = current_mb + free_mb;
    return want > current_mb ? want : current_mb;
}

/// Whether Windows grows a page file on demand: system-managed (`max_mb` -1), or an initial size below the maximum.
inline bool page_file_grows(int64_t initial_mb, int64_t max_mb) { return max_mb <= 0 || initial_mb < max_mb; }

/// The configured page files (HKLM\...\Memory Management\PagingFiles, what the Virtual memory dialog writes), each
/// counted by page_file_guaranteed_mb.  `known` false off Windows or when the setting cannot be read.
struct PageFiles {
    bool known = false;
    uint64_t total_mb = 0;   ///< what they have for sure, every drive summed
    bool grows = false;      ///< one of them is set to grow on demand (page_file_grows)
    std::string detail;      ///< "C: 64000 MB, D: system-managed, now 6000 MB", for the log
};
PageFiles page_files();

/// #357/#577: whether the OS file cache could keep the `read_bytes` the expert files are read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the files (the experts
/// outside its resident RAM copy) and the RAM that copy really holds - not every shard's bytes and the requested
/// budget, which on a 96 GB PC (#577) made the file tier read unbuffered when the cache could keep its reads.
/// #1194: the file tier's reads are a skewed set (the GPU cache and the RAM copy took the hottest experts; the same
/// few of the rest come back token after token), so the cache does not have to hold all `read_bytes` to serve most of
/// them.  With `hot_fraction` < 1 it is enough that the room holds that share of them, but never less than
/// `floor_bytes` (below it the cache holds nothing worth having) and never more than every read.  The defaults ask for
/// every byte, as the Windows rule does.  Measured on Linux (#1194, a Tesla P100 box under 20 to 32 GB cgroups): through
/// the cache beat the unbuffered reads at every room tried, down to 7% of the read bytes (decode +7% to +118%, a third to
/// a sixth of the drive traffic).
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30, double hot_fraction = 1.0, uint64_t floor_bytes = 0) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    uint64_t need = (uint64_t) ((double) read_bytes * hot_fraction);
    if (need < floor_bytes) need = floor_bytes;
    if (need > read_bytes) need = read_bytes;
    return room >= need;
}

/// Whether `advise_willneed` asks the OS for anything: not on Windows, nor with STRATA_READ_AHEAD=0.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  Linux reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

/// What the OS says this process read from storage: `read_bytes` (/proc/self/io: bytes fetched from the block layer on
/// its behalf, page faults on a mapping included, page-cache hits not) and `major_faults` (/proc/self/stat).  Linux
/// only; `valid` is false elsewhere.  Take one at the start of a request and one at its end.
struct ProcIo {
    bool valid = false;
    uint64_t read_bytes = 0, major_faults = 0;
};
ProcIo proc_io_sample();

}  // namespace strata::platform
