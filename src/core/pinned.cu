// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"
#include "strata/core/pin_budget.hpp"
#include <cstddef>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>

// Loader fix: `fseek`/`ftell` are 32-bit on Windows by default (and the pack is 42.9 GB), and the 64-bit
// spelling is not the same on the two platforms the engine builds for.
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/vfs.h>
#endif
#include <unistd.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#endif

namespace strata::core {

namespace {

constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "STRATA-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);

#ifdef _WIN32
// Shared-section views: MapViewOfFile address -> the section handle, so release() unmaps and closes instead of VirtualFree.
std::mutex g_views_mu;
std::map<void*, HANDLE> g_views;
#endif

#ifndef _WIN32
// MemAvailable and, in a cgroup v2 group with a limit, the room left under it; ~0 when neither can be read
uint64_t free_ram_room() {
    uint64_t room = ~uint64_t{0};
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { room = (uint64_t) kb * 1024u; break; }
        }
        std::fclose(f);
    }
    std::string cg;
    if (std::FILE* f = std::fopen("/proc/self/cgroup", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr)
            if (std::strncmp(line, "0::", 3) == 0) { cg = line + 3; break; }
        std::fclose(f);
    }
    while (!cg.empty() && (cg.back() == '\n' || cg.back() == '\r')) cg.pop_back();
    if (!cg.empty()) {
        unsigned long long lim = 0, cur = 0;
        bool have_lim = false;
        if (std::FILE* f = std::fopen(("/sys/fs/cgroup" + cg + "/memory.max").c_str(), "r")) {
            have_lim = std::fscanf(f, "%llu", &lim) == 1;   // ("max" does not parse: no limit)
            std::fclose(f);
        }
        if (have_lim) {
            if (std::FILE* f = std::fopen(("/sys/fs/cgroup" + cg + "/memory.current").c_str(), "r")) {
                if (std::fscanf(f, "%llu", &cur) != 1) cur = 0;
                std::fclose(f);
            }
            room = std::min<uint64_t>(room, lim > cur ? (uint64_t) (lim - cur) : 0);
        }
    }
    return room;
}
#endif

#ifndef _WIN32
// bytes in free blocks of 2 MiB or more (/proc/buddyinfo, order >= 9): what MADV_HUGEPAGE can get without compaction
uint64_t free_huge_bytes() {
    uint64_t total = 0;
    if (std::FILE* f = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            const char* q = z + 4;
            while (*q == ' ') ++q;
            while (*q != '\0' && *q != ' ') ++q;   // the zone's name
            unsigned long long c[11] = {0};
            int n = 0;
            for (; n < 11; ++n) {
                char* e = nullptr;
                const unsigned long long v = std::strtoull(q, &e, 10);
                if (e == q) break;
                c[n] = v;
                q = e;
            }
            for (int o = 9; o < n; ++o) total += c[o] * (4096ull << o);
        }
        std::fclose(f);
    }
    return total;
}
#endif

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
// A non-empty shared_file instead maps one file whose first 4 KiB identify the pack and whose remaining bytes
// are the resident arena.  This shared-file layout is intended for tmpfs (/dev/shm); hugetlbfs would need
// hugepage-aligned file size and arena offset rather than the 4 KiB header layout used here.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note, const std::string& shared_file,
              uint64_t shared_pack_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;

#ifdef _WIN32
    if (!shared_file.empty()) {
        // The shared arena on Windows is a named, pagefile-backed section (CreateFileMappingW) that every engine of the
        // replicas opens by name: "win:Local\\strata-replicas-<server pid>" (a name without the prefix is used as it is).
        // The first engine creates it at the full size, the others open it; it lives while any process holds a handle or
        // a view, so a replica that restarts joins the same pages.  The first 4 KiB are the same header as on Linux (the
        // pack hash, the size, the load state word); a new section is all zeros, so "no magic" means "nobody has been
        // here" - two engines that both find it empty write the same header.
        if (shared_pack_hash == 0) {
            note = "shared arena requires a nonzero pack hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = bytes + kSharedArenaHeaderBytes;
        std::string nm = shared_file.rfind("win:", 0) == 0 ? shared_file.substr(4) : shared_file;
        std::wstring wname(nm.begin(), nm.end());
        // the section is committed memory (RAM plus pagefile) from the moment it exists: say so before taking it
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms) && file_bytes + (2ull << 30) > ms.ullAvailPageFile) {
            HANDLE probe = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());   // (already there: nothing new is committed)
            if (probe == nullptr) {
                note = "not enough free memory (RAM + pagefile) for the shared arena " + nm + ": it needs " +
                       std::to_string((unsigned long long) (file_bytes >> 20)) + " MiB and " +
                       std::to_string((unsigned long long) (ms.ullAvailPageFile >> 20)) + " MiB are available (2 GiB are kept back)";
                return nullptr;
            }
            CloseHandle(probe);
        }
        HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD) (file_bytes >> 32),
                                      (DWORD) (file_bytes & 0xffffffffu), wname.c_str());
        if (h == nullptr) {
            note = "cannot create shared arena section " + nm + ": Windows error " + std::to_string((unsigned long long) GetLastError());
            return nullptr;
        }
        void* view = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 0);   // (0: the whole section, as big as it was created)
        if (view == nullptr) {
            const DWORD e = GetLastError();
            CloseHandle(h);
            note = "cannot map shared arena section " + nm + ": Windows error " + std::to_string((unsigned long long) e);
            return nullptr;
        }
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(view, &mbi, sizeof mbi) == 0 || (uint64_t) mbi.RegionSize < file_bytes) {
            UnmapViewOfFile(view);
            CloseHandle(h);
            note = "shared arena section " + nm + " is smaller than the " + std::to_string((unsigned long long) file_bytes) +
                   " B this engine needs (including its header)";
            return nullptr;
        }
        auto* hdr = static_cast<SharedArenaHeader*>(view);
        bool empty = true;
        for (char c : hdr->magic) empty = empty && c == 0;
        if (empty) {
            SharedArenaHeader fresh{};
            std::memcpy(fresh.magic, kSharedArenaMagic, sizeof fresh.magic);
            fresh.version = 1;
            fresh.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            fresh.arena_bytes = bytes;
            fresh.pack_hash = shared_pack_hash;
            // fields after the magic first, the magic last: an engine that sees the magic sees a complete header (the
            // state word, reserved[0], stays zero)
            std::memcpy(reinterpret_cast<char*>(hdr) + sizeof fresh.magic, reinterpret_cast<char*>(&fresh) + sizeof fresh.magic,
                        sizeof fresh - sizeof fresh.magic);
            MemoryBarrier();
            std::memcpy(hdr->magic, fresh.magic, sizeof fresh.magic);
        } else {
            // (an engine that arrives in the instant between another one's header fields and its magic never sees a
            // half header: the magic is written last)
            if (std::memcmp(hdr->magic, kSharedArenaMagic, sizeof hdr->magic) != 0 || hdr->version != 1 ||
                hdr->header_bytes != kSharedArenaHeaderBytes || hdr->arena_bytes != bytes) {
                UnmapViewOfFile(view);
                CloseHandle(h);
                note = "shared arena section " + nm + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr->pack_hash != shared_pack_hash) {
                char b[256];
                std::snprintf(b, sizeof b, "shared arena %s was written for pack hash %016llx, expected %016llx", nm.c_str(),
                              (unsigned long long) hdr->pack_hash, (unsigned long long) shared_pack_hash);
                UnmapViewOfFile(view);
                CloseHandle(h);
                note = b;
                return nullptr;
            }
        }
        {
            std::lock_guard<std::mutex> lk(g_views_mu);
            g_views[view] = h;
        }
        mapping_base = view;
        mapping_bytes = file_bytes;
        got = PageBacking::NormalPages;
        note = "shared section " + nm + " (pack hash checked)";
        return static_cast<uint8_t*>(view) + kSharedArenaHeaderBytes;
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege.  Having it assigned to the account is not enough: the
    // PROCESS must enable it in its own token (AdjustTokenPrivileges) before VirtualAlloc, or the call fails.
    // An account without the assignment, or a failure to enable, leaves the process as it was: VirtualAlloc
    // then refuses and the 4 KB fallback below runs - that is the EXPECTED outcome on a desktop.
    {
        HANDLE tok = nullptr;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp{};
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                if (!AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() != ERROR_NOT_ALL_ASSIGNED)
                    (void) 0;   // nothing actionable: the large-page attempt below reports the outcome
            }
            CloseHandle(tok);
        }
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege; a normal account does not have it and VirtualAlloc then
    // fails with ERROR_PRIVILEGE_NOT_HELD.  That is the EXPECTED outcome on a desktop, not an error.
    SIZE_T large = GetLargePageMinimum();
    // A/B switch: STRATA_NO_LARGEPAGES=1 skips the large-page attempt, same run, same boot.
    if (large > 0 && std::getenv("STRATA_NO_LARGEPAGES") == nullptr) {
        // MEM_LARGE_PAGES requires the allocation size to be an exact multiple of the large page size -
        // anything else is ERROR_INVALID_PARAMETER (87), which reads like a privilege problem but is not.
        // Round up: the slack is under 2 MB and the tail stays unused.
        const SIZE_T lbytes = (SIZE_T) (((SIZE_T) bytes + large - 1) / large * large);
        void* p = VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) {
            got = PageBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long) large) + " B)";
            return p;
        }
        // 1450 (ERROR_NO_SYSTEM_RESOURCES) is the large-page pool saying no, 87 is a size that is not a
        // multiple of the minimum, 1314 is the privilege: without the byte count the three read as one bug.
        note = "large pages refused for " + std::to_string((unsigned long long) lbytes) + " B (GetLargePageMinimum=" +
               std::to_string((unsigned long long) large) + ", VirtualAlloc error " +
               std::to_string((unsigned long long) GetLastError()) + "); using 4 KB pages";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = PageBacking::NormalPages;
    if (p == nullptr) {   // 1455: Windows' commit limit (RAM + page file) is reached
        const DWORD e = GetLastError();
        note += "; the 4 KB allocation failed too (VirtualAlloc error " + std::to_string((unsigned long long) e) +
                (e == ERROR_COMMITMENT_LIMIT ? ": the commit limit, RAM + page file, is reached - increase the page "
                                               "file (64000 MB) or close programs" : "") + ")";
    }
    return p;
#else
    if (!shared_file.empty()) {
        if (shared_pack_hash == 0) {
            note = "shared arena requires a nonzero pack hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t) file_bytes) != 0) {
                const int e = errno;
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.pack_hash = shared_pack_hash;
            const ssize_t written = pwrite(fd, &hdr, sizeof hdr, 0);
            if (written != (ssize_t) sizeof hdr) {
                const int e = errno;
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (written < 0 ? std::string(std::strerror(e)) : std::string("short write"));
                return nullptr;
            }
        } else {
            if (st.st_size < 0 || (uint64_t) st.st_size != file_bytes) {
                close(fd);
                note = "shared arena " + shared_file + " is " +
                       std::to_string((unsigned long long) st.st_size) + " B, expected " +
                       std::to_string((unsigned long long) file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            ssize_t got_header = pread(fd, &hdr, sizeof hdr, 0);
            // the creator sizes the file first and writes the header right after: an engine that starts in that gap waits
            // for it (up to 10 s) instead of calling the file broken
            for (int w = 0; w < 100 && got_header == (ssize_t) sizeof hdr &&
                                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 &&
                                hdr.magic[0] == 0 && hdr.magic[1] == 0; ++w) {
                usleep(100 * 1000);
                got_header = pread(fd, &hdr, sizeof hdr, 0);
            }
            if (got_header == (ssize_t) sizeof hdr && std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 &&
                hdr.magic[0] == 0 && hdr.magic[1] == 0) {
                // still no header after 10 s: the engine that sized the file died before writing it.  Every engine
                // fills the arena itself, so nothing in the file is worth keeping: write the header as the creator would
                std::memset(&hdr, 0, sizeof hdr);
                std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
                hdr.version = 1;
                hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
                hdr.arena_bytes = bytes;
                hdr.pack_hash = shared_pack_hash;
                got_header = pwrite(fd, &hdr, sizeof hdr, 0);
            }
            if (got_header != (ssize_t) sizeof hdr ||
                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.pack_hash != shared_pack_hash) {
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b,
                              "shared arena %s was written for pack hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long) hdr.pack_hash,
                              (unsigned long long) shared_pack_hash);
                note = b;
                return nullptr;
            }
        }

        // tmpfs: take the pages now (ftruncate only sets the length).  With /dev/shm too small the load would die with a bus
        // error part-way; here a full tmpfs is a plain error at start, for every engine that maps the file (pages already
        // taken cost nothing), and the server then starts its replicas with private copies.  A file on a disk stays sparse.
#ifdef __linux__
        {
            struct statfs sf{};
            if (fstatfs(fd, &sf) == 0 && (unsigned long) sf.f_type == 0x01021994ul) {
                // a memfd (or any tmpfs file not capped by a mount size) would take the RAM and then the OOM killer: check the
                // free RAM and the cgroup's room for the pages that are not there yet before taking them
                struct stat st2{};
                const uint64_t have = fstat(fd, &st2) == 0 ? (uint64_t) st2.st_blocks * 512u : 0;
                const uint64_t need = file_bytes > have ? file_bytes - have : 0;
                if (need > 0) {
                    const uint64_t room = free_ram_room();
                    if (room != ~uint64_t{0} && need + (2ull << 30) > room) {
                        close(fd);
                        note = "not enough free RAM for the shared arena " + shared_file + ": it needs " +
                               std::to_string((unsigned long long) (need >> 20)) + " MiB more and " +
                               std::to_string((unsigned long long) (room >> 20)) + " MiB are free (2 GiB are kept back)";
                        return nullptr;
                    }
                }
                const int fe = posix_fallocate(fd, 0, (off_t) file_bytes);
                if (fe != 0 && fe != EOPNOTSUPP && fe != EINVAL) {
                    close(fd);
                    note = "not enough room for the shared arena " + shared_file + " (" +
                           std::to_string((unsigned long long) (file_bytes >> 20)) + " MiB): " + std::strerror(fe);
                    return nullptr;
                }
            }
        }
#endif
        void* map = mmap(nullptr, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = PageBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (pack hash checked)";
        return (uint8_t*) map + kSharedArenaHeaderBytes;
    }

    // STRATA_NO_LARGEPAGES=1 is the same-run A/B switch the Windows branch documents; honor it
    // here too, so the large-page path can be compared without changing the pool or rebooting.
    if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            got = PageBacking::LargePages;
            note = "hugetlb 2 MB pages";
            return p;
        }
        // MAP_HUGETLB is all-or-nothing: a pool smaller than the mapping fails exactly like an absent
        // one, and the old message guessed "no hugetlb pool configured" either way. Name the shortfall:
        // how many 2 MiB pages the mapping needs against what vm.nr_hugepages actually holds.
        const unsigned long long need = ((unsigned long long) bytes + (1ull << 21) - 1) / (1ull << 21);
        unsigned long long pool = 0;
        bool have_pool = false;
        if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
            have_pool = std::fscanf(f, "%llu", &pool) == 1;
            std::fclose(f);
        }
        note = "MAP_HUGETLB unavailable (needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
               (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = PageBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    // #771: with THP "always" the kernel already backs this mapping with huge pages where it can; the extra
    // MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
    // machine stretched a 25 s start to 432 s.  So it is not asked for there.  STRATA_NO_ARENA_THP=1 skips it
    // anywhere (the request alone; STRATA_NO_LARGEPAGES also skips the hugetlb try).
    std::string thp_mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) thp_mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    const bool skip_thp_request = std::getenv("STRATA_NO_ARENA_THP") != nullptr || thp_mode == "always";
    if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && skip_thp_request) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += std::getenv("STRATA_NO_ARENA_THP") != nullptr
                    ? "; transparent huge pages skipped (STRATA_NO_ARENA_THP)"
                    : "; transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && std::getenv("STRATA_ARENA_THP") == nullptr &&
               free_huge_bytes() < bytes / 2) {
        // #2126: THP "madvise" + defrag "madvise" makes every fault of a MADV_HUGEPAGE mapping reclaim and compact
        // until it finds a 2 MiB block.  When less than half of the arena could be backed by blocks that are free
        // now (a 62 GB box holding 47 GB of anonymous memory: 148K failed compactions, 100% system time, the load
        // never ended), that is minutes of kernel work per GiB for a TLB gain.  Said, and STRATA_ARENA_THP=1 asks for it anyway.
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages NOT requested: only " + std::to_string(free_huge_bytes() >> 30) +
                " GiB of free 2 MiB blocks for a " + std::to_string(bytes >> 30) +
                " GiB arena, compaction would stall the load (STRATA_ARENA_THP=1 requests them anyway) (#2126)";
        std::fprintf(stderr, "strata: expert arena: %s\n", note.c_str());   // said now: the load that follows can take minutes
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
#endif
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
#ifdef _WIN32
    (void) bytes;
    HANDLE section = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_views_mu);
        auto it = g_views.find(p);
        if (it != g_views.end()) {
            section = it->second;
            g_views.erase(it);
        }
    }
    if (section != nullptr) {          // a shared-section view: the pages stay while another engine has the section
        UnmapViewOfFile(p);
        CloseHandle(section);
        return;
    }
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }
}  // namespace

namespace {
std::atomic<uint64_t> g_idle_stack_lo{0}, g_idle_stack_hi{0};
std::vector<uint64_t> g_held;   // sorted; set once by main() before any thread that pins exists
}  // namespace

void set_idle_thread_stack(uint64_t lo, uint64_t hi) {
    g_idle_stack_lo.store(lo);
    g_idle_stack_hi.store(hi);
}

void set_held_address_space(std::vector<uint64_t> bases) {
    std::sort(bases.begin(), bases.end());
    g_held = std::move(bases);
}

bool arena_no_register() {
    const char* e = std::getenv("STRATA_ARENA_PIN_GIB");
    if (e != nullptr && *e != '\0') return std::string(e) == "none";   // an explicit setting wins over the engine's flag
    return arena_uma_no_register_flag().load(std::memory_order_relaxed);
}

namespace {
std::vector<int>& pin_order() {
    static std::vector<int> v;
    return v;
}
}  // namespace

void set_arena_pin_order(const std::vector<int>& slices) { pin_order() = slices; }

namespace {
struct SliceDevices {
    std::vector<int> dev;
    std::vector<uint64_t> cap;
    std::vector<int> stage;   ///< per slice: its stage (a new stage starts where it changes); empty: where dev changes
    std::vector<std::string> why;   ///< per device: where its cap came from (the log line)
    std::vector<char> live;         ///< per device: re-read its adapter's budget before each of its stages registers
};
SliceDevices& slice_devices() {
    static SliceDevices v;
    return v;
}
}  // namespace

void set_arena_slice_devices(const std::vector<int>& slice_dev, const std::vector<uint64_t>& per_dev_cap,
                             const std::vector<int>& slice_stage, const std::vector<std::string>& per_dev_why,
                             const std::vector<char>& per_dev_live) {
    slice_devices().dev = slice_dev;
    slice_devices().cap = per_dev_cap;
    slice_devices().why = per_dev_why;
    slice_devices().live = per_dev_live;
    slice_devices().stage = slice_stage.size() == slice_dev.size() ? slice_stage : std::vector<int>{};
}

bool arena_slice_devices_set() { return !slice_devices().dev.empty(); }

int arena_pin_per_stage_env() {
    const char* e = std::getenv("STRATA_ARENA_PIN_PER_STAGE");
    if (e == nullptr || *e == '\0') return -1;
    return e[0] == '0' ? 0 : 1;
}

int arena_pin_cap_gib() {
    const char* e = std::getenv("STRATA_ARENA_PIN_GIB");
    if (e == nullptr || *e == '\0') return -1;
    if (std::string(e) == "none") return -1;   // 153-igpu: handled by arena_no_register(); no cap otherwise
    if (std::string(e) == "auto") return -2;   // #243: the Windows shared-memory budget sets the sliced pin's cap
    if (std::string(e) == "none") return -3;   // extras/wddm-pin: nothing pinned (`0` stays the whole arena, as documented)
    const int v = std::atoi(e);
    return v < 0 ? -1 : v;
}

uint64_t shared_pin_margin() {
    // extras/wddm-pin: STRATA_PIN_MARGIN_GIB, what stays free of the shared-memory budget (default 4 GiB, #243's)
    static const uint64_t v = [] {
        const char* e = std::getenv("STRATA_PIN_MARGIN_GIB");
        return e != nullptr && *e != '\0' && std::atof(e) >= 0.0 ? (uint64_t) (std::atof(e) * 1073741824.0) : (4ull << 30);
    }();
    return v;
}

bool shared_pin_budget(int dev, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        why = "no CUDA device properties";
        return false;
    }
    return strata::platform::gpu_shared_memory_budget(p.luid, budget, usage, why);   // false off Windows
}

// #243: how much a sliced registration may pin on Windows.  Page-locked memory a GPU maps is charged to its shared
// (non-local) WDDM segment; pinned slice by slice until the driver refused one (28 GiB of a 63 GB PC), that segment was
// left full and every later cudaMalloc failed "out of memory".  The margin (4 GiB) stays free for what the engine
// allocates after the arena; without the DXGI numbers, RAM/2 - 8 GiB (the budget is about half the RAM).
// extras/wddm-pin: for a given device (its adapter's budget), and public.
bool shared_pin_limit_for(int dev, uint64_t& limit, std::string& why) {
#ifdef _WIN32
    constexpr uint64_t GiB = 1ull << 30;
    char buf[256];
    uint64_t budget = 0, usage = 0;
    std::string err;
    const uint64_t margin = shared_pin_margin();
    if (shared_pin_budget(dev, budget, usage, err)) {
        limit = budget > usage + margin ? budget - usage - margin : 0;
        std::snprintf(buf, sizeof buf, "the shared-memory budget of CUDA%d's adapter %.1f GiB - %.1f GiB in use - %.1f GiB",
                      dev, (double) budget / GiB, (double) usage / GiB, (double) margin / GiB);
        why = buf;
        return true;
    }
    const uint64_t ram = strata::platform::total_physical_memory();
    if (ram == 0) { why = err; return false; }
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "%s: RAM/2 - 8 GiB of %.1f GiB", err.c_str(), (double) ram / GiB);
    why = buf;
    return true;
#else
    (void) dev;
    limit = 0;
    why = "no shared-memory budget outside Windows";
    return false;
#endif
}

bool shared_pin_limit(uint64_t& limit, std::string& why) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); dev = 0; }
    return shared_pin_limit_for(dev, limit, why);
}

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}

#if defined(_WIN32) && defined(STRATA_USE_HIP)
// #3431.  On Windows the HIP runtime gives every cudaHostRegister'ed range a second address for the GPU (its alias), and it
// takes any host pointer inside that alias for the registered memory.  Unlike the addresses of cudaMalloc, the alias is not
// reserved in the process's own address space, so a thread stack, a heap block or a mapping can be placed there.  In about one
// engine start in ten one was (the stack of the thread that pins the arena): a cudaMemcpy from a local variable then read the
// arena, a copy back into one wrote the arena, and every long prompt was read wrong until the process ended, with no error.
// So the alias range is reserved here (address space only, nothing committed), and a range whose alias already covers memory
// in use is not kept: true when the alias is clear (or there is none), with its reservations added to `claims`.  A runtime that
// reserves the alias itself would have every range refused here, and logged.
bool claim_alias(void* host, uint64_t n, std::vector<void*>& claims, std::string& why) {
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, host, 0) != cudaSuccess || d == nullptr || d == host) {
        (void) cudaGetLastError();
        return true;
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t gran = si.dwAllocationGranularity;                 // where VirtualAlloc can start a reservation
    const uint64_t top = (uint64_t) si.lpMaximumApplicationAddress;  // nothing of the process lies above
    const uint64_t a = (uint64_t) d, end = std::min<uint64_t>(a + n, top);
    std::vector<void*> got;
    MEMORY_BASIC_INFORMATION mbi;
    char msg[256] = "";
    // the granule the alias starts in: when it is free in front of the alias too, the reservation starts at its boundary, or a
    // small allocation could still be placed there and reach into the alias
    uint64_t at = a;
    const uint64_t a0 = a & ~(gran - 1);
    if (a0 < a && a0 < end && VirtualQuery((void*) a0, &mbi, sizeof mbi) == sizeof mbi && mbi.State == MEM_FREE &&
        (uint64_t) mbi.BaseAddress + mbi.RegionSize > a)
        at = a0;
    while (at < end) {
        if (VirtualQuery((void*) at, &mbi, sizeof mbi) != sizeof mbi) {
            std::snprintf(msg, sizeof msg, "VirtualQuery failed at %#llx", (unsigned long long) at);
            break;
        }
        const uint64_t re = std::min<uint64_t>((uint64_t) mbi.BaseAddress + mbi.RegionSize, end);
        const uint64_t idle_lo = g_idle_stack_lo.load(), idle_hi = g_idle_stack_hi.load();
        const bool held = mbi.State == MEM_RESERVE && std::binary_search(g_held.begin(), g_held.end(), (uint64_t) mbi.AllocationBase);
        const bool idle = mbi.State != MEM_FREE && idle_lo < idle_hi && (uint64_t) mbi.AllocationBase == idle_lo && re <= idle_hi;
        if (held || idle) {
            // held: main()'s reservation, kept for the life of the process.  idle: the first thread's stack, which only waits
            at = (uint64_t) mbi.BaseAddress + mbi.RegionSize;
            continue;
        }
        if (mbi.State != MEM_FREE) {
            ULONG_PTR lo = 0, hi = 0;
            GetCurrentThreadStackLimits(&lo, &hi);
            const bool own_stack = at < (uint64_t) hi && (uint64_t) lo < re;
            const char* kind = own_stack ? "this thread's stack" : mbi.Type == MEM_IMAGE ? "a module" :
                               mbi.Type == MEM_MAPPED ? "a mapped view" : "private memory";
            std::snprintf(msg, sizeof msg, "the GPU alias [%#llx, +%llu MiB) of the range at %p covers memory in use at %#llx (%s, from %p)",
                          (unsigned long long) a, (unsigned long long) (n >> 20), host, (unsigned long long) at, kind,
                          mbi.AllocationBase);
            break;
        }
        // a free piece that starts inside a granule shares it with an allocation: nothing new can start there
        const uint64_t b = (at + gran - 1) & ~(gran - 1);
        if (b < re) {
            void* r = VirtualAlloc((void*) b, (SIZE_T) (re - b), MEM_RESERVE, PAGE_NOACCESS);
            if (r == nullptr) {
                std::snprintf(msg, sizeof msg, "could not reserve the GPU alias [%#llx, +%llu MiB) of the range at %p (error %lu)",
                              (unsigned long long) a, (unsigned long long) (n >> 20), host, GetLastError());
                break;
            }
            got.push_back(r);
        }
        at = (uint64_t) mbi.BaseAddress + mbi.RegionSize;
    }
    if (msg[0] != '\0') {
        for (void* g : got) VirtualFree(g, 0, MEM_RELEASE);
        why = msg;
        return false;
    }
    claims.insert(claims.end(), got.begin(), got.end());
    return true;
}
#endif

// after the ranges are unregistered
void release_alias_claims(std::vector<void*>& claims) {
#ifdef _WIN32
    for (void* c : claims) VirtualFree(c, 0, MEM_RELEASE);
#endif
    claims.clear();
}

// pin_register for the arena's ranges.  Windows HIP: a range whose GPU alias covers memory in use is unregistered again and
// reported as refused, so the caller's "the driver refused it" path applies to it (#3431).
cudaError_t register_range(PinnedArena& arena, void* p, uint64_t n) {
    const cudaError_t e = pin_register(p, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped);
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    std::string why;
    if (e == cudaSuccess && !claim_alias(p, n, arena.alias_claims, why)) {
        (void) pin_unregister(p);
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata pin: %s; that range stays unregistered (#3431)\n", why.c_str());
        arena.note = "left unregistered: " + why + " (#3431); " + arena.note;
        return cudaErrorMemoryAllocation;
    }
#else
    (void) arena;
#endif
    return e;
}
}  // namespace

namespace {
// The working-set lock of every run of unregistered slices and of the bytes past the last bound (a scattered pin).
void lock_unregistered(PinnedArena& a, const std::vector<uint64_t>& bounds, uint64_t bytes) {
    const char* env = std::getenv("STRATA_ARENA_LOCK");
    if (a.backing == PageBacking::LargePages) {
        a.note = "large pages are resident without a lock; " + a.note;
        return;
    }
    if (env != nullptr && std::string(env) == "0") return;
    std::string lnote;
    const size_t n_sl = bounds.size() - 1;
    for (size_t i = 0; i < n_sl;) {
        if (a.slice_pinned[i]) { ++i; continue; }
        size_t j = i;
        while (j < n_sl && !a.slice_pinned[j]) ++j;   // [i, j): a run of unregistered slices
        const uint64_t off = bounds[i], len = bounds[j] - bounds[i];
        const strata::platform::LockResult lr = strata::platform::lock_resident((uint8_t*) a.base + off, len);
        if (lr.locked_bytes) a.locked_ranges.emplace_back(off, lr.locked_bytes);
        a.locked_bytes += lr.locked_bytes;
        lnote = lr.note;
        i = j;
    }
    const uint64_t last = bounds.back();
    if (last < bytes) {
        const strata::platform::LockResult lr = strata::platform::lock_resident((uint8_t*) a.base + last, bytes - last);
        if (lr.locked_bytes) a.locked_ranges.emplace_back(last, lr.locked_bytes);
        a.locked_bytes += lr.locked_bytes;
    }
    if (!lnote.empty()) a.note = lnote + "; " + a.note;
}

// extras/wddm-pin: each device registers the slices it reads, in its own context, without cudaHostRegisterPortable:
// a portable registration is mapped into EVERY context, and under WDDM every adapter then charges the whole pinned
// range to its shared (non-local) segment - the 5080 + 3090 rig's refused allocations (#253).  Stage by stage in layer
// order, each from its first slice up, until its room (the GPU's cap, or its adapter's budget read just before) or the
// driver's first refusal (which ends every later stage on that GPU too).
void register_per_stage(PinnedArena& a, const std::vector<uint64_t>& bounds, uint64_t bytes) {
    constexpr double GiB = 1073741824.0;
    const SliceDevices& sd = slice_devices();
    const size_t n = bounds.size() - 1;
    a.per_stage = true;
    a.scattered = true;
    a.slice_bytes = 1;
    a.slice_owner = sd.dev;
    a.slice_pinned.assign(n, 0);
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) { (void) cudaGetLastError(); prev = 0; }
    // the stages: runs of consecutive slices with one owner, in layer order; a device's cap and a refusal by its
    // driver hold for every stage on it (one adapter: --split-device 0 puts both stages on CUDA0)
    std::vector<uint64_t> dev_got;
    std::vector<std::string> dev_stop;
    const auto t0 = std::chrono::steady_clock::now();
    int stage = 0;
    for (size_t s0 = 0; s0 < n;) {
        const int d = sd.dev[s0];
        size_t s1 = s0;
        while (s1 < n && sd.dev[s1] == d && (sd.stage.empty() || sd.stage[s1] == sd.stage[s0])) ++s1;   // one stage
        stage = sd.stage.empty() ? stage + 1 : sd.stage[s0] + 1;
        if (d < 0) { s0 = s1; continue; }
        if ((size_t) d >= dev_got.size()) { dev_got.resize((size_t) d + 1, 0); dev_stop.resize((size_t) d + 1); }
        // what this stage may pin: the GPU's cap less what its earlier stages pinned; with a live budget the room its
        // adapter has NOW (budget - usage - margin, read just before this stage): whatever the earlier registrations,
        // on this adapter or - if a driver charges a non-portable one to every adapter - on the others, it shows here
        uint64_t room = UINT64_MAX;
        std::string why_now;
        const uint64_t cap = (size_t) d < sd.cap.size() ? sd.cap[(size_t) d] : 0;
        if ((size_t) d < sd.live.size() && sd.live[(size_t) d]) {
            uint64_t lim = 0;
            if (shared_pin_limit_for(d, lim, why_now)) room = lim;
            else if (cap > 0) room = cap > dev_got[(size_t) d] ? cap - dev_got[(size_t) d] : 0;
        } else if (cap > 0) {
            room = cap > dev_got[(size_t) d] ? cap - dev_got[(size_t) d] : 0;
        }
        uint64_t want = 0, got = 0;
        int n_ok = 0;
        std::string& stop = dev_stop[(size_t) d];
        const auto td = std::chrono::steady_clock::now();
        if (stop.empty() && cudaSetDevice(d) != cudaSuccess)
            stop = std::string("cudaSetDevice failed: ") + cudaGetErrorString(cudaGetLastError());
        for (size_t i = s0; i < s1; ++i) {
            const uint64_t off = bounds[i], len = bounds[i + 1] - bounds[i];
            want += len;
            if (!stop.empty()) continue;
            if (room != UINT64_MAX && got + len > room) {
                char b[128];
                std::snprintf(b, sizeof b, "the cap: %.1f GiB left for this stage on CUDA%d", (double) room / GiB, d);
                stop = b;
                continue;
            }
            const cudaError_t e = pin_register((uint8_t*) a.base + off, (size_t) len, cudaHostRegisterMapped);
            if (e != cudaSuccess) {
                (void) cudaGetLastError();
                stop = std::string("the driver refused a slice (") + cudaGetErrorString(e) + ")";
                continue;
            }
            void* dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, (uint8_t*) a.base + off, 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dp = nullptr;
            }
            a.slice_pinned[i] = 1;
            a.slice_starts.push_back(off);
            a.slice_index.push_back((int) i);
            a.reg_dev.push_back(d);
            a.reg_alias.push_back(dp);
            got += len;
            dev_got[(size_t) d] += len;
            ++n_ok;
            ++a.registered_slices;
        }
        char b[384];
        std::snprintf(b, sizeof b, "%sstage %d layers %lld-%lld: pinned %.2f of %.2f GiB (%d of %d slices) in the context of "
                                   "cuda%d in %.1f s%s%s", a.stage_note.empty() ? "" : "; ", stage, (long long) s0,
                      (long long) s1 - 1, (double) got / GiB, (double) want / GiB, n_ok, (int) (s1 - s0), d,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count(),
                      got < want ? ", the rest unpinned: " : "", got < want ? stop.c_str() : "");
        a.stage_note += b;
        if (!why_now.empty()) a.stage_note += " (cap read now: " + why_now + ")";
        else if ((size_t) d < sd.why.size() && !sd.why[(size_t) d].empty()) a.stage_note += " (cap: " + sd.why[(size_t) d] + ")";
        if (stop.rfind("the cap: ", 0) == 0) stop.clear();   // a cap ends this stage only: the next stage reads its own
        s0 = s1;
    }
    cudaSetDevice(prev);
    for (size_t i = 0; i < n && a.slice_pinned[i]; ++i) a.registered_bytes = bounds[i + 1];   // the pinned PREFIX
    char t[64];
    std::snprintf(t, sizeof t, " (%.1f s)", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    a.note = "cudaHostRegister per stage, each stage's layers in its own CUDA context, not portable "
             "(STRATA_ARENA_PIN_PER_STAGE)" + std::string(t) + ": " + a.stage_note + "; " + a.note;
    lock_unregistered(a, bounds, bytes);
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         uint64_t max_pinned_bytes, const std::string& shared_file,
                         uint64_t shared_pack_hash) : capacity(bytes) {
    if (bytes == 0) return;
    base = reserve(bytes, backing, note, shared_file, shared_pack_hash, mapping_base, mapping_bytes);
    if (!shared_file.empty() && base != nullptr && mapping_base != nullptr)
        shared_state = reinterpret_cast<std::atomic<uint64_t>*>(static_cast<char*>(mapping_base) +
                                                                offsetof(SharedArenaHeader, reserved));
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base && arena_no_register()) {
        // 153-igpu (UMA mode): nothing is registered; the whole arena stays resident through the working-set lock instead
        slice_bytes = 1;   // "sliced" with no slice registered: the destructor unlocks from offset registered_bytes (0)
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (backing == PageBacking::LargePages) {
            note = "large pages are resident without a lock; " + note;
        } else if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr = strata::platform::lock_resident((uint8_t*) base, bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        } else {
            note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
        }
        note = "NOT registered with the GPU driver (integrated GPU, UMA mode; STRATA_ARENA_PIN_GIB=0 registers it); " + note;
        return;
    }
    // extras/wddm-pin: a layer split's stages each register their own layers in their own context (not portable)
    const bool by_stage = base != nullptr && bounds.size() >= 2 && slice_devices().dev.size() == bounds.size() - 1;
    if (by_stage) register_per_stage(*this, bounds, bytes);
    if (base && !by_stage) {
        // #243: STRATA_ARENA_PIN_GIB=N caps the registration from the start where the caller set no cap
        const int env_gib = arena_pin_cap_gib();
        uint64_t cap = max_pinned_bytes;
        std::string cap_why = "by the engine (multi-GPU under WDDM, or remote experts)";
        if (cap == 0 && env_gib > 0) {
            cap = (uint64_t) env_gib << 30;
            cap_why = "by STRATA_ARENA_PIN_GIB";
        }
        if (env_gib == -3 && cap <= 1) {   // extras/wddm-pin: STRATA_ARENA_PIN_GIB=none - no slice fits one byte
            cap = 1;
            cap_why = "by STRATA_ARENA_PIN_GIB=none (nothing pinned)";
        }
        const bool capped = cap > 0 && cap < bytes && bounds.size() >= 2;
        const cudaError_t e = capped ? cudaSuccess :
            register_range(*this, base, bytes);
        if (!capped && e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // Plan v0.3 P5: the whole range is refused, so pin it slice by slice from the start.  The rest stays
            // resident through the working-set lock below.  (P6: slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            bool limited = capped;
            uint64_t limit = cap;
            std::string limit_why;
#ifdef _WIN32
            // #243 (opt-in, STRATA_ARENA_PIN_GIB=auto): not up to the driver's refusal but below the shared-memory
            // budget, for a PC where the full sliced pin leaves WDDM refusing later allocations.  Not the default: a
            // 64 GB PC pins 30 GiB past that budget without trouble, and capping it at 26 cost ~20% prompt speed.
            if (!capped && env_gib == -2) limited = shared_pin_limit(limit, limit_why);
#endif
            const std::vector<int>& order = pin_order();
            if (capped && !order.empty()) {
                // extras/mgpu-fixes (D7): the cap's slices in the caller's order (a layer split's stages in turn)
                scattered = true;
                slice_pinned.assign(bounds.size() - 1, 0);
                uint64_t total = 0;
                for (const int i : order) {
                    if (i < 0 || (size_t) i + 1 >= bounds.size() || slice_pinned[(size_t) i]) continue;
                    const uint64_t off = bounds[(size_t) i], n = bounds[(size_t) i + 1] - bounds[(size_t) i];
                    if (total + n > limit) continue;
                    if (pin_register((uint8_t*) base + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                        (void) cudaGetLastError();
                        break;
                    }
                    slice_pinned[(size_t) i] = 1;
                    slice_starts.push_back(off);
                    slice_index.push_back(i);
                    total += n;
                    ++registered_slices;
                }
                // `registered_bytes` stays the pinned PREFIX (its readers' meaning); the slices are in slice_pinned
                for (size_t i = 0; i + 1 < bounds.size() && slice_pinned[i]; ++i) registered_bytes = bounds[i + 1];
            } else
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (limited && (off > limit || n > limit - off)) break;
                if (register_range(*this, (uint8_t*) base + off, n) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) limit / (double) (1ull << 30));
            note = (capped ? "cudaHostRegister limited to " + std::to_string(cap >> 30) + " GiB " + cap_why + "; " :
                             "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); " +
                             (limited ? "slices capped at " + std::string(gib) + " GiB (" + limit_why +
                                        "; STRATA_ARENA_PIN_GIB=auto; N sets a cap; #243); " : std::string())) +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB)" + (scattered ? std::string(", shared between the layer split's stages") : std::string()) +
                   "; " + note;
            if (scattered) {
                // every unregistered run of slices resident through the working-set lock, as the tail below
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                if (backing == PageBacking::LargePages) {
                    note = "large pages are resident without a lock; " + note;
                } else if (env == nullptr || std::string(env) != "0") {
                    std::string lnote;
                    const size_t n_sl = bounds.size() - 1;
                    for (size_t i = 0; i < n_sl;) {
                        if (slice_pinned[i]) { ++i; continue; }
                        size_t j = i;
                        while (j < n_sl && !slice_pinned[j]) ++j;   // [i, j): a run of unregistered slices
                        const uint64_t off = bounds[i], len = bounds[j] - bounds[i];
                        const strata::platform::LockResult lr = strata::platform::lock_resident((uint8_t*) base + off, len);
                        if (lr.locked_bytes) locked_ranges.emplace_back(off, lr.locked_bytes);
                        locked_bytes += lr.locked_bytes;
                        lnote = lr.note;
                        i = j;
                    }
                    // the bytes after the last slice bound (the arena's slack past the last layer)
                    const uint64_t last = bounds.back();
                    if (last < bytes) {
                        const strata::platform::LockResult lr = strata::platform::lock_resident((uint8_t*) base + last, bytes - last);
                        if (lr.locked_bytes) locked_ranges.emplace_back(last, lr.locked_bytes);
                        locked_bytes += lr.locked_bytes;
                    }
                    if (!lnote.empty()) note = lnote + "; " + note;
                }
            } else if (registered_bytes < bytes) {
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                if (backing == PageBacking::LargePages) {   // #779: large pages cannot be paged out: nothing to lock
                    note = "large pages are resident without a lock; " + note;
                } else if (env == nullptr || std::string(env) != "0") {
                    const strata::platform::LockResult lr =
                        strata::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                    locked_bytes = lr.locked_bytes;
                    note = lr.note + "; " + note;
                }
            }
        } else {
            note = std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Plan v0.3 P0.1: keep it RESIDENT instead. Unpinned, Windows trims the arena under memory pressure
            // and the CPU pool's rate then depends on the OS; locking it through the working set needs no
            // special privilege. STRATA_ARENA_LOCK=0 is the A/B arm.
            const char* env = std::getenv("STRATA_ARENA_LOCK");
            if (backing == PageBacking::LargePages) {       // #779: large pages cannot be paged out: nothing to lock
                note = "large pages are resident without a lock; " + note;
            } else if (env == nullptr || std::string(env) != "0") {
                const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            } else {
                note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
            }
        }
    }
}

PinnedArena::PinnedArena(uint64_t bytes, ByNeed, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                         uint64_t max_pinned_bytes, const std::string& shared_file,
                         uint64_t shared_pack_hash) : capacity(bytes) {
    by_need = true;
    if (bytes == 0) return;
    base = reserve(bytes, backing, note, shared_file, shared_pack_hash, mapping_base, mapping_bytes);
    if (!shared_file.empty() && base != nullptr && mapping_base != nullptr)
        shared_state = reinterpret_cast<std::atomic<uint64_t>*>(static_cast<char*>(mapping_base) +
                                                                offsetof(SharedArenaHeader, reserved));
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    if (base == nullptr) return;
    // Register BEFORE any page is touched (see the first constructor).  Same caps as there: the engine's
    // `max_pinned_bytes`, else STRATA_ARENA_PIN_GIB=N (> 0); "auto" (-2) is the sliced pin's and does not apply.
    const int env_gib = arena_pin_cap_gib();
    uint64_t cap = max_pinned_bytes;
    if (cap == 0 && env_gib > 0) cap = (uint64_t) env_gib << 30;
#ifdef _WIN32
    // WDDM: the pin must leave margin (the budget is ~RAM/2 and later cudaMallocs need room).  Unset: 29 GiB;
    // "auto" (-2): the sliced pin's shared-memory-budget limit (29 GiB when it cannot be worked out);
    // STRATA_ARENA_PIN_GIB=0 ("the whole arena"): no cap, up to the driver's refusal.
    if (cap == 0 && env_gib != 0) {
        uint64_t lim = 0;
        std::string why;
        cap = env_gib == -2 && shared_pin_limit(lim, why) && lim > 0 ? lim : (29ull << 30);   // (#243 sliced_pin_limit, renamed by extras/wddm-pin)
    }
#endif
    constexpr uint64_t kPage = 4096;
    std::map<uint64_t, uint64_t> done;   // registered [start, end), by start: ranges never overlap
    uint64_t total = 0;
    int refused = 0;
    bool capped = false;
    const bool none = arena_no_register();   // 153-igpu UMA mode: register no range at all; every byte is a locked "hole" below
    for (const auto& rg : ranges) {
        if (none) break;
        if (rg.second == 0 || rg.first >= bytes) continue;
        // INWARD: a registered page holds no byte of an unpinned neighbour blob (a copy that STARTS in a registered page
        // and runs past the registration is rejected by CUDA); the boundary pages stay unregistered
        uint64_t a = (rg.first + kPage - 1) & ~(kPage - 1);
        uint64_t b = std::min<uint64_t>(rg.first + rg.second, bytes) & ~(kPage - 1);
        // a page shared with a range registered before (blobs smaller than a page, or odd offsets) is already
        // registered: registering it twice is an error, so trim to what is left; a range inside one is skipped
        auto it = done.upper_bound(a);
        if (it != done.begin()) {
            auto p = std::prev(it);
            if (p->second > a) a = p->second;
        }
        if (it != done.end() && it->first < b) b = it->first;
        if (a >= b) continue;
        if (cap > 0 && total + (b - a) > cap) { capped = true; break; }
        if (register_range(*this, (uint8_t*) base + a, b - a) != cudaSuccess) {
            (void) cudaGetLastError();
            refused = 1;
            break;
        }
        done[a] = b;
        total += b - a;
    }
    for (const auto& d : done) pin_ranges.emplace_back(d.first, d.second - d.first);
    registered_bytes = total;
    registered_slices = (int) pin_ranges.size();
    // the holes stay resident through the working-set lock, as the sliced fallback does for its tail
    std::vector<std::pair<void*, uint64_t>> holes;
    uint64_t cur = 0, hole_bytes = 0;
    for (const auto& r : pin_ranges) {
        if (r.first > cur) { locked_ranges.emplace_back(cur, r.first - cur); hole_bytes += r.first - cur; }
        cur = r.first + r.second;
    }
    if (cur < bytes) { locked_ranges.emplace_back(cur, bytes - cur); hole_bytes += bytes - cur; }
    note = "pin by need: " + std::to_string(pin_ranges.size()) + " ranges registered (" + std::to_string(total >> 20) +
           " MiB)" + (refused ? ", the driver refused the next one" : capped ? ", the cap was reached" : "") +
           "; " + note;
    const char* env = std::getenv("STRATA_ARENA_LOCK");
    if (backing == PageBacking::LargePages) {   // #779: large pages cannot be paged out: nothing to lock
        note = "large pages are resident without a lock; " + note;
        locked_ranges.clear();
    } else if (env != nullptr && std::string(env) == "0") {
        note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
        locked_ranges.clear();
    } else {
        for (const auto& r : locked_ranges) holes.emplace_back((uint8_t*) base + r.first, r.second);
        const strata::platform::LockResult lr = strata::platform::lock_resident_ranges(holes);
        locked_bytes = lr.locked_bytes;
        note = lr.note + " (" + std::to_string(hole_bytes >> 20) + " MiB of holes); " + note;
        // a partial lock is unlocked region by region in the destructor; what it did not reach was never locked
        // (VirtualUnlock/munlock of an unlocked page is harmless), so the list is kept whole
    }
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred)
    : capacity(bytes), bounds_(bounds) {
    if (bytes == 0 || bounds.size() < 2) return;
    base = reserve(bytes, backing, note, std::string{}, 0, mapping_base, mapping_bytes);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    slice_bytes = 1;   // sliced: one registration per layer
}

void PinnedArena::register_slices(std::atomic<int>& ready) {
    const int n = (int) bounds_.size() - 1;
    int i = 0;
    for (; base != nullptr && i < n; ++i) {
        const uint64_t off = bounds_[(size_t) i], len = bounds_[(size_t) i + 1] - off;
        if (register_range(*this, (uint8_t*) base + off, len) != cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        slice_starts.push_back(off);
        registered_bytes = off + len;
        ++registered_slices;
        ready.store(i + 1, std::memory_order_release);
    }
    note = "cudaHostRegister per layer, pipelined with the load: " + std::to_string(registered_slices) + " of " +
           std::to_string(n) + " slices pinned; " + note;
    if (i < n && base != nullptr) {   // the rest stays resident through the working-set lock, as the sliced fallback does
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (backing == PageBacking::LargePages) {           // #779: large pages cannot be paged out: nothing to lock
            note = "large pages are resident without a lock; " + note;
        } else if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr =
                strata::platform::lock_resident((uint8_t*) base + registered_bytes, capacity - registered_bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
    ready.store(INT_MAX, std::memory_order_release);   // every slice done (or given up on)
}

PinnedArena::~PinnedArena() {
    if (base && by_need) {
        for (const auto& r : locked_ranges) strata::platform::unlock_resident((uint8_t*) base + r.first, r.second);
        for (const auto& r : pin_ranges) pin_unregister((uint8_t*) base + r.first);
        release_alias_claims(alias_claims);
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
        return;
    }
    if (base) {
        if (scattered) {
            for (const auto& r : locked_ranges) strata::platform::unlock_resident((uint8_t*) base + r.first, r.second);
        } else if (locked_bytes) {
            strata::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        }
        if (per_stage) {   // a non-portable registration is undone in the context that made it
            int prev = 0;
            if (cudaGetDevice(&prev) != cudaSuccess) { (void) cudaGetLastError(); prev = 0; }
            for (size_t k = 0; k < slice_starts.size(); ++k) {
                if (k < reg_dev.size()) cudaSetDevice(reg_dev[k]);
                pin_unregister((uint8_t*) base + slice_starts[k]);
            }
            (void) cudaGetLastError();
            cudaSetDevice(prev);
        } else if (slice_bytes) {
            for (uint64_t off : slice_starts) pin_unregister((uint8_t*) base + off);
        } else {
            pin_unregister(base);
        }
        release_alias_claims(alias_claims);
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_direct(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    LoadStats st;
    st.ok = false;
#ifdef _WIN32
    constexpr uint64_t kAlign = 4096;
    if (((uintptr_t) dst % kAlign) != 0 || chunk == 0 || chunk % kAlign != 0) return st;
    struct Piece { uint64_t off, n; int layer; };
    std::vector<Piece> pieces;
    uint64_t bytes = 0;
    for (size_t L = 0; L < layer_off.size(); ++L) {
        if (layer_off[L] % kAlign != 0 || layer_bytes[L] % kAlign != 0) return st;
        for (uint64_t p = 0; p < layer_bytes[L]; p += chunk)
            pieces.push_back({layer_off[L] + p, std::min<uint64_t>(chunk, layer_bytes[L] - p), (int) L});
        bytes += layer_bytes[L];
    }
    if (threads < 1) threads = 1;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::mutex err_mu;
    std::string err;
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (std::max)(wide, 1), L'\0');
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    auto worker = [&]() {
        HANDLE h = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            std::lock_guard<std::mutex> g(err_mu);
            if (err.empty())
                err = "cannot open " + path + " unbuffered (error " + std::to_string((unsigned long long) GetLastError()) + ")";
            next = pieces.size();
            return;
        }
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= pieces[i].layer + 1) std::this_thread::yield();
            OVERLAPPED ov{};
            ov.Offset = (DWORD) pieces[i].off;
            ov.OffsetHigh = (DWORD) (pieces[i].off >> 32);
            DWORD got = 0;
            if (!ReadFile(h, dst + pieces[i].off, (DWORD) pieces[i].n, &got, &ov) || got != pieces[i].n) {
                std::lock_guard<std::mutex> g(err_mu);
                if (err.empty())
                    err = "short unbuffered read at offset " + std::to_string(pieces[i].off) + ": got " + std::to_string(got) +
                          " of " + std::to_string(pieces[i].n) + " B (error " +
                          std::to_string((unsigned long long) GetLastError()) + ")";
                next = pieces.size();
                break;
            }
        }
        CloseHandle(h);
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    st.layers = layer_off.size();
    if (!err.empty()) {
        st.seconds = -1.0;
        st.error = err;
        return st;
    }
    st.ok = true;
    st.bytes = bytes;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#else
    (void) path; (void) dst; (void) layer_off; (void) layer_bytes; (void) threads; (void) chunk; (void) ready;
#endif
    return st;
}

bool experts_unbuffered(const std::vector<std::string>& files, uint64_t arena_bytes, std::string& why,
                        bool cache_counts, uint64_t read_bytes) {
    const char* env = std::getenv("STRATA_UNBUFFERED_LOAD");
    if (env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return env[0] != '0';
    }
#ifdef _WIN32
    LARGE_INTEGER freq{}, a{}, b{};
    QueryPerformanceFrequency(&freq);
    constexpr DWORD kRead = 64 << 10;
    constexpr int kSamples = 16;
    std::vector<uint8_t> buf(kRead);
    uint64_t total_bytes = 0;
    int fast = 0, n = 0;
    uint64_t seed = (uint64_t) GetTickCount64() * 6364136223846793005ull + 1442695040888963407ull;
    for (const std::string& f : files) {
        const int wide = MultiByteToWideChar(CP_UTF8, 0, f.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> w((size_t) (std::max)(wide, 1), L'\0');
        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, f.c_str(), -1, w.data(), wide);
        HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS,
                               nullptr);
        if (h == INVALID_HANDLE_VALUE) continue;
        LARGE_INTEGER size{};
        GetFileSizeEx(h, &size);
        total_bytes += (uint64_t) size.QuadPart;
        // random offsets: a probe must not find the blocks an earlier probe put into the cache
        for (int i = 0; i < kSamples && (uint64_t) size.QuadPart > 2ull * kRead; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const uint64_t off = ((seed >> 17) % ((uint64_t) size.QuadPart - kRead)) / kRead * kRead;
            OVERLAPPED ov{};
            ov.Offset = (DWORD) off;
            ov.OffsetHigh = (DWORD) (off >> 32);
            DWORD got = 0;
            QueryPerformanceCounter(&a);
            const BOOL ok = ReadFile(h, buf.data(), kRead, &got, &ov);
            QueryPerformanceCounter(&b);
            if (!ok) continue;
            ++n;
            fast += (double) (b.QuadPart - a.QuadPart) * 1e6 / (double) freq.QuadPart < 30.0;
        }
        CloseHandle(h);
    }
    const bool cached = n > 0 && fast * 4 >= n * 3;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    const uint64_t avail = ms.ullAvailPhys;
    // what the cache could keep beside the arena (~4 GiB for everything else)
    const uint64_t need = read_bytes == kAllFileBytes ? total_bytes : read_bytes;
    const bool keepable = strata::platform::file_cache_keeps(avail, arena_bytes, need);
    char msg[256];
    if (read_bytes == kAllFileBytes)
        std::snprintf(msg, sizeof msg, "%d of %d probe reads from the file cache; %.1f GiB available, %.1f GiB of files",
                      fast, n, (double) avail / (1ull << 30), (double) total_bytes / (1ull << 30));
    else
        std::snprintf(msg, sizeof msg, "%.1f GiB available, %.1f GiB of it still to be taken by the RAM copy, %.1f GiB "
                      "of experts read from the files: the file cache %s keep them",
                      (double) avail / (1ull << 30), (double) arena_bytes / (1ull << 30), (double) need / (1ull << 30),
                      keepable ? "can" : "cannot");
    why = msg;
    return (!cached || !cache_counts) && !keepable;
#else
    (void) files; (void) arena_bytes; (void) cache_counts; (void) read_bytes;
    why = "buffered (not Windows)";
    return false;
#endif
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    return load_experts_ranges(path, dst, layer_off, layer_bytes, threads, chunk, ready, nullptr);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready, const PadMap* pad) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: see the header - MSVC's `basic_filebuf::xsgetn` splits any request larger than
        // `_INTERNAL_BUFSIZ - 1` into 4095-byte freads, which turned one 8 MiB chunk into ~2048 4 KiB reads.
        // `fread` sees a request bigger than the stream buffer and passes it to `_read()`/
        // `ReadFile()` unchanged, so the chunk size reaches the disk.  Buffered, not `FILE_FLAG_NO_BUFFERING`:
        // the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= (int) L + 1) std::this_thread::yield();
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                if (pad == nullptr) {
                    std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                } else {   // the padded layout: a chunk may span blobs, each lands at its own page-aligned start
                    const uint64_t B = pad->blob[(size_t) L], S = pad->stride[(size_t) L];
                    for (uint64_t q = 0; q < n;) {
                        const uint64_t p = pos + q, bi = p / B, in = p % B;
                        const uint64_t m = std::min<uint64_t>(n - q, B - in);
                        std::memcpy(dst + pad->dst_off[(size_t) L] + bi * S + in, buf.data() + q, (size_t) m);
                        q += m;
                    }
                }
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace strata::core
