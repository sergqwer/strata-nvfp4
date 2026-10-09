// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <vector>
#pragma comment(lib, "advapi32.lib")
#else
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    // diag (ram88): STRATA_EMULATE_RAM_GIB=N reports N GiB, for testing the RAM rules beside a RAM ballast
    if (const char* e = std::getenv("STRATA_EMULATE_RAM_GIB"); e != nullptr && std::atof(e) > 0.0)
        return (uint64_t) (std::atof(e) * 1073741824.0);
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

namespace {
// a REG_MULTI_SZ value of Memory Management as narrow strings (drive letters and digits only matter here)
bool mm_multi_sz(const wchar_t* name, std::vector<std::string>& out) {
    const wchar_t* key = L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management";
    DWORD bytes = 0;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, name, RRF_RT_REG_MULTI_SZ, nullptr, nullptr, &bytes) != ERROR_SUCCESS)
        return false;
    std::vector<wchar_t> buf(bytes / sizeof(wchar_t) + 2, 0);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, name, RRF_RT_REG_MULTI_SZ, nullptr, buf.data(), &bytes) != ERROR_SUCCESS)
        return false;
    for (const wchar_t* p = buf.data(); *p != 0; p += std::wcslen(p) + 1) {
        std::string s;
        for (const wchar_t* c = p; *c != 0; ++c) s += *c < 128 ? (char) *c : '_';
        out.push_back(s);
    }
    return true;
}
}  // namespace

PageFiles page_files() {
    PageFiles r;
    std::vector<std::string> set;
    if (!mm_multi_sz(L"PagingFiles", set)) return r;   // no value at all: Windows has no page file setting to read
    r.known = true;
    // "?:\pagefile.sys": automatic on every drive - Windows keeps it where ExistingPageFiles says, else the system drive
    std::vector<char> auto_drives;
    {
        std::vector<std::string> now;
        mm_multi_sz(L"ExistingPageFiles", now);
        for (const std::string& s : now) {
            const size_t c = s.find(':');
            if (c != std::string::npos && c > 0) auto_drives.push_back((char) std::toupper((unsigned char) s[c - 1]));
        }
        if (auto_drives.empty()) {
            wchar_t win[MAX_PATH] = {};
            auto_drives.push_back(GetSystemWindowsDirectoryW(win, MAX_PATH) > 0 ? (char) std::toupper((int) win[0]) : 'C');
        }
    }
    for (const std::string& line : set) {
        char drive = 0;
        int64_t initial = -1, max = -1;
        if (!parse_paging_file(line, drive, initial, max)) continue;
        std::vector<char> drives = drive == '?' ? auto_drives : std::vector<char>{drive};
        for (const char d : drives) {
            const wchar_t root[] = {(wchar_t) d, L':', L'\\', 0};
            ULARGE_INTEGER free_b{}, total_b{};
            const bool vol = GetDiskFreeSpaceExW(root, &free_b, &total_b, nullptr) != 0;
            uint64_t current_mb = 0;
            const wchar_t file[] = {(wchar_t) d, L':', L'\\', L'p', L'a', L'g', L'e', L'f', L'i', L'l', L'e', L'.',
                                    L's', L'y', L's', 0};
            WIN32_FIND_DATAW fd{};
            if (HANDLE h = FindFirstFileW(file, &fd); h != INVALID_HANDLE_VALUE) {   // the directory entry: the
                current_mb = ((((uint64_t) fd.nFileSizeHigh) << 32) | fd.nFileSizeLow) >> 20;  // file itself is locked
                FindClose(h);
            }
            // what is there for sure: a file Windows grows on demand may not grow in time for WDDM's charge (#60)
            const uint64_t mb = vol ? page_file_guaranteed_mb(initial, free_b.QuadPart >> 20, current_mb) : current_mb;
            r.total_mb += mb;
            r.grows = r.grows || page_file_grows(initial, max);
            char b[160];
            if (max > 0 && initial == max)
                std::snprintf(b, sizeof b, "%c: %lld MB", d, (long long) max);
            else if (max > 0)
                std::snprintf(b, sizeof b, "%c: %lld-%lld MB, grows on demand, now %llu MB", d, (long long) initial,
                              (long long) max, (unsigned long long) current_mb);
            else
                std::snprintf(b, sizeof b, "%c: system-managed, grows on demand, now %llu MB", d,
                              (unsigned long long) current_mb);
            std::string part = b;
            if (initial > 0 && mb < (uint64_t) initial)
                part += " (only " + std::to_string((unsigned long long) mb) + " fit the free disk)";
            r.detail += (r.detail.empty() ? "" : ", ") + part;
        }
    }
    if (r.detail.empty()) r.detail = "no page file";
    return r;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

PageFiles page_files() { return {}; }   // Windows' commit limit only

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}
#endif

ProcIo proc_io_sample() {
    ProcIo r;
#if defined(__linux__)
    if (std::FILE* f = std::fopen("/proc/self/io", "r")) {
        char key[64];
        unsigned long long v;
        while (std::fscanf(f, "%63[^:]: %llu\n", key, &v) == 2)
            if (std::strcmp(key, "read_bytes") == 0) { r.read_bytes = v; r.valid = true; }
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        if (const char* p = std::strrchr(buf, ')')) {   // fields after the command: state ppid ... minflt cminflt majflt
            unsigned long long minflt = 0, cminflt = 0, majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %llu %llu %llu", &minflt, &cminflt, &majflt) == 3)
                r.major_faults = majflt;
        }
    }
#endif
    return r;
}

}  // namespace strata::platform
