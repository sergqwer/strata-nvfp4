// The disk tier of --serve's parked conversations (--conversation-disk-dir): the RAM cache's evictions
// (ConversationCache::set_on_evict) are written to session files (conversation_file.hpp, one per layer-split stage)
// by a background thread and read back when a request continues them, within a time limit and a byte budget.
// By default only this process reads its files back, and the folder's files of this tier are removed at start and at
// the end.  `persist`: the files are bound to the session files' identity (model and settings) instead, a small index
// file beside each conversation lists its tokens, and the next start picks them up (no time limit, the budget only).
// (xrip, upstream PR #1937 + b0c4d361 / 65f3fd00.)
//
// strata-nvfp4 additions (extras/dialog-park): an engine killed without QUIT (the tray stops it with taskkill /F /T)
// never reaches the QUIT-time save, so the engine also writes on its own while it is idle (write_now: on the
// caller's thread, given up within one 16 MiB block when a request arrives) - the live conversation and every parked
// one that has no copy on the disk yet.  An entry a request reads back may stay on the disk (take(..., keep)): it is
// still a valid resume point if the engine dies before its next save, and a newer copy of the same conversation
// removes it once that copy is written (supersede, the RAM cache's #342 rule).  The index carries a hash.  The
// background writer starts a job only while the engine is idle (set_busy) unless RAM is waited for.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace strata::core {

struct ConversationDiskOptions {
    std::string dir;                 ///< empty: the tier is off
    uint64_t budget_bytes = 0;       ///< at most this many bytes of files; 0: off
    int64_t ttl_s = 30 * 60;         ///< a file older than this is removed (<= 0: kept until the budget evicts it)
    uint64_t min_free_bytes = 0;     ///< a write needs the file plus this much free on the disk (session preflight)
    bool persist = false;            ///< keep the files for later runs, bound to `identity`
    SessionFileIdentity identity;    ///< persist: the session files' identity (model and settings fingerprints)
    /// (strata-nvfp4) several engines use the folder at once - the replicas of one server, one per GPU - as one pool:
    /// rescan() takes in what the others wrote, and open() leaves alone the files of other identities and the young
    /// leftovers another engine may still be writing.  Needs `persist` (the files are bound to the identity).
    bool shared = false;
    /// Called with every diagnostic line (no trailing newline), from the caller's or the writer's thread.
    std::function<void(const std::string&)> log;
    /// Tests: called on the writer thread before each write starts.
    std::function<void()> before_write;
};

/// What the tier did since it started (diagnostics).
struct ConversationDiskStats {
    size_t queued = 0, written = 0, write_failed = 0, ram_hits = 0, disk_hits = 0, read_failed = 0;
    size_t expired = 0, evicted = 0, waits = 0;
    size_t synced = 0, sync_cancelled = 0, superseded = 0, kept = 0;   ///< write_now; supersede; take(keep)
    uint64_t bytes_written = 0, bytes_read = 0;
    uint64_t tokens_back = 0;                 ///< prompt tokens given back (not read again) by ram or disk hits
    double write_ms = 0, read_ms = 0, wait_ms = 0;
};

class ConversationDiskTier {
public:
    using Clock = std::chrono::steady_clock;

    explicit ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now = Clock::now);
    ~ConversationDiskTier();
    ConversationDiskTier(const ConversationDiskTier&) = delete;
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;

    bool enabled() const { return open_; }
    bool persistent() const { return open_ && options_.persist; }
    bool shared() const { return open_ && options_.shared; }
    /// Creates the folder (0700 on POSIX) and starts the writer.  This tier's files a previous run left there are
    /// removed, or with `persist` taken in when their identity matches (the others removed).  False (and the tier
    /// stays off) when the folder cannot be used.
    bool open(std::string& error);

    /// Queues `image` for the writer; its RAM is freed when its write ends, stored or not.  Every checkpoint is written
    /// with it (they hold no K/V), so a prompt that leaves the conversation early still resumes from one.  False when
    /// the tier is off or over budget.  `flush` (RAM pressure): the file is flushed before its RAM is counted as back -
    /// unflushed, Windows keeps the written bytes as modified file cache, which the available figure leaves out.
    bool put(SavedConversation&& image, bool flush = false);

    struct Match {
        uint64_t id = 0;        ///< the entry (stable while the writer runs; 0 = none)
        int64_t tokens = 0;
        bool live = false;
    };
    /// The longest prefix of `prompt` an entry holds (its live state or its checkpoint), queued or on the disk, as
    /// ConversationCache::best; ties prefer the newest entry.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lock(m_);
        Match best;
        auto scan = [&](const Entry& e) {
            if (e.cvec != cvec) return;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens || (n == best.tokens && n > 0 && e.id > best.id)) best = {e.id, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        };
        for (const auto& e : entries_) scan(e);
        for (const auto& j : queue_) scan(j.entry);
        if (writing_) scan(writing_entry_);   // its file appears when the write ends: take() waits for it
        return best;
    }

    /// Hands entry `id` back in `out` (its stage images included) and removes it from the tier either way: from RAM
    /// when it is still queued, else from the disk (waiting for its write when it is being written).  `admit(need,
    /// why)` is asked before each file's payload is allocated (session_file_read's RAM preflight).  `tokens` is what
    /// the request reuses (diagnostics).  On failure `out` is empty and `error` says why.
    /// `keep`: an entry on the disk stays there (its files are read, not removed) and becomes the newest - a request
    /// that borrows a sibling's prefix, or (persist) a resume point that must outlive a crash until a newer copy of
    /// the conversation is written.  A queued entry still leaves the queue (its RAM image is handed over).
    bool take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit = {}, bool keep = false);

    /// Writes `image` NOW on the caller's thread (no queue), then registers it, makes room for the budget and removes
    /// the older entries it supersedes - in that order, so a crash at any point leaves at least one copy.  `sources`
    /// (optional): the first stage's K/V streamed from the device instead of `image.kv` (session_file_write's
    /// streaming form; `image.kv` must then be empty).  `cancel` is asked before every 16 MiB block: true gives the
    /// write up (`*cancelled`, temporary files removed, nothing registered).  An entry with exactly these keys already
    /// on the disk is not written again: its id comes back.  `id` = the entry.  False with `error` on a failure.
    bool write_now(const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                   const std::function<bool()>& cancel, uint64_t& id, std::string& error, bool* cancelled = nullptr);
    /// Whether entry `id` is on the disk (or queued / being written).
    bool has(uint64_t id) const;
    /// `shared`: the entries other engines published since the last look come in (their index files), and the ones
    /// another engine removed (index gone) leave this list.  Cheap (one folder listing); returns how many came in.
    size_t rescan();
    /// The engine runs a request (true) or is idle (false): the background writer starts a queued job only while
    /// idle, or when someone waits for it (wait_idle, take).
    void set_busy(bool busy);

    /// Whether images wait in the queue or are being written (their RAM is not back yet).
    bool pending() const;
    /// Blocks until the queue is empty and no write runs; returns the ms it waited.
    double wait_idle();
    /// #879: every conversation goes - queued, being written (waited for) and on the disk, files too.  Returns how many.
    size_t clear();

    /// Removes the entries older than the time limit; returns how many.
    size_t expire();

    size_t size() const;                    ///< entries on the disk
    uint64_t bytes() const;                 ///< their files' bytes
    ConversationDiskStats stats() const;
    /// One line of totals for the log.
    std::string summary() const;
    /// The file name prefix of this tier (tests).
    static const char* file_prefix() { return "strata-conv-"; }

private:
    struct Entry {
        uint64_t id = 0;
        std::vector<std::string> files;          ///< the first stage's image, then one per later stage
        ConversationCheckpoint live;             ///< ids and images only: what best() matches
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        uint64_t bytes = 0;                      ///< the files' bytes (queued: the image's estimate)
        Clock::time_point at;
        std::string index;                       ///< persist: the index file
        int64_t written_s = 0;                   ///< persist: when it was written (seconds since the epoch)
    };
    struct Job {
        Entry entry;
        SavedConversation image;
        bool flush = false;                      ///< put(flush): written durably
    };
    void writer();
    void write(Job& job);                                    // without the lock
    // the files of `img` (one per stage) under entry `e`'s id; `sources`: stage 0's K/V streamed; without the lock
    bool write_files(Entry& e, const SavedConversation& img, const std::vector<SessionKvSource>* sources,
                     const std::function<bool()>& cancel, std::string& error, bool flush = false);
    // a written entry joins the list: the budget and the entries it supersedes go after it (m_ held); log part back
    std::string publish_locked(Entry&& e, size_t& made_room, size_t& dropped);
    void remove_locked(size_t index, bool delete_files);     // m_ held
    size_t expire_locked();                                  // m_ held
    size_t find_locked(uint64_t id) const;                   // m_ held; entries_.size() when absent
    void log(const std::string& line) const;
    SessionFileIdentity file_identity() const;
    bool write_index(Entry& e, std::string& error) const;
    bool read_index(const std::string& path, Entry& e) const;
    std::string scan_folder(size_t& stale, std::error_code& ec);   // open(): take in or remove; its log part
    std::set<std::string> seen_index_;   ///< shared: index files looked at already (taken in, or another identity's)

    ConversationDiskOptions options_;
    std::function<Clock::time_point()> now_;
    bool open_ = false;
    uint64_t nonce_ = 0, next_id_ = 0, bytes_ = 0;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Entry> entries_;   ///< on the disk, oldest first
    std::deque<Job> queue_;       ///< waiting for the writer, oldest first
    bool writing_ = false;        ///< the writer holds a job (writing_entry_ is its keys)
    Entry writing_entry_;
    bool busy_ = false;           ///< set_busy: the engine runs a request; the writer starts no job then
    int urgent_ = 0;              ///< callers waiting for the writer (wait_idle, take): it starts a job even when busy
    uint64_t reading_ = 0;        ///< take(keep) reads this entry's files: the budget and supersede leave it alone
    bool stop_ = false;
    std::thread thread_;
    ConversationDiskStats stats_;
};

}  // namespace strata::core
