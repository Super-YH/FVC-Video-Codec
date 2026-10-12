// FVC 内部: 区間の実時間計測 (環境変数 FVC_PROF があれば終了時に標準エラーへ出力)
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace fvc {

struct ProfSlot {
    const char* name;
    std::atomic<long long> ns{0};
    explicit ProfSlot(const char* n);
};

struct ProfRegistry {
    ProfSlot* slots[64] = {};
    int n = 0;
    ~ProfRegistry() {
        if (!std::getenv("FVC_PROF")) return;
        for (int i = 0; i < n; ++i) std::fprintf(stderr, "[prof] %-16s %8.3f s\n", slots[i]->name, slots[i]->ns.load() * 1e-9);
    }
    static ProfRegistry& get() { static ProfRegistry r; return r; }
};

inline ProfSlot::ProfSlot(const char* n) : name(n) {
    ProfRegistry& r = ProfRegistry::get();
    if (r.n < 64) r.slots[r.n++] = this;
}

struct ProfScope {
    ProfSlot& s;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ProfScope(ProfSlot& sl) : s(sl) {}
    ~ProfScope() { s.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); }
};

}  // namespace fvc

#define FVC_PROF_CAT2(a, b) a##b
#define FVC_PROF_CAT(a, b) FVC_PROF_CAT2(a, b)
#define FVC_PROF(name)                                          \
    static ::fvc::ProfSlot FVC_PROF_CAT(prof_slot_, __LINE__){name}; \
    ::fvc::ProfScope FVC_PROF_CAT(prof_scope_, __LINE__){FVC_PROF_CAT(prof_slot_, __LINE__)}
