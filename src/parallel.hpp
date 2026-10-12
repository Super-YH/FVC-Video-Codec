// FVC 内部: 単純な並列 for (0..n-1 の各 i を f(i) で処理、スレッド数は論理コア数)
#pragma once
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace fvc {

template <class F>
void parallel_for(int n, F&& f) {
    static const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int nth = std::min(hw, n);
    if (nth <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::atomic<int> next{0};
    auto work = [&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); };
    std::vector<std::thread> pool;
    for (int k = 1; k < nth; ++k) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
}

}  // namespace fvc
