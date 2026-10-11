// FVC 内部: ブロック辞書 (仕様 §10)。最大 65536 項目、0..4095 は予約 (規範生成)。
#pragma once
#include <cstdint>
#include <vector>

#include "fvc/frame.hpp"

namespace fvc {

class Dictionary {
public:
    static constexpr int kCapacity = 65536;
    static constexpr int kReserved = 4096;
    static constexpr int kSizes[2] = {8, 16};  // 辞書予測を許すブロックサイズ

    Dictionary();
    // サイズ s の項目 ID 一覧 (検索・符号化用、昇順ではなく登録順)
    const std::vector<int>& ids_for(int s) const { return by_size_[s == 8 ? 0 : 1]; }
    const std::vector<int32_t>& data(int id) const { return entries_[id].d; }
    int size_of(int id) const { return entries_[id].s; }
    bool valid(int id) const { return id >= 0 && id < kCapacity && entries_[id].s > 0; }
    // 再構成から切り出して登録 (ADD_FROM_RECON)。容量超過時は最古の動的項目を置換 (FIFO)。
    int add_from(const Plane& p, int x, int y, int s);
    void reset_dynamic();
    // 項目を捨てる (DISCARD)。予約項目は捨てられない
    void remove(int id) { if (id >= kReserved && id < kCapacity && entries_[id].s) { index_remove(id); entries_[id] = Entry{}; } }
    // 符号器の管理用 (規範外): 最後に使われた (または登録された) 更新番号
    int stamp(int id) const { return stamp_[id]; }
    void touch(int id) { if (id >= 0 && id < kCapacity) stamp_[id] = epoch_; }
    int epoch() const { return epoch_; }
    void next_epoch() { ++epoch_; }
    int count() const { return static_cast<int>(by_size_[0].size() + by_size_[1].size()); }

private:
    struct Entry { int s = 0; std::vector<int32_t> d; };
    std::vector<Entry> entries_;
    std::vector<int> by_size_[2];
    int next_dyn_ = kReserved;
    std::vector<int> stamp_ = std::vector<int>(kCapacity, 0);
    int epoch_ = 0;
    void add_reserved(int id, int s, std::vector<int32_t> d);
    void index_remove(int id);
};

}  // namespace fvc
