// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include <roaring/roaring.h>
#include <rocksdb/slice.h>
#include "db/common/constants.h"
#include "fts_column/posting/bitpacked_posting_list.h"

namespace zvec::detail {

// Merge borrowed serialized postings one high-16-bit container at a time.
// Large merges use an 8 KiB bitmap, a 128-byte occupancy map, and O(operand
// count) cursors. A bounded fast path retains native Roaring operations for
// inputs with at most two containers and 8 KiB of encoded data in total.
// No decoded bitmap grows without limit with posting cardinality. The final
// string must remain RocksDB-owned (its MergeOperator API cannot accept pool
// pages). Do not turn a soft cache-budget miss into FullMerge=false: RocksDB
// interprets that as a merge error, including on reads and background
// compaction.
class PostingMerge {
  static constexpr uint32_t kEnd = 1u << 16;
  static constexpr uint32_t kCookie = 12347, kNoRunCookie = 12346;
  template <typename T>
  static T load(const char *p) {
    T value;
    std::memcpy(&value, p, sizeof(value));
    return value;
  }
  template <typename T>
  static void store(char *p, T value) {
    std::memcpy(p, &value, sizeof(value));
  }
  static uint32_t popcount(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    return static_cast<uint32_t>(
        (((x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL) * 0x0101010101010101ULL) >>
        56);
  }
  // Sparse postings must not scan/clear all 1024 words for every high key.
  struct Bits {
    std::array<uint64_t, 1024> words{};
    std::array<uint64_t, 16> active{};
    bool dense{false};
    void add(size_t index, uint64_t value) {
      if (!value) return;
      words[index] |= value;
      active[index / 64] |= uint64_t{1} << (index % 64);
    }
    template <typename Emit>
    void each_word(Emit emit) const {
      if (dense) {
        for (uint32_t i = 0; i < words.size(); ++i) emit(i, words[i]);
        return;
      }
      for (uint32_t i = 0; i < active.size(); ++i) {
        auto mask = active[i];
        while (mask) {
          const uint32_t index = i * 64 + popcount((mask & (~mask + 1)) - 1);
          emit(index, words[index]);
          mask &= mask - 1;
        }
      }
    }
    void clear() {
      if (dense)
        words.fill(0);
      else
        each_word([&](uint32_t index, uint64_t) { words[index] = 0; });
      active.fill(0);
      dense = false;
    }
  };
  static void set(Bits &bits, uint32_t id) {
    bits.add((id & 65535) / 64, uint64_t{1} << (id % 64));
  }
  static void set_range(Bits &bits, uint32_t begin, uint32_t end) {
    while (begin <= end) {
      const uint32_t last = std::min(end, begin | 63u);
      bits.add(begin / 64, (~uint64_t{0} << (begin % 64)) &
                               (~uint64_t{0} >> (63 - last % 64)));
      begin = last + 1;
    }
  }
  struct Cursor {
    const char *raw{}, *headers{}, *flags{}, *first{}, *payload{};
    size_t size{};
    uint32_t count{}, index{}, doc{};
    std::array<uint32_t, INVERT_ID_LIST_SIZE_THRESHOLD> ids{};
    std::unique_ptr<fts::BitPackedPostingIterator> packed;
    bool small{false};

    bool open(rocksdb::Slice input, bool scalar) {
      raw = input.data();
      size = input.size();
      if (scalar) {
        if (!size) return false;
        count = static_cast<unsigned char>(*raw++);
        --size;
        if (count > INVERT_ID_LIST_SIZE_THRESHOLD) return false;
        if (count) {
          if (size != count * sizeof(uint32_t)) return false;
          small = true;
          for (uint32_t i = 0; i < count; ++i)
            ids[i] = load<uint32_t>(raw + i * 4);
          std::sort(ids.begin(), ids.begin() + count);
          return true;
        }
      } else if (fts::BitPackedPostingList::is_bitpacked_format(raw, size)) {
        packed = std::make_unique<fts::BitPackedPostingIterator>();
        return reset();
      }
      if (size < 4) return false;
      const uint32_t cookie = load<uint32_t>(raw);
      if (cookie == kNoRunCookie) {
        if (size < 8) return false;
        count = load<uint32_t>(raw + 4);
        headers = raw + 8;
      } else if ((cookie & 65535) == kCookie) {
        count = (cookie >> 16) + 1;
        if (size < 4 + (count + 7) / 8) return false;
        flags = raw + 4;
        headers = flags + (count + 7) / 8;
      } else
        return false;
      if (count > kEnd ||
          roaring_bitmap_portable_deserialize_size(raw, size) != size)
        return false;
      first = headers + count * 4 + ((!flags || count >= 4) ? count * 4 : 0);
      for (uint32_t i = 1; i < count; ++i)
        if (load<uint16_t>(headers + i * 4) <=
            load<uint16_t>(headers + (i - 1) * 4))
          return false;
      payload = first;
      return true;
    }
    bool reset() {
      index = 0;
      payload = first;
      if (packed) {
        if (packed->open(raw, size) != 0) return false;
        doc = packed->next_doc();
        if (doc == fts::BitPackedPostingIterator::NO_MORE_DOCS &&
            packed->cost())
          return false;
      }
      return true;
    }
    uint32_t key() const {
      if (packed)
        return doc == fts::BitPackedPostingIterator::NO_MORE_DOCS ? kEnd
                                                                  : doc >> 16;
      if (index == count) return kEnd;
      return small ? ids[index] >> 16 : load<uint16_t>(headers + index * 4);
    }
    bool merge(uint32_t high, Bits &bits) {
      if (packed) {
        while (key() == high) {
          set(bits, doc);
          ++index;
          const uint32_t previous = doc;
          doc = packed->next_doc();
          if (doc <= previous) return false;
        }
        if (doc == fts::BitPackedPostingIterator::NO_MORE_DOCS &&
            index != packed->cost())
          return false;
      } else if (small) {
        while (key() == high) set(bits, ids[index++]);
      } else {
        const uint32_t card =
            uint32_t(load<uint16_t>(headers + index * 4 + 2)) + 1;
        const bool run =
            flags && (static_cast<unsigned char>(flags[index / 8]) &
                      (1u << (index % 8)));
        if (run) {
          const uint32_t runs = load<uint16_t>(payload);
          payload += 2;
          for (uint32_t r = 0; r < runs; ++r, payload += 4) {
            const uint32_t begin = load<uint16_t>(payload);
            const uint32_t end = begin + load<uint16_t>(payload + 2);
            if (end >= kEnd) return false;
            set_range(bits, begin, end);
          }
        } else if (card > 4096) {
          bits.dense = true;
          for (size_t i = 0; i < bits.words.size(); ++i)
            bits.words[i] |= load<uint64_t>(payload + i * 8);
          payload += 8192;
        } else {
          for (uint32_t i = 0; i < card; ++i)
            set(bits, load<uint16_t>(payload + i * 2));
          payload += card * 2;
        }
        ++index;
      }
      return true;
    }
  };
  struct Container {
    uint32_t card{0}, runs{0};
    bool run() const {
      return 2 + 4 * runs < std::min(card * 2, 8192u);
    }
    size_t bytes() const {
      return run() ? 2 + 4 * runs : std::min(card * 2, 8192u);
    }
  };
  template <typename Emit>
  static bool walk(std::vector<Cursor> &cursors, Emit emit) {
    for (auto &cursor : cursors)
      if (!cursor.reset()) return false;
    Bits bits;
    for (;;) {
      uint32_t key = kEnd;
      for (const auto &cursor : cursors) key = std::min(key, cursor.key());
      if (key == kEnd) return true;
      bits.clear();
      for (auto &cursor : cursors)
        if (cursor.key() == key && !cursor.merge(key, bits)) return false;
      Container container;
      uint64_t carry = 0;
      auto add_word = [&](uint64_t word) {
        container.card += popcount(word);
        container.runs += popcount(word & ~((word << 1) | carry));
        carry = word >> 63;
      };
      if (bits.dense) {
        for (auto word : bits.words) add_word(word);
      } else {
        uint32_t previous = 0;
        bits.each_word([&](uint32_t index, uint64_t word) {
          if (previous + 1 != index) carry = 0;
          add_word(word);
          previous = index;
        });
      }
      if (!container.card || !emit(key, bits, container)) return false;
    }
  }
  template <typename Emit>
  static void each_id(uint32_t key, const Bits &bits, Emit emit) {
    bits.each_word([&](uint32_t w, uint64_t word) {
      while (word) {
        const uint32_t bit = popcount((word & (~word + 1)) - 1);
        emit((key << 16) | (w * 64 + bit));
        word &= word - 1;
      }
    });
  }
  template <typename Emit>
  static void each_run(const Bits &bits, Emit emit) {
    uint64_t carry = 0;
    uint32_t start = 0, previous = 0;
    bits.each_word([&](uint32_t w, uint64_t word) {
      if (previous + 1 != w) {
        if (carry) emit(start, (previous + 1) * 64 - start - 1);
        carry = 0;
      }
      uint64_t transitions = word ^ ((word << 1) | carry);
      while (transitions) {
        const uint64_t bit = transitions & (~transitions + 1);
        const uint32_t id = w * 64 + popcount(bit - 1);
        if (word & bit)
          start = id;
        else
          emit(start, id - start - 1);
        transitions &= transitions - 1;
      }
      carry = word >> 63;
      previous = w;
    });
    if (carry) emit(start, (previous + 1) * 64 - start - 1);
  }

  static bool small_roaring_merge(const std::vector<Cursor> &cursors,
                                  bool scalar, std::string *out) {
    using Bitmap =
        std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)>;
    Bitmap result(nullptr, roaring_bitmap_free);
    for (const auto &cursor : cursors) {
      Bitmap next(
          roaring_bitmap_portable_deserialize_safe(cursor.raw, cursor.size),
          roaring_bitmap_free);
      if (!next) return false;
      if (result)
        roaring_bitmap_or_inplace(result.get(), next.get());
      else
        result = std::move(next);
    }
    const auto cardinality = roaring_bitmap_get_cardinality(result.get());
    if (scalar && !cardinality) return false;
    if (scalar && cardinality <= INVERT_ID_LIST_SIZE_THRESHOLD) {
      std::array<uint32_t, INVERT_ID_LIST_SIZE_THRESHOLD> ids{};
      roaring_bitmap_to_uint32_array(result.get(), ids.data());
      out->resize(1 + cardinality * sizeof(uint32_t));
      (*out)[0] = static_cast<char>(cardinality);
      std::memcpy(out->data() + 1, ids.data(), cardinality * sizeof(uint32_t));
    } else {
      roaring_bitmap_run_optimize(result.get());
      out->resize(size_t(scalar) +
                  roaring_bitmap_portable_size_in_bytes(result.get()));
      if (scalar) (*out)[0] = 0;
      roaring_bitmap_portable_serialize(result.get(),
                                        out->data() + size_t(scalar));
    }
    return true;
  }

 public:
  static bool merge(const rocksdb::Slice *inputs, size_t count,
                    const rocksdb::Slice *existing, bool scalar,
                    std::string *out) try {
    std::vector<Cursor> cursors(count + (existing != nullptr));
    for (size_t i = 0; i < cursors.size(); ++i)
      if (!cursors[i].open(i < count ? inputs[i] : *existing, scalar))
        return false;
    size_t small_containers = 0, small_bytes = 0;
    bool small = !cursors.empty();
    for (const auto &cursor : cursors) {
      if (cursor.small || cursor.packed || cursor.count > 2 ||
          cursor.size > 8192) {
        small = false;
        break;
      }
      small_containers += cursor.count;
      small_bytes += cursor.size;
      if (small_containers > 2 || small_bytes > 8192) {
        small = false;
        break;
      }
    }
    if (small) return small_roaring_merge(cursors, scalar, out);
    uint32_t containers = 0;
    uint64_t cardinality = 0;
    size_t payload_size = 0;
    if (!walk(cursors, [&](uint32_t, const Bits &, const Container &c) {
          ++containers;
          cardinality += c.card;
          payload_size += c.bytes();
          return true;
        }))
      return false;
    if (scalar && !cardinality) return false;
    if (scalar && cardinality <= INVERT_ID_LIST_SIZE_THRESHOLD) {
      out->resize(1 + cardinality * 4);
      (*out)[0] = static_cast<char>(cardinality);
      size_t offset = 1;
      return walk(cursors,
                  [&](uint32_t key, const Bits &bits, const Container &) {
                    each_id(key, bits, [&](uint32_t id) {
                      store(out->data() + offset, id);
                      offset += 4;
                    });
                    return true;
                  });
    }
    if (!containers) {
      out->resize(8);
      store(out->data(), kNoRunCookie);
      store(out->data() + 4, uint32_t{0});
      return true;
    }
    const size_t flags_size = (containers + 7) / 8;
    const size_t header_size = 4 + flags_size + containers * 4 +
                               (containers >= 4 ? containers * 4 : 0);
    out->resize(size_t(scalar) + header_size + payload_size);
    char *raw = out->data() + size_t(scalar);
    if (scalar) (*out)[0] = 0;
    std::memset(raw, 0, header_size);
    store(raw, kCookie | ((containers - 1) << 16));
    size_t offset = header_size;
    uint32_t index = 0;
    return walk(
        cursors, [&](uint32_t key, const Bits &bits, const Container &c) {
          store(raw + 4 + flags_size + index * 4, static_cast<uint16_t>(key));
          store(raw + 4 + flags_size + index * 4 + 2,
                static_cast<uint16_t>(c.card - 1));
          if (containers >= 4)
            store(raw + 4 + flags_size + containers * 4 + index * 4,
                  static_cast<uint32_t>(offset));
          if (c.run()) {
            raw[4 + index / 8] |= static_cast<char>(1u << (index % 8));
            store(raw + offset, static_cast<uint16_t>(c.runs));
            offset += 2;
            each_run(bits, [&](uint32_t start, uint32_t length) {
              store(raw + offset, static_cast<uint16_t>(start));
              store(raw + offset + 2, static_cast<uint16_t>(length));
              offset += 4;
            });
          } else if (c.card > 4096) {
            std::memcpy(raw + offset, bits.words.data(), 8192);
            offset += 8192;
          } else
            each_id(key, bits, [&](uint32_t id) {
              store(raw + offset, static_cast<uint16_t>(id));
              offset += 2;
            });
          ++index;
          return true;
        });
  } catch (const std::exception &) {
    return false;
  }
};
}  // namespace zvec::detail
