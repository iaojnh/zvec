// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bitpacked_posting_list.h"
#include <bitpackinghelpers.h>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <zvec/ailego/logger/logger.h>
#include "bitpacked_simd_dispatch.h"

#ifdef _MSC_VER
#include <intrin.h>
#include <malloc.h>
#endif

namespace zvec::fts {

// ============================================================
// BitPacked Posting List on-disk format
// ============================================================
//
// Encodes doc_id deltas, term frequencies, and document lengths using
// per-block bitpacking.  Each block stores up to 128 entries and carries
// a precomputed BM25 score upper bound to support Block-Max WAND pruning.
//
// File layout:
//   [Header 16B] [SkipList N*12B] [Block0] [Block1] ...
//
// Block layout:
//   [BlockHeader 16B] [packed_deltas] [packed_tfs] [packed_dlens]

namespace {

/// Round up \p value to the next multiple of \p alignment.
constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

// ============================================================
// Low-level bitpacking primitives
// ============================================================

uint8_t BitPackedPostingList::bits_needed(uint32_t max_value) {
  if (max_value == 0) return 0;
#ifdef _MSC_VER
  unsigned long index = 0;
  _BitScanReverse(&index, max_value);
  return static_cast<uint8_t>(index + 1);
#else
  return static_cast<uint8_t>(32 - __builtin_clz(max_value));
#endif
}

void BitPackedPostingList::pack_uint32(const uint32_t *in, uint8_t bitwidth,
                                       uint32_t count, uint8_t *out) {
  if (bitwidth == 0 || count == 0) return;

  // Full block path: 128 values at once via dispatch (SIMD or scalar)
  if (count == DOCS_PER_BLOCK) {
    simd::get_dispatch().pack_uint32_128(in, bitwidth, out);
    return;
  }

  // Tail block path (count < 128): use scalar fastpack, 32 at a time
  const size_t total_bytes = packed_byte_size(bitwidth, count);
  std::memset(out, 0, total_bytes);

  uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
  uint32_t offset = 0;

  while (offset + 32 <= count) {
    FastPForLib::fastpackwithoutmask(in + offset, out32, bitwidth);
    out32 += bitwidth;
    offset += 32;
  }

  // Tail: fewer than 32 integers
  if (offset < count) {
    alignas(16) uint32_t padded_in[32] = {};
    std::memcpy(padded_in, in + offset, (count - offset) * sizeof(uint32_t));
    alignas(16) uint32_t padded_out[32] = {};
    FastPForLib::fastpackwithoutmask(padded_in, padded_out, bitwidth);
    size_t tail_bytes = packed_byte_size(bitwidth, count - offset);
    std::memcpy(out32, padded_out, tail_bytes);
  }
}

void BitPackedPostingList::unpack_uint32(const uint8_t *in, uint8_t bitwidth,
                                         uint32_t count, uint32_t *out) {
  if (bitwidth == 0 || count == 0) {
    for (uint32_t i = 0; i < count; ++i) {
      out[i] = 0;
    }
    return;
  }

  // Full block path: 128 values at once via dispatch (SIMD or scalar)
  if (count == DOCS_PER_BLOCK) {
    simd::get_dispatch().unpack_uint32_128(in, bitwidth, out);
    return;
  }

  // Tail block path (count < 128): use scalar fastunpack, 32 at a time
  const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in);
  uint32_t offset = 0;

  while (offset + 32 <= count) {
    FastPForLib::fastunpack(in32, out + offset, bitwidth);
    in32 += bitwidth;
    offset += 32;
  }

  // Tail: fewer than 32 integers
  if (offset < count) {
    const size_t tail_bytes = packed_byte_size(bitwidth, count - offset);
    alignas(16) uint32_t padded_in[32] = {};
    std::memcpy(padded_in, in32, tail_bytes);
    alignas(16) uint32_t padded_out[32] = {};
    FastPForLib::fastunpack(padded_in, padded_out, bitwidth);
    std::memcpy(out + offset, padded_out, (count - offset) * sizeof(uint32_t));
  }
}

// ============================================================
// Encoder
// ============================================================

std::string BitPackedPostingList::encode(const uint32_t *doc_ids,
                                         const uint32_t *tfs,
                                         const uint32_t *doc_lens, size_t count,
                                         uint64_t df,
                                         const BM25Scorer &scorer) {
  return encode_impl(
      count, df, scorer,
      [&](size_t start, size_t n, uint32_t *ids, uint32_t *freqs,
          uint32_t *lengths) {
        std::memcpy(ids, doc_ids + start, n * sizeof(uint32_t));
        std::memcpy(freqs, tfs + start, n * sizeof(uint32_t));
        std::memcpy(lengths, doc_lens + start, n * sizeof(uint32_t));
      },
      nullptr);
}

BitPackedPostingList::EncodedPosting BitPackedPostingList::encode_blocks(
    size_t count, uint64_t df, const BM25Scorer &scorer,
    const BlockReader &reader) {
  EncodedPosting result;
  result.data = encode_impl(count, df, scorer, reader, &result.reservation);
  return result;
}

std::string BitPackedPostingList::encode_impl(
    size_t count, uint64_t df, const BM25Scorer &scorer,
    const BlockReader &reader, ailego::MemoryReservation *reservation) {
  if (count > UINT32_MAX)
    throw std::length_error("Posting count exceeds format");
  const size_t num_blocks =
      count / DOCS_PER_BLOCK + (count % DOCS_PER_BLOCK != 0);
  const size_t data_start =
      align_up(HEADER_SIZE + num_blocks * sizeof(BlockMeta), 16);
  alignas(64) uint32_t ids[DOCS_PER_BLOCK]{};
  alignas(64) uint32_t tfs[DOCS_PER_BLOCK]{};
  alignas(64) uint32_t lengths[DOCS_PER_BLOCK]{};
  alignas(64) uint32_t deltas[DOCS_PER_BLOCK]{};
  uint32_t previous = 0;
  auto load_block = [&](size_t start, uint32_t n, bool score) {
    reader(start, n, ids, tfs, lengths);
    uint32_t max_delta = 0, max_tf = 0, max_dl = 0;
    float max_score = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if ((start || i) && ids[i] < previous)
        throw std::invalid_argument("Unsorted posting IDs");
      deltas[i] = ids[i] - previous;
      previous = ids[i];
      if (score)
        max_score = std::max(max_score, scorer.score(df, tfs[i], lengths[i]));
    }
    if (n == DOCS_PER_BLOCK) {
      simd::get_dispatch().max_128(deltas, tfs, lengths, 0, n, max_delta,
                                   max_tf, max_dl);
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        max_delta = std::max(max_delta, deltas[i]);
        max_tf = std::max(max_tf, tfs[i]);
        max_dl = std::max(max_dl, lengths[i]);
      }
    }
    BlockHeader header{};
    header.min_doc_id = ids[0];
    header.num_docs = static_cast<uint8_t>(n);
    header.bitwidth_id = bits_needed(max_delta);
    header.bitwidth_tf = bits_needed(max_tf);
    header.bitwidth_dl = bits_needed(max_dl);
    header.block_max_score = max_score;
    return header;
  };
  auto block_bytes = [](const BlockHeader &header) {
    return sizeof(BlockHeader) +
           packed_byte_size(header.bitwidth_id, header.num_docs) +
           packed_byte_size(header.bitwidth_tf, header.num_docs) +
           packed_byte_size(header.bitwidth_dl, header.num_docs);
  };

  // Two passes avoid count-sized deltas, metadata and offset arrays. The sole
  // contiguous allocation is the serialized value required by RocksDB::Put.
  size_t total_size = data_start;
  for (size_t start = 0; start < count; start += DOCS_PER_BLOCK) {
    auto n =
        static_cast<uint32_t>(std::min<size_t>(DOCS_PER_BLOCK, count - start));
    auto header = load_block(start, n, false);
    size_t bytes = align_up(block_bytes(header), 16);
    if (total_size > UINT32_MAX - bytes)
      throw std::length_error("Posting bytes exceed format");
    total_size += bytes;
  }
  if (reservation) reservation->add(total_size + 1);
  std::string result(total_size, '\0');
  Header header{MAGIC, VERSION, static_cast<uint32_t>(count),
                static_cast<uint32_t>(num_blocks)};
  std::memcpy(result.data(), &header, HEADER_SIZE);
  previous = 0;
  size_t offset = data_start;
  for (size_t b = 0, start = 0; start < count; ++b, start += DOCS_PER_BLOCK) {
    auto n =
        static_cast<uint32_t>(std::min<size_t>(DOCS_PER_BLOCK, count - start));
    auto block = load_block(start, n, true);
    const size_t bytes = align_up(block_bytes(block), 16);
    if (offset > result.size() || bytes > result.size() - offset)
      throw std::runtime_error("Posting source changed between passes");
    BlockMeta meta{ids[n - 1], static_cast<uint32_t>(offset),
                   block.block_max_score};
    std::memcpy(result.data() + HEADER_SIZE + b * sizeof(BlockMeta), &meta,
                sizeof(meta));
    std::memcpy(result.data() + offset, &block, sizeof(block));
    auto *packed =
        reinterpret_cast<uint8_t *>(result.data() + offset + sizeof(block));
    pack_uint32(deltas, block.bitwidth_id, n, packed);
    packed += packed_byte_size(block.bitwidth_id, n);
    pack_uint32(tfs, block.bitwidth_tf, n, packed);
    packed += packed_byte_size(block.bitwidth_tf, n);
    pack_uint32(lengths, block.bitwidth_dl, n, packed);
    offset += bytes;
  }
  if (offset != total_size)
    throw std::runtime_error("Posting source changed between passes");
  return result;
}

// ============================================================
// Iterator
// ============================================================

int BitPackedPostingIterator::open(const char *data, size_t size) {
  if (!data || size < BitPackedPostingList::HEADER_SIZE) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: truncated data, "
        "size[%zu] expected_min[%zu]",
        size, BitPackedPostingList::HEADER_SIZE);
    return -1;
  }

  // Parse file header
  BitPackedPostingList::Header hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));

  if (hdr.magic != BitPackedPostingList::MAGIC) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: bad magic, "
        "got[0x%x] expected[0x%x]",
        hdr.magic, BitPackedPostingList::MAGIC);
    return -1;
  }
  if (hdr.version != BitPackedPostingList::VERSION) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: unsupported version, "
        "got[%u] expected[%u]",
        hdr.version, BitPackedPostingList::VERSION);
    return -1;
  }

  num_docs_ = hdr.num_docs;
  num_blocks_ = hdr.num_blocks;
  data_ = data;
  data_size_ = size;

  if (num_docs_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return 0;
  }

  // Validate skip list fits
  const size_t skip_list_offset = BitPackedPostingList::HEADER_SIZE;
  const size_t skip_list_size =
      num_blocks_ * sizeof(BitPackedPostingList::BlockMeta);
  if (skip_list_offset + skip_list_size > size) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: skip list overruns buffer, "
        "num_blocks[%u] data_size[%zu] need[%zu]",
        num_blocks_, size, skip_list_offset + skip_list_size);
    return -1;
  }

  skip_list_ = reinterpret_cast<const BitPackedPostingList::BlockMeta *>(
      data + skip_list_offset);

  // Compute global max score
  global_max_score_ = 0.0f;
  for (uint32_t b = 0; b < num_blocks_; ++b) {
    global_max_score_ =
        std::max(global_max_score_, skip_list_[b].block_max_score);
  }

  // Initialize to before-first-block state
  current_block_idx_ = 0;
  in_block_pos_ = 0;
  current_block_size_ = 0;
  block_decoded_ = false;
  current_doc_id_ = NO_MORE_DOCS;

  // Cache SIMD dispatch function pointers to avoid PLT overhead on hot path
  const auto &dispatch = simd::get_dispatch();
  prefix_sum_fn_ = dispatch.prefix_sum_128;
  find_first_ge_fn_ = dispatch.find_first_ge;
  unpack_fn_ = dispatch.unpack_uint32_128;

  return 0;
}

void BitPackedPostingIterator::decode_block(size_t block_idx) {
  if (block_idx >= num_blocks_) {
    LOG_WARN(
        "BitPackedPostingIterator decode_block out of range: "
        "block_idx[%zu] num_blocks[%u]",
        block_idx, num_blocks_);
    current_block_size_ = 0;
    block_decoded_ = false;
    return;
  }

  const auto &meta = skip_list_[block_idx];
  const char *block_ptr = data_ + meta.block_offset;

  // Parse block header
  BitPackedPostingList::BlockHeader bhdr{};
  std::memcpy(&bhdr, block_ptr, sizeof(bhdr));

  current_block_size_ = bhdr.num_docs;
  current_block_idx_ = block_idx;
  in_block_pos_ = 0;

  const uint8_t *packed_ptr =
      reinterpret_cast<const uint8_t *>(block_ptr + sizeof(bhdr));

  const bool is_full_block =
      (bhdr.num_docs == BitPackedPostingList::DOCS_PER_BLOCK);

  // Unpack doc_id deltas
  const size_t id_bytes =
      is_full_block
          ? BitPackedPostingList::simd_packed_byte_size(bhdr.bitwidth_id)
          : BitPackedPostingList::packed_byte_size(bhdr.bitwidth_id,
                                                   bhdr.num_docs);
  alignas(16) uint32_t deltas[BitPackedPostingList::DOCS_PER_BLOCK];
  if (is_full_block) {
    // Fast path: use cached function pointer directly for full blocks
    unpack_fn_(packed_ptr, bhdr.bitwidth_id, deltas);
  } else {
    BitPackedPostingList::unpack_uint32(packed_ptr, bhdr.bitwidth_id,
                                        bhdr.num_docs, deltas);
  }
  packed_ptr += id_bytes;

  // Reconstruct absolute doc_ids from deltas using prefix-sum
  if (is_full_block) {
    prefix_sum_fn_(deltas, bhdr.min_doc_id,
                   BitPackedPostingList::DOCS_PER_BLOCK, block_doc_ids_);
  } else {
    // Scalar prefix-sum for tail block
    block_doc_ids_[0] = bhdr.min_doc_id;
    for (uint32_t i = 1; i < bhdr.num_docs; ++i) {
      block_doc_ids_[i] = block_doc_ids_[i - 1] + deltas[i];
    }
  }

  // Lazy decode: record packed data pointers and bitwidths for tf/doc_len.
  // Actual decoding is deferred until term_freq() or doc_len() is called.
  const size_t tf_bytes =
      is_full_block
          ? BitPackedPostingList::simd_packed_byte_size(bhdr.bitwidth_tf)
          : BitPackedPostingList::packed_byte_size(bhdr.bitwidth_tf,
                                                   bhdr.num_docs);
  packed_tf_ptr_ = packed_ptr;
  current_bitwidth_tf_ = bhdr.bitwidth_tf;
  packed_ptr += tf_bytes;

  packed_dl_ptr_ = packed_ptr;
  current_bitwidth_dl_ = bhdr.bitwidth_dl;

  current_block_num_docs_ = bhdr.num_docs;
  current_block_is_full_ = is_full_block;

  // Reset lazy decode flags
  tf_decoded_ = false;
  dl_decoded_ = false;

  block_decoded_ = true;
}

uint32_t BitPackedPostingIterator::next_doc() {
  if (num_docs_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // If no block is decoded yet, decode the first block
  if (!block_decoded_) {
    decode_block(0);
    if (current_block_size_ == 0) {
      current_doc_id_ = NO_MORE_DOCS;
      return NO_MORE_DOCS;
    }
    current_doc_id_ = block_doc_ids_[0];
    in_block_pos_ = 0;
    return current_doc_id_;
  }

  // Advance within current block
  ++in_block_pos_;
  if (in_block_pos_ < current_block_size_) {
    current_doc_id_ = block_doc_ids_[in_block_pos_];
    return current_doc_id_;
  }

  // Move to next block
  size_t next_block = current_block_idx_ + 1;
  if (next_block >= num_blocks_) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  decode_block(next_block);
  if (current_block_size_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  current_doc_id_ = block_doc_ids_[0];
  in_block_pos_ = 0;
  return current_doc_id_;
}

size_t BitPackedPostingIterator::simd_find_first_ge(uint32_t target,
                                                    size_t start) const {
  return find_first_ge_fn_(block_doc_ids_, current_block_size_, target, start);
}

uint32_t BitPackedPostingIterator::advance(uint32_t target) {
  if (num_docs_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // If current doc_id already >= target, return it
  if (current_doc_id_ != NO_MORE_DOCS && current_doc_id_ >= target) {
    return current_doc_id_;
  }

  // Use skip list to find the target block via binary search.
  // Find the first block whose max_doc_id >= target.
  size_t lo = 0, hi = num_blocks_;

  // If we have a current block and its max_doc_id >= target,
  // we can search within the current block first.
  if (block_decoded_ && current_block_idx_ < num_blocks_ &&
      skip_list_[current_block_idx_].max_doc_id >= target) {
    // Target might be in current block - SIMD scan from current position
    {
      size_t pos = simd_find_first_ge(target, in_block_pos_);
      if (pos < current_block_size_) {
        in_block_pos_ = pos;
        current_doc_id_ = block_doc_ids_[pos];
        return current_doc_id_;
      }
    }
    // Not found in current block (shouldn't happen if skip list is correct)
    lo = current_block_idx_ + 1;
  } else if (block_decoded_) {
    // Current block's max_doc_id < target, start search from next block
    lo = current_block_idx_ + 1;
  }

  // Binary search in skip list for the first block with max_doc_id >= target
  size_t target_block = hi;  // sentinel: no block found
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (skip_list_[mid].max_doc_id >= target) {
      target_block = mid;
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  if (target_block >= num_blocks_) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // Decode the target block
  decode_block(target_block);
  if (current_block_size_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // SIMD scan within the block for the first doc_id >= target
  {
    size_t pos = simd_find_first_ge(target, 0);
    if (pos < current_block_size_) {
      in_block_pos_ = pos;
      current_doc_id_ = block_doc_ids_[pos];
      return current_doc_id_;
    }
  }

  // All docs in this block are < target (shouldn't happen with correct skip
  // list), try next block
  size_t next = target_block + 1;
  if (next >= num_blocks_) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  decode_block(next);
  if (current_block_size_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  {
    size_t pos = simd_find_first_ge(target, 0);
    if (pos < current_block_size_) {
      in_block_pos_ = pos;
      current_doc_id_ = block_doc_ids_[pos];
      return current_doc_id_;
    }
  }
  current_doc_id_ = NO_MORE_DOCS;
  return NO_MORE_DOCS;
}

void BitPackedPostingIterator::ensure_tf_decoded() {
  if (tf_decoded_) {
    return;
  }
  if (current_block_is_full_) {
    unpack_fn_(packed_tf_ptr_, current_bitwidth_tf_, block_tfs_);
  } else {
    BitPackedPostingList::unpack_uint32(packed_tf_ptr_, current_bitwidth_tf_,
                                        current_block_num_docs_, block_tfs_);
  }
  tf_decoded_ = true;
}

void BitPackedPostingIterator::ensure_dl_decoded() {
  if (dl_decoded_) {
    return;
  }
  if (current_block_is_full_) {
    unpack_fn_(packed_dl_ptr_, current_bitwidth_dl_, block_doc_lens_);
  } else {
    BitPackedPostingList::unpack_uint32(packed_dl_ptr_, current_bitwidth_dl_,
                                        current_block_num_docs_,
                                        block_doc_lens_);
  }
  dl_decoded_ = true;
}

uint32_t BitPackedPostingIterator::term_freq() {
  if (!block_decoded_ || in_block_pos_ >= current_block_size_) {
    return 0;
  }
  ensure_tf_decoded();
  return block_tfs_[in_block_pos_];
}

uint32_t BitPackedPostingIterator::doc_len() {
  if (!block_decoded_ || in_block_pos_ >= current_block_size_) {
    return 1;
  }
  ensure_dl_decoded();
  return block_doc_lens_[in_block_pos_];
}

BitPackedPostingIterator::BlockMaxInfo
BitPackedPostingIterator::block_max_info_for(uint32_t target) const {
  if (num_blocks_ == 0 || skip_list_ == nullptr) {
    return {0.0f, NO_MORE_DOCS};
  }

  // Fast path: check if target falls within the previously cached block
  if (cached_bmi_valid_ && target <= cached_bmi_last_doc_) {
    // target is in the same or earlier block as last query.
    // Check if it's still in the same block (block_idx is correct).
    if (cached_bmi_block_idx_ == 0 ||
        target > skip_list_[cached_bmi_block_idx_ - 1].max_doc_id) {
      return {cached_bmi_score_, cached_bmi_last_doc_};
    }
  }

  size_t lo = 0, hi = num_blocks_;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (skip_list_[mid].max_doc_id >= target) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  if (lo >= num_blocks_) {
    return {0.0f, NO_MORE_DOCS};
  }

  // Update cache
  cached_bmi_block_idx_ = lo;
  cached_bmi_last_doc_ = skip_list_[lo].max_doc_id;
  cached_bmi_score_ = skip_list_[lo].block_max_score;
  cached_bmi_valid_ = true;

  return {cached_bmi_score_, cached_bmi_last_doc_};
}

}  // namespace zvec::fts
