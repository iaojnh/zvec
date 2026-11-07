// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <stdexcept>
#include <utility>

namespace zvec::ailego {

// Some standard libraries allocate sentinel nodes even for an empty map.
// Keep an unused directory allocation-free, and destroy it without creating
// a replacement map when releasing its last entry. Allocations still use the
// underlying map's allocator, including any non-reclaiming lock discipline.
template <typename Map>
class LazyMap {
 public:
  using iterator = typename Map::iterator;
  using const_iterator = typename Map::const_iterator;
  using size_type = typename Map::size_type;
  using value_type = typename Map::value_type;

  bool empty() const noexcept {
    return !map_ || map_->empty();
  }
  size_type size() const noexcept {
    return map_ ? map_->size() : 0;
  }
  iterator begin() {
    return map_ ? map_->begin() : iterator{};
  }
  iterator end() {
    return map_ ? map_->end() : iterator{};
  }
  const_iterator begin() const {
    return map_ ? map_->begin() : const_iterator{};
  }
  const_iterator end() const {
    return map_ ? map_->end() : const_iterator{};
  }
  template <typename Key>
  iterator find(const Key &key) {
    return map_ ? map_->find(key) : iterator{};
  }
  template <typename Key>
  const_iterator find(const Key &key) const {
    return map_ ? map_->find(key) : const_iterator{};
  }
  template <typename Key>
  auto &at(const Key &key) {
    if (!map_) throw std::out_of_range("Empty lazy map");
    return map_->at(key);
  }
  template <typename Key>
  auto &operator[](Key &&key) {
    return ensure()[std::forward<Key>(key)];
  }
  template <typename... Args>
  auto emplace(Args &&...args) {
    return ensure().emplace(std::forward<Args>(args)...);
  }
  template <typename... Args>
  auto try_emplace(Args &&...args) {
    return ensure().try_emplace(std::forward<Args>(args)...);
  }
  template <typename Arg>
  auto insert(Arg &&arg) {
    return ensure().insert(std::forward<Arg>(arg));
  }
  template <typename Arg>
  auto erase(Arg &&arg) {
    using Result =
        decltype(std::declval<Map &>().erase(std::forward<Arg>(arg)));
    return map_ ? map_->erase(std::forward<Arg>(arg)) : Result{};
  }
  template <typename Arg>
  auto extract(Arg &&arg) {
    using Result =
        decltype(std::declval<Map &>().extract(std::forward<Arg>(arg)));
    return map_ ? map_->extract(std::forward<Arg>(arg)) : Result{};
  }
  void reserve(size_type count) {
    if (count) ensure().reserve(count);
  }
  size_type bucket_count() const {
    return map_ ? map_->bucket_count() : 0;
  }
  float max_load_factor() const {
    return map_ ? map_->max_load_factor() : 1.0f;
  }
  void reset() noexcept {
    map_.reset();
  }

 private:
  Map &ensure() {
    if (!map_) map_.emplace();
    return *map_;
  }
  std::optional<Map> map_;
};
}  // namespace zvec::ailego
