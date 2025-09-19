#pragma once

#include <cstddef>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace infinicore::common {
template <typename Key, typename Value>
class LRUCache {
public:
    using KeyValuePair = std::pair<Key, Value>;
    using ListIt = typename std::list<KeyValuePair>::iterator;

    explicit LRUCache(size_t capacity = 100) : capacity_(capacity) {
        if (capacity == 0) {
            capacity_ = UINT64_MAX; // effectively unbounded
        }
    }

    bool contains(const Key &key) const {
        return map_.find(key) != map_.end();
    }

    void put(const Key &key, const Value &value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            // update existing
            it->second->second = value;
            touch(it);
        } else {
            // insert new
            if (list_.size() >= capacity_) {
                // evict least recently used (back of list)
                auto &kv = list_.back();
                map_.erase(kv.first);
                list_.pop_back();
            }
            list_.emplace_front(key, value);
            map_[key] = list_.begin();
        }
    }

    Value &get(const Key &key) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            throw std::out_of_range("key not found");
        }
        touch(it);
        return it->second->second;
    }

    const Value &get(const Key &key) const {
        auto it = map_.find(key);
        if (it == map_.end()) {
            throw std::out_of_range("key not found");
        }
        // can't "touch" in const context → treat this as non-mutating lookup
        return it->second->second;
    }

private:
    void touch(typename std::unordered_map<Key, ListIt>::iterator it) {
        // move this key to front (most recent)
        list_.splice(list_.begin(), list_, it->second);
        it->second = list_.begin();
    }

    size_t capacity_;
    std::list<KeyValuePair> list_; // front = most recent, back = least
    std::unordered_map<Key, ListIt> map_;
};

} // namespace infinicore::common
