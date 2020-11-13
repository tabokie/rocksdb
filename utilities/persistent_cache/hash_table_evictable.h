//  Copyright (c) 2013, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
#pragma once

#ifndef ROCKSDB_LITE

#include <functional>

#include <assert.h>
#include <list>
#include <vector>

#ifdef OS_LINUX
#include <sys/mman.h>
#endif

#include "include/rocksdb/env.h"
#include "util/mutexlock.h"

#include "util/random.h"
#include "utilities/persistent_cache/lrulist.h"

namespace rocksdb {

// Evictable Hash Table
//
// Hash table index where least accessed (or one of the least accessed) elements
// can be evicted.
//
// Please note EvictableHashTable can only be created for pointer type objects
template <class T, class Hash, class Equal>
class EvictableHashTable : private HashTable<T*, Hash, Equal> {
  using PtrT = T*;

 public:
  explicit EvictableHashTable(const size_t capacity = 1024 * 1024,
                              const float load_factor = 2.0,
                              const uint32_t nlocks = 256)
      : nbuckets_(
            static_cast<uint32_t>(load_factor ? capacity / load_factor : 0)),
        nlocks_(nlocks),
        lru_lists_(new LRUList<T>[nlocks]) {
    fprintf(stderr,
            "initialize HashTable2 with capacity = %lu, load_factor = %f, "
            "nlocks = %u, nbuckets = %u\n",
            capacity, load_factor, nlocks, nbuckets_);
    // pre-conditions
    assert(capacity);
    assert(load_factor);
    assert(nbuckets_);
    assert(nlocks_);

    buckets_.reset(new Bucket[nbuckets_]);
#ifdef OS_LINUX
    mlock(buckets_.get(), nbuckets_ * sizeof(Bucket));
#endif

    // initialize locks
    locks_.reset(new port::RWMutex[nlocks_]);
    fprintf(stderr, "hash table initialize lock array %lu\n",
            reinterpret_cast<uint64_t>(locks_.get()));
#ifdef OS_LINUX
    mlock(locks_.get(), nlocks_ * sizeof(port::RWMutex));
#endif

    // post-conditions
    assert(buckets_);
    assert(locks_);
    fprintf(stderr,
            "initialize EvictableHashTable with capacity = %lu, load = %f, "
            "nlocks = %u\n",
            capacity, load_factor, nlocks);
    fprintf(stderr, "check base class members: nlocks = %u, nbucket = %u\n",
            nlocks_, nbuckets_);
    assert(lru_lists_);
  }

  virtual ~EvictableHashTable() {
    AssertEmptyLRU();
    AssertEmptyBuckets();
  }

  //
  // Insert given record to hash table (and LRU list)
  //
  bool Insert(T* t) {
    const uint64_t h = Hash()(t);
    auto& bucket = GetBucket(h);
    LRUListType& lru = GetLRUList(h);
    port::RWMutex& lock = GetMutex(h);

    fprintf(stderr, "insert to evictable hash table %lu with lock %lu\n",
            reinterpret_cast<uint64_t>(t), reinterpret_cast<uint64_t>(&lock));
    WriteLock _(&lock);
    if (Insert(&bucket, t)) {
      lru.Push(t);
      return true;
    }
    return false;
  }

  //
  // Lookup hash table
  //
  // Please note that read lock should be held by the caller. This is because
  // the caller owns the data, and should hold the read lock as long as he
  // operates on the data.
  bool Find(T* t, T** ret) {
    const uint64_t h = Hash()(t);
    auto& bucket = GetBucket(h);
    LRUListType& lru = GetLRUList(h);
    port::RWMutex& lock = GetMutex(h);

    ReadLock _(&lock);
    if (Find(&bucket, t, ret)) {
      ++(*ret)->refs_;
      lru.Touch(*ret);
      return true;
    }
    return false;
  }

  //
  // Evict one of the least recently used object
  //
  T* Evict(const std::function<void(T*)>& fn = nullptr) {
    uint32_t random = Random::GetTLSInstance()->Next();
    const size_t start_idx = random % nlocks_;
    T* t = nullptr;

    // iterate from start_idx .. 0 .. start_idx
    for (size_t i = 0; !t && i < nlocks_; ++i) {
      const size_t idx = (start_idx + i) % nlocks_;

      WriteLock _(&locks_[idx]);
      LRUListType& lru = lru_lists_[idx];
      if (!lru.IsEmpty() && (t = lru.Pop()) != nullptr) {
        assert(!t->refs_);
        // We got an item to evict, erase from the bucket
        const uint64_t h = Hash()(t);
        auto& bucket = GetBucket(h);
        T* tmp = nullptr;
        bool status = Erase(&bucket, t, &tmp);
        assert(t == tmp);
        (void)status;
        assert(status);
        if (fn) {
          fn(t);
        }
        break;
      }
      assert(!t);
    }
    return t;
  }

  void Clear(void (*fn)(T*)) {
    for (uint32_t i = 0; i < nbuckets_; ++i) {
      const uint32_t lock_idx = i % nlocks_;
      WriteLock _(&locks_[lock_idx]);
      auto& lru_list = lru_lists_[lock_idx];
      auto& bucket = buckets_[i];
      for (auto* t : bucket.list_) {
        lru_list.Unlink(t);
        (*fn)(t);
      }
      bucket.list_.clear();
    }
    // make sure that all LRU lists are emptied
    AssertEmptyLRU();
  }

  void AssertEmptyLRU() {
#ifndef NDEBUG
    for (uint32_t i = 0; i < nlocks_; ++i) {
      WriteLock _(&locks_[i]);
      auto& lru_list = lru_lists_[i];
      assert(lru_list.IsEmpty());
    }
#endif
  }

 private:
  typedef LRUList<T> LRUListType;
  // Models bucket of keys that hash to the same bucket number
  struct Bucket {
    std::list<PtrT> list_;
  };

  Bucket& GetBucket(const uint64_t h) {
    const uint32_t bucket_idx = h % nbuckets_;
    return buckets_[bucket_idx];
  }

  LRUListType& GetLRUList(const uint64_t h) {
    const uint32_t bucket_idx = h % nbuckets_;
    const uint32_t lock_idx = bucket_idx % nlocks_;
    return lru_lists_[lock_idx];
  }

  port::RWMutex& GetMutex(const uint64_t h) {
    const uint32_t bucket_idx = h % nbuckets_;
    const uint32_t lock_idx = bucket_idx % nlocks_;
    fprintf(stderr, "h = %lu, nbuckets_ = %u, bucket_idx = %u, nlock = %u\n", h,
            nbuckets_, bucket_idx, nlocks_);
    fprintf(stderr, "GetMutex() lock array %lu with index %u\n",
            reinterpret_cast<uint64_t>(locks_.get()), lock_idx);
    return locks_[lock_idx];
  }


  // Base Class //
  //
  // Erase a given key from the hash table
  //
  bool Erase(const PtrT& t, PtrT* ret) {
    const uint64_t h = Hash()(t);
    const uint32_t bucket_idx = h % nbuckets_;
    const uint32_t lock_idx = bucket_idx % nlocks_;

    WriteLock _(&locks_[lock_idx]);

    auto& bucket = buckets_[bucket_idx];
    return Erase(&bucket, t, ret);
  }

  // Fetch the mutex associated with a key
  // This call is used to hold the lock for a given data for extended period of
  // time.
  port::RWMutex* GetMutex(const PtrT& t) {
    const uint64_t h = Hash()(t);
    const uint32_t bucket_idx = h % nbuckets_;
    const uint32_t lock_idx = bucket_idx % nlocks_;
    fprintf(stderr, "h = %u, nbuckets_ = %u, bucket_idx = %u, nlock = %u\n", h,
            nbuckets_, bucket_idx, nlocks_);
    fprintf(stderr, "GetMutex() lock array %lu with index %u\n",
            reinterpret_cast<uint64_t>(locks_.get()), lock_idx);

    return &locks_[lock_idx];
  }

 protected:


  // Substitute for std::find with custom comparator operator
  typename std::list<PtrT>::iterator Find(std::list<PtrT>* list,
                                          const PtrT& t) {
    for (auto it = list->begin(); it != list->end(); ++it) {
      if (Equal()(*it, t)) {
        return it;
      }
    }
    return list->end();
  }

  bool Insert(Bucket* bucket, const PtrT& t) {
    // Check if the key already exists
    auto it = Find(&bucket->list_, t);
    if (it != bucket->list_.end()) {
      return false;
    }

    // insert to bucket
    bucket->list_.push_back(t);
    return true;
  }

  bool Find(Bucket* bucket, const PtrT& t, PtrT* ret) {
    auto it = Find(&bucket->list_, t);
    if (it != bucket->list_.end()) {
      if (ret) {
        *ret = *it;
      }
      return true;
    }
    return false;
  }

  bool Erase(Bucket* bucket, const PtrT& t, PtrT* ret) {
    auto it = Find(&bucket->list_, t);
    if (it != bucket->list_.end()) {
      if (ret) {
        *ret = *it;
      }

      bucket->list_.erase(it);
      return true;
    }
    return false;
  }

  // assert that all buckets are empty
  void AssertEmptyBuckets() {
#ifndef NDEBUG
    for (size_t i = 0; i < nbuckets_; ++i) {
      WriteLock _(&locks_[i % nlocks_]);
      assert(buckets_[i].list_.empty());
    }
#endif
  }

  const uint32_t nbuckets_;                 // No. of buckets in the spine
  std::unique_ptr<Bucket[]> buckets_;       // Spine of the hash buckets
  const uint32_t nlocks_;                   // No. of locks
  std::unique_ptr<port::RWMutex[]> locks_;  // Granular locks


  std::unique_ptr<LRUListType[]> lru_lists_;
};

}  // namespace rocksdb

#endif
