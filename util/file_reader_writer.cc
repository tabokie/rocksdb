//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "util/file_reader_writer.h"

#include <algorithm>
#include <mutex>

#include "monitoring/histogram.h"
#include "monitoring/iostats_context_imp.h"
#include "port/port.h"
#include "test_util/sync_point.h"
#include "util/random.h"
#include "util/rate_limiter.h"

namespace rocksdb {

#ifndef NDEBUG
namespace {
bool IsFileSectorAligned(const size_t off, size_t sector_size) {
  return off % sector_size == 0;
}
}
#endif

Status SequentialFileReader::Read(size_t n, Slice* result, char* scratch) {
  Status s;
  if (use_direct_io()) {
#ifndef ROCKSDB_LITE
    size_t offset = offset_.fetch_add(n);
    size_t alignment = file_->GetRequiredBufferAlignment();
    size_t aligned_offset = TruncateToPageBoundary(alignment, offset);
    size_t offset_advance = offset - aligned_offset;
    size_t size = Roundup(offset + n, alignment) - aligned_offset;
    size_t r = 0;
    AlignedBuffer buf;
    buf.Alignment(alignment);
    buf.AllocateNewBuffer(size);
    Slice tmp;
    s = file_->PositionedRead(aligned_offset, size, &tmp, buf.BufferStart());
    if (s.ok() && offset_advance < tmp.size()) {
      buf.Size(tmp.size());
      r = buf.Read(scratch, offset_advance,
                   std::min(tmp.size() - offset_advance, n));
    }
    *result = Slice(scratch, r);
#endif  // !ROCKSDB_LITE
  } else {
    s = file_->Read(n, result, scratch);
  }
  IOSTATS_ADD(bytes_read, result->size());
  return s;
}


Status SequentialFileReader::Skip(uint64_t n) {
#ifndef ROCKSDB_LITE
  if (use_direct_io()) {
    offset_ += static_cast<size_t>(n);
    return Status::OK();
  }
#endif  // !ROCKSDB_LITE
  return file_->Skip(n);
}

Status RandomAccessFileReader::Read(uint64_t offset, size_t n, Slice* result,
                                    char* scratch, bool for_compaction) const {
  Status s;
  uint64_t elapsed = 0;
  {
    StopWatch sw(env_, stats_, hist_type_,
                 (stats_ != nullptr) ? &elapsed : nullptr, true /*overwrite*/,
                true /*delay_enabled*/);
    auto prev_perf_level = GetPerfLevel();
    IOSTATS_TIMER_GUARD(read_nanos);
    if (use_direct_io()) {
#ifndef ROCKSDB_LITE
      size_t alignment = file_->GetRequiredBufferAlignment();
      size_t aligned_offset = TruncateToPageBoundary(alignment, static_cast<size_t>(offset));
      size_t offset_advance = static_cast<size_t>(offset) - aligned_offset;
      size_t read_size = Roundup(static_cast<size_t>(offset + n), alignment) - aligned_offset;
      AlignedBuffer buf;
      buf.Alignment(alignment);
      buf.AllocateNewBuffer(read_size);
      while (buf.CurrentSize() < read_size) {
        size_t allowed;
        if (for_compaction && rate_limiter_ != nullptr) {
          allowed = rate_limiter_->RequestToken(
              buf.Capacity() - buf.CurrentSize(), buf.Alignment(),
              Env::IOPriority::IO_LOW, stats_, RateLimiter::OpType::kRead);
        } else {
          assert(buf.CurrentSize() == 0);
          allowed = read_size;
        }
        Slice tmp;

        FileOperationInfo::TimePoint start_ts;
        uint64_t orig_offset = 0;
        if (ShouldNotifyListeners()) {
          start_ts = std::chrono::system_clock::now();
          orig_offset = aligned_offset + buf.CurrentSize();
        }
        {
          IOSTATS_CPU_TIMER_GUARD(cpu_read_nanos, env_);
          s = file_->Read(aligned_offset + buf.CurrentSize(), allowed, &tmp,
                          buf.Destination());
        }
        if (ShouldNotifyListeners()) {
          auto finish_ts = std::chrono::system_clock::now();
          NotifyOnFileReadFinish(orig_offset, tmp.size(), start_ts, finish_ts,
                                 s);
        }

        buf.Size(buf.CurrentSize() + tmp.size());
        if (!s.ok() || tmp.size() < allowed) {
          break;
        }
      }
      size_t res_len = 0;
      if (s.ok() && offset_advance < buf.CurrentSize()) {
        res_len = buf.Read(scratch, offset_advance,
                           std::min(buf.CurrentSize() - offset_advance, n));
      }
      *result = Slice(scratch, res_len);
#endif  // !ROCKSDB_LITE
    } else {
      size_t pos = 0;
      const char* res_scratch = nullptr;
      while (pos < n) {
        size_t allowed;
        if (for_compaction && rate_limiter_ != nullptr) {
          if (rate_limiter_->IsRateLimited(RateLimiter::OpType::kRead)) {
            sw.DelayStart();
          }
          allowed = rate_limiter_->RequestToken(n - pos, 0 /* alignment */,
                                                Env::IOPriority::IO_LOW, stats_,
                                                RateLimiter::OpType::kRead);
          if (rate_limiter_->IsRateLimited(RateLimiter::OpType::kRead)) {
            sw.DelayStop();
          }
        } else {
          allowed = n;
        }
        Slice tmp_result;

#ifndef ROCKSDB_LITE
        FileOperationInfo::TimePoint start_ts;
        if (ShouldNotifyListeners()) {
          start_ts = std::chrono::system_clock::now();
        }
#endif
        {
          IOSTATS_CPU_TIMER_GUARD(cpu_read_nanos, env_);
          s = file_->Read(offset + pos, allowed, &tmp_result, scratch + pos);
        }
#ifndef ROCKSDB_LITE
        if (ShouldNotifyListeners()) {
          auto finish_ts = std::chrono::system_clock::now();
          NotifyOnFileReadFinish(offset + pos, tmp_result.size(), start_ts,
                                 finish_ts, s);
        }
#endif

        if (res_scratch == nullptr) {
          // we can't simply use `scratch` because reads of mmap'd files return
          // data in a different buffer.
          res_scratch = tmp_result.data();
        } else {
          // make sure chunks are inserted contiguously into `res_scratch`.
          assert(tmp_result.data() == res_scratch + pos);
        }
        pos += tmp_result.size();
        if (!s.ok() || tmp_result.size() < allowed) {
          break;
        }
      }
      *result = Slice(res_scratch, s.ok() ? pos : 0);
    }
    IOSTATS_ADD_IF_POSITIVE(bytes_read, result->size());
    SetPerfLevel(prev_perf_level);
  }
  if (stats_ != nullptr && file_read_hist_ != nullptr) {
    file_read_hist_->Add(elapsed);
  }

  return s;
}

Status RandomAccessFileReader::MultiRead(ReadRequest* read_reqs,
                                         size_t num_reqs) const {
  Status s;
  uint64_t elapsed = 0;
  assert(!use_direct_io());
  {
    StopWatch sw(env_, stats_, hist_type_,
                 (stats_ != nullptr) ? &elapsed : nullptr, true /*overwrite*/,
                true /*delay_enabled*/);
    auto prev_perf_level = GetPerfLevel();
    IOSTATS_TIMER_GUARD(read_nanos);

#ifndef ROCKSDB_LITE
      FileOperationInfo::TimePoint start_ts;
      if (ShouldNotifyListeners()) {
        start_ts = std::chrono::system_clock::now();
      }
#endif // ROCKSDB_LITE
      {
        IOSTATS_CPU_TIMER_GUARD(cpu_read_nanos, env_);
        s = file_->MultiRead(read_reqs, num_reqs);
      }
      for (size_t i = 0; i < num_reqs; ++i) {
#ifndef ROCKSDB_LITE
        if (ShouldNotifyListeners()) {
          auto finish_ts = std::chrono::system_clock::now();
            NotifyOnFileReadFinish(read_reqs[i].offset,
                read_reqs[i].result.size(), start_ts, finish_ts,
                read_reqs[i].status);
        }
#endif // ROCKSDB_LITE
        IOSTATS_ADD_IF_POSITIVE(bytes_read, read_reqs[i].result.size());
      }
    SetPerfLevel(prev_perf_level);
  }
  if (stats_ != nullptr && file_read_hist_ != nullptr) {
    file_read_hist_->Add(elapsed);
  }

  return s;
}

Status WritableFileWriter::Append(const Slice& data) {
  const char* src = data.data();
  size_t left = data.size();
  Status s;
  pending_sync_ = true;

  TEST_KILL_RANDOM("WritableFileWriter::Append:0",
                   rocksdb_kill_odds * REDUCE_ODDS2);

  {
    IOSTATS_TIMER_GUARD(prepare_write_nanos);
    TEST_SYNC_POINT("WritableFileWriter::Append:BeforePrepareWrite");
    writable_file_->PrepareWrite(static_cast<size_t>(GetFileSize()), left);
  }

  // See whether we need to enlarge the buffer to avoid the flush
  size_t total = left + buf_.CurrentSize();
  if (buf_.Capacity() < total && max_buffer_size_ >= total) {
    for (size_t cap = buf_.Capacity();
         cap < max_buffer_size_;  // There is still room to increase
         cap *= 2) {
      // See whether the next available size is large enough.
      // Buffer will never be increased to more than max_buffer_size_.
      size_t desired_capacity = std::min(cap * 2, max_buffer_size_);
      if (desired_capacity - buf_.CurrentSize() >= left ||
          (use_direct_io() && desired_capacity == max_buffer_size_)) {
        buf_.AllocateNewBuffer(desired_capacity, true);
        break;
      }
    }
  }

  while (left > 0) {
    size_t appended = buf_.Append(src, left);
    filesize_ += appended;
    left -= appended;
    src += appended;

    if (left > 0) {
      s = Flush();
      if (!s.ok()) {
        return s;
      }
    }
    if (!use_direct_io() && left >= buf_.Capacity()) {
      assert(buf_.CurrentSize() == 0);
      s = WriteBuffered(src, left);
      if (s.ok()) {
        filesize_ += left;
        s = IncrementalSync();
      }
      break;
    }
  }

  TEST_KILL_RANDOM("WritableFileWriter::Append:1", rocksdb_kill_odds);
  return s;
}

Status WritableFileWriter::Pad(const size_t pad_bytes) {
  assert(pad_bytes < kDefaultPageSize);
  size_t left = pad_bytes;
  size_t cap = buf_.Capacity() - buf_.CurrentSize();

  // Assume pad_bytes is small compared to buf_ capacity. So we always
  // use buf_ rather than write directly to file in certain cases like
  // Append() does.
  while (left) {
    size_t append_bytes = std::min(cap, left);
    buf_.PadWith(append_bytes, 0);
    left -= append_bytes;
    if (left > 0) {
      Status s = Flush();
      if (!s.ok()) {
        return s;
      }
    }
    cap = buf_.Capacity() - buf_.CurrentSize();
  }
  pending_sync_ = true;
  filesize_ += pad_bytes;
  return Status::OK();
}

Status WritableFileWriter::Close() {

  // Do not quit immediately on failure the file MUST be closed
  Status s;

  // Possible to close it twice now as we MUST close
  // in __dtor, simply flushing is not enough
  // Windows when pre-allocating does not fill with zeros
  // also with unbuffered access we also set the end of data.
  if (!writable_file_) {
    return s;
  }

  s = Flush();  // flush cache to OS

  Status interim;
  // In direct I/O mode we write whole pages so
  // we need to let the file know where data ends.
  if (use_direct_io()) {
    interim = writable_file_->Truncate(filesize_);
    if (interim.ok()) {
      interim = writable_file_->Fsync();
    }
    if (!interim.ok() && s.ok()) {
      s = interim;
    }
  }

  TEST_KILL_RANDOM("WritableFileWriter::Close:0", rocksdb_kill_odds);
  interim = writable_file_->Close();
  if (!interim.ok() && s.ok()) {
    s = interim;
  }

  writable_file_.reset();
  TEST_KILL_RANDOM("WritableFileWriter::Close:1", rocksdb_kill_odds);

  return s;
}

// write out the cached data to the OS cache or storage if direct I/O
// enabled
Status WritableFileWriter::Flush() {
  Status s;
  TEST_KILL_RANDOM("WritableFileWriter::Flush:0",
                   rocksdb_kill_odds * REDUCE_ODDS2);

  if (buf_.CurrentSize() > 0) {
    if (use_direct_io()) {
#ifndef ROCKSDB_LITE
      if (pending_sync_) {
        s = WriteDirect();
      }
#endif  // !ROCKSDB_LITE
    } else {
      s = WriteBuffered(buf_.BufferStart(), buf_.CurrentSize());
    }
    if (!s.ok()) {
      return s;
    }
  }

  s = writable_file_->Flush();

  if (!s.ok()) {
    return s;
  }

  // sync OS cache to disk for every bytes_per_sync_
  // TODO: give log file and sst file different options (log
  // files could be potentially cached in OS for their whole
  // life time, thus we might not want to flush at all).

  // We try to avoid sync to the last 1MB of data. For two reasons:
  // (1) avoid rewrite the same page that is modified later.
  // (2) for older version of OS, write can block while writing out
  //     the page.
  // Xfs does neighbor page flushing outside of the specified ranges. We
  // need to make sure sync range is far from the write offset.
  if (!use_direct_io() && rate_limiter_ != nullptr) {
    const uint64_t kBytesNotSyncRange = 1024 * 1024;  // recent 1MB is not synced.
    const uint64_t kBytesAlignWhenSync = 4 * 1024;    // Align 4KB.
    if (filesize_ > kBytesNotSyncRange) {
      uint64_t offset_sync_to = filesize_ - kBytesNotSyncRange;
      offset_sync_to -= offset_sync_to % kBytesAlignWhenSync;
      assert(offset_sync_to >= last_sync_size_);
      if (offset_sync_to > 0) {
        uint64_t pos = last_sync_size_;
        uint64_t allowed = 0;
        while (offset_sync_to > pos) {
          allowed = rate_limiter_->RequestToken(
              offset_sync_to - pos, 0, writable_file_->GetIOPriority(), stats_,
              RateLimiter::OpType::kWrite);
          s = RangeSync(pos, allowed);
          if (s.ok()) {
            last_sync_size_ = pos;
            pos += allowed;
          } else {
            break;
          }
        }
      }
    }
  }

  return s;
}

Status WritableFileWriter::Sync(bool use_fsync) {
  Status s = Flush();
  if (!s.ok()) {
    return s;
  }
  TEST_KILL_RANDOM("WritableFileWriter::Sync:0", rocksdb_kill_odds);
  if (!use_direct_io() && pending_sync_) {
    s = SyncInternal(use_fsync);
    if (!s.ok()) {
      return s;
    }
  }
  TEST_KILL_RANDOM("WritableFileWriter::Sync:1", rocksdb_kill_odds);
  pending_sync_ = false;
  return Status::OK();
}

Status WritableFileWriter::SyncWithoutFlush(bool use_fsync) {
  if (!writable_file_->IsSyncThreadSafe()) {
    return Status::NotSupported(
      "Can't WritableFileWriter::SyncWithoutFlush() because "
      "WritableFile::IsSyncThreadSafe() is false");
  }
  TEST_SYNC_POINT("WritableFileWriter::SyncWithoutFlush:1");
  Status s = SyncInternal(use_fsync);
  TEST_SYNC_POINT("WritableFileWriter::SyncWithoutFlush:2");
  return s;
}

Status WritableFileWriter::SyncInternal(bool use_fsync) {
  Status s;
  IOSTATS_TIMER_GUARD(fsync_nanos);
  TEST_SYNC_POINT("WritableFileWriter::SyncInternal:0");
  auto prev_perf_level = GetPerfLevel();
  IOSTATS_CPU_TIMER_GUARD(cpu_write_nanos, env_);
  if (use_fsync) {
    s = writable_file_->Fsync();
  } else {
    s = writable_file_->Sync();
  }
  SetPerfLevel(prev_perf_level);
  return s;
}

Status WritableFileWriter::IncrementalSync() {
  // TODO: give log file and sst file different options (log
  // files could be potentially cached in OS for their whole
  // life time, thus we might not want to flush at all).

  // We try to avoid sync to the last 1MB of data. For two reasons:
  // (1) avoid rewrite the same page that is modified later.
  // (2) for older version of OS, write can block while writing out
  //     the page.
  // Xfs does neighbor page flushing outside of the specified ranges. We
  // need to make sure sync range is far from the write offset.
  const uint64_t kBytesNotSyncRange = 1024 * 1024;  // recent 1MB is not synced.
  const uint64_t kBytesAlignWhenSync = 4 * 1024;    // Align 4KB.
  Status s;
  if (bytes_per_sync_ && filesize_ > kBytesNotSyncRange) {
    uint64_t offset_sync_to = filesize_ - kBytesNotSyncRange;
    offset_sync_to -= offset_sync_to % kBytesAlignWhenSync;
    assert(offset_sync_to >= last_sync_size_);
    if (offset_sync_to > 0 &&
        offset_sync_to - last_sync_size_ >= bytes_per_sync_) {
      s = RangeSync(last_sync_size_, offset_sync_to - last_sync_size_);
      last_sync_size_ = offset_sync_to;
    }
  }
  return s;
}

Status WritableFileWriter::RangeSync(uint64_t offset, uint64_t nbytes) {
  IOSTATS_TIMER_GUARD(range_sync_nanos);
  TEST_SYNC_POINT("WritableFileWriter::RangeSync:0");
  return writable_file_->RangeSync(offset, nbytes);
}

// This method writes to disk the specified data and makes use of the rate
// limiter if available
Status WritableFileWriter::WriteBuffered(const char* data, size_t size) {
  Status s;
  assert(!use_direct_io());
  const char* src = data;
  size_t left = size;

  while (left > 0) {
    size_t allowed;
    // if (rate_limiter_ != nullptr) {
    //   allowed = rate_limiter_->RequestToken(
    //       left, 0 /* alignment */, writable_file_->GetIOPriority(), stats_,
    //       RateLimiter::OpType::kWrite);
    // } else {
    //   allowed = left;
    // }
    allowed = left;

    {
      IOSTATS_TIMER_GUARD(write_nanos);
      TEST_SYNC_POINT("WritableFileWriter::Flush:BeforeAppend");

#ifndef ROCKSDB_LITE
      FileOperationInfo::TimePoint start_ts;
      uint64_t old_size = writable_file_->GetFileSize();
      if (ShouldNotifyListeners()) {
        start_ts = std::chrono::system_clock::now();
        old_size = next_write_offset_;
      }
#endif
      {
        auto prev_perf_level = GetPerfLevel();
        IOSTATS_CPU_TIMER_GUARD(cpu_write_nanos, env_);
        s = writable_file_->Append(Slice(src, allowed));
        SetPerfLevel(prev_perf_level);
      }
#ifndef ROCKSDB_LITE
      if (ShouldNotifyListeners()) {
        auto finish_ts = std::chrono::system_clock::now();
        NotifyOnFileWriteFinish(old_size, allowed, start_ts, finish_ts, s);
      }
#endif
      if (!s.ok()) {
        return s;
      }
    }

    IOSTATS_ADD(bytes_written, allowed);
    TEST_KILL_RANDOM("WritableFileWriter::WriteBuffered:0", rocksdb_kill_odds);

    left -= allowed;
    src += allowed;
  }
  buf_.Size(0);
  return s;
}


// This flushes the accumulated data in the buffer. We pad data with zeros if
// necessary to the whole page.
// However, during automatic flushes padding would not be necessary.
// We always use RateLimiter if available. We move (Refit) any buffer bytes
// that are left over the
// whole number of pages to be written again on the next flush because we can
// only write on aligned
// offsets.
#ifndef ROCKSDB_LITE
Status WritableFileWriter::WriteDirect() {
  assert(use_direct_io());
  Status s;
  const size_t alignment = buf_.Alignment();
  assert((next_write_offset_ % alignment) == 0);

  // Calculate whole page final file advance if all writes succeed
  size_t file_advance =
    TruncateToPageBoundary(alignment, buf_.CurrentSize());

  // Calculate the leftover tail, we write it here padded with zeros BUT we
  // will write
  // it again in the future either on Close() OR when the current whole page
  // fills out
  size_t leftover_tail = buf_.CurrentSize() - file_advance;

  // Round up and pad
  buf_.PadToAlignmentWith(0);

  const char* src = buf_.BufferStart();
  uint64_t write_offset = next_write_offset_;
  size_t left = buf_.CurrentSize();

  while (left > 0) {
    // Check how much is allowed
    size_t size;
    if (rate_limiter_ != nullptr) {
      size = rate_limiter_->RequestToken(left, buf_.Alignment(),
                                         writable_file_->GetIOPriority(),
                                         stats_, RateLimiter::OpType::kWrite);
    } else {
      size = left;
    }

    {
      IOSTATS_TIMER_GUARD(write_nanos);
      TEST_SYNC_POINT("WritableFileWriter::Flush:BeforeAppend");
      FileOperationInfo::TimePoint start_ts;
      if (ShouldNotifyListeners()) {
        start_ts = std::chrono::system_clock::now();
      }
      // direct writes must be positional
      s = writable_file_->PositionedAppend(Slice(src, size), write_offset);
      if (ShouldNotifyListeners()) {
        auto finish_ts = std::chrono::system_clock::now();
        NotifyOnFileWriteFinish(write_offset, size, start_ts, finish_ts, s);
      }
      if (!s.ok()) {
        buf_.Size(file_advance + leftover_tail);
        return s;
      }
    }

    IOSTATS_ADD(bytes_written, size);
    left -= size;
    src += size;
    write_offset += size;
    assert((next_write_offset_ % alignment) == 0);
  }

  if (s.ok()) {
    // Move the tail to the beginning of the buffer
    // This never happens during normal Append but rather during
    // explicit call to Flush()/Sync() or Close()
    buf_.RefitTail(file_advance, leftover_tail);
    // This is where we start writing next time which may or not be
    // the actual file size on disk. They match if the buffer size
    // is a multiple of whole pages otherwise filesize_ is leftover_tail
    // behind
    next_write_offset_ += file_advance;
  }
  return s;
}
#endif  // !ROCKSDB_LITE

namespace {
class ReadaheadRandomAccessFile : public RandomAccessFile {
 public:
  ReadaheadRandomAccessFile(std::unique_ptr<RandomAccessFile>&& file,
                            size_t readahead_size)
      : file_(std::move(file)),
        alignment_(file_->GetRequiredBufferAlignment()),
        readahead_size_(Roundup(readahead_size, alignment_)),
        buffer_(),
        buffer_offset_(0) {
    buffer_.Alignment(alignment_);
    buffer_.AllocateNewBuffer(readahead_size_);
  }

 ReadaheadRandomAccessFile(const ReadaheadRandomAccessFile&) = delete;

 ReadaheadRandomAccessFile& operator=(const ReadaheadRandomAccessFile&) = delete;

 Status Read(uint64_t offset, size_t n, Slice* result,
             char* scratch) const override {
   // Read-ahead only make sense if we have some slack left after reading
   if (n + alignment_ >= readahead_size_) {
     return file_->Read(offset, n, result, scratch);
   }

   std::unique_lock<std::mutex> lk(lock_);

   size_t cached_len = 0;
   // Check if there is a cache hit, meaning that [offset, offset + n) is either
   // completely or partially in the buffer.
   // If it's completely cached, including end of file case when offset + n is
   // greater than EOF, then return.
   if (TryReadFromCache(offset, n, &cached_len, scratch) &&
       (cached_len == n || buffer_.CurrentSize() < readahead_size_)) {
     // We read exactly what we needed, or we hit end of file - return.
     *result = Slice(scratch, cached_len);
     return Status::OK();
   }
   size_t advanced_offset = static_cast<size_t>(offset + cached_len);
   // In the case of cache hit advanced_offset is already aligned, means that
   // chunk_offset equals to advanced_offset
   size_t chunk_offset = TruncateToPageBoundary(alignment_, advanced_offset);

   Status s = ReadIntoBuffer(chunk_offset, readahead_size_);
   if (s.ok()) {
     // The data we need is now in cache, so we can safely read it
     size_t remaining_len;
     TryReadFromCache(advanced_offset, n - cached_len, &remaining_len,
                      scratch + cached_len);
     *result = Slice(scratch, cached_len + remaining_len);
   }
   return s;
 }

 Status Prefetch(uint64_t offset, size_t n) override {
   if (n < readahead_size_) {
     // Don't allow smaller prefetches than the configured `readahead_size_`.
     // `Read()` assumes a smaller prefetch buffer indicates EOF was reached.
     return Status::OK();
   }

   std::unique_lock<std::mutex> lk(lock_);

   size_t offset_ = static_cast<size_t>(offset);
   size_t prefetch_offset = TruncateToPageBoundary(alignment_, offset_);
   if (prefetch_offset == buffer_offset_) {
     return Status::OK();
   }
   return ReadIntoBuffer(prefetch_offset,
                         Roundup(offset_ + n, alignment_) - prefetch_offset);
 }

 size_t GetUniqueId(char* id, size_t max_size) const override {
   return file_->GetUniqueId(id, max_size);
 }

 void Hint(AccessPattern pattern) override { file_->Hint(pattern); }

 Status InvalidateCache(size_t offset, size_t length) override {
   std::unique_lock<std::mutex> lk(lock_);
   buffer_.Clear();
   return file_->InvalidateCache(offset, length);
 }

 bool use_direct_io() const override { return file_->use_direct_io(); }

private:
 // Tries to read from buffer_ n bytes starting at offset. If anything was read
 // from the cache, it sets cached_len to the number of bytes actually read,
 // copies these number of bytes to scratch and returns true.
 // If nothing was read sets cached_len to 0 and returns false.
 bool TryReadFromCache(uint64_t offset, size_t n, size_t* cached_len,
                       char* scratch) const {
   if (offset < buffer_offset_ ||
       offset >= buffer_offset_ + buffer_.CurrentSize()) {
     *cached_len = 0;
     return false;
   }
   uint64_t offset_in_buffer = offset - buffer_offset_;
   *cached_len = std::min(
       buffer_.CurrentSize() - static_cast<size_t>(offset_in_buffer), n);
   memcpy(scratch, buffer_.BufferStart() + offset_in_buffer, *cached_len);
   return true;
  }

  // Reads into buffer_ the next n bytes from file_ starting at offset.
  // Can actually read less if EOF was reached.
  // Returns the status of the read operastion on the file.
  Status ReadIntoBuffer(uint64_t offset, size_t n) const {
    if (n > buffer_.Capacity()) {
      n = buffer_.Capacity();
    }
    assert(IsFileSectorAligned(offset, alignment_));
    assert(IsFileSectorAligned(n, alignment_));
    Slice result;
    Status s = file_->Read(offset, n, &result, buffer_.BufferStart());
    if (s.ok()) {
      buffer_offset_ = offset;
      buffer_.Size(result.size());
      assert(result.size() == 0 || buffer_.BufferStart() == result.data());
    }
    return s;
  }

  const std::unique_ptr<RandomAccessFile> file_;
  const size_t alignment_;
  const size_t readahead_size_;

  mutable std::mutex lock_;
  // The buffer storing the prefetched data
  mutable AlignedBuffer buffer_;
  // The offset in file_, corresponding to data stored in buffer_
  mutable uint64_t buffer_offset_;
};

// This class wraps a SequentialFile, exposing same API, with the differenece
// of being able to prefetch up to readahead_size bytes and then serve them
// from memory, avoiding the entire round-trip if, for example, the data for the
// file is actually remote.
class ReadaheadSequentialFile : public SequentialFile {
 public:
  ReadaheadSequentialFile(std::unique_ptr<SequentialFile>&& file,
                          size_t readahead_size)
      : file_(std::move(file)),
        alignment_(file_->GetRequiredBufferAlignment()),
        readahead_size_(Roundup(readahead_size, alignment_)),
        buffer_(),
        buffer_offset_(0),
        read_offset_(0) {
    buffer_.Alignment(alignment_);
    buffer_.AllocateNewBuffer(readahead_size_);
  }

  ReadaheadSequentialFile(const ReadaheadSequentialFile&) = delete;

  ReadaheadSequentialFile& operator=(const ReadaheadSequentialFile&) = delete;

  Status Read(size_t n, Slice* result, char* scratch) override {
    std::unique_lock<std::mutex> lk(lock_);

    size_t cached_len = 0;
    // Check if there is a cache hit, meaning that [offset, offset + n) is
    // either completely or partially in the buffer. If it's completely cached,
    // including end of file case when offset + n is greater than EOF, then
    // return.
    if (TryReadFromCache(n, &cached_len, scratch) &&
        (cached_len == n || buffer_.CurrentSize() < readahead_size_)) {
      // We read exactly what we needed, or we hit end of file - return.
      *result = Slice(scratch, cached_len);
      return Status::OK();
    }
    n -= cached_len;

    Status s;
    // Read-ahead only make sense if we have some slack left after reading
    if (n + alignment_ >= readahead_size_) {
      s = file_->Read(n, result, scratch + cached_len);
      if (s.ok()) {
        read_offset_ += result->size();
        *result = Slice(scratch, cached_len + result->size());
      }
      buffer_.Clear();
      return s;
    }

    s = ReadIntoBuffer(readahead_size_);
    if (s.ok()) {
      // The data we need is now in cache, so we can safely read it
      size_t remaining_len;
      TryReadFromCache(n, &remaining_len, scratch + cached_len);
      *result = Slice(scratch, cached_len + remaining_len);
    }
    return s;
  }

  Status Skip(uint64_t n) override {
    std::unique_lock<std::mutex> lk(lock_);
    Status s = Status::OK();
    // First check if we need to skip already cached data
    if (buffer_.CurrentSize() > 0) {
      // Do we need to skip beyond cached data?
      if (read_offset_ + n >= buffer_offset_ + buffer_.CurrentSize()) {
        // Yes. Skip whaterver is in memory and adjust offset accordingly
        n -= buffer_offset_ + buffer_.CurrentSize() - read_offset_;
        read_offset_ = buffer_offset_ + buffer_.CurrentSize();
      } else {
        // No. The entire section to be skipped is entirely i cache.
        read_offset_ += n;
        n = 0;
      }
    }
    if (n > 0) {
      // We still need to skip more, so call the file API for skipping
      s = file_->Skip(n);
      if (s.ok()) {
        read_offset_ += n;
      }
      buffer_.Clear();
    }
    return s;
  }

  Status PositionedRead(uint64_t offset, size_t n, Slice* result,
                        char* scratch) override {
    return file_->PositionedRead(offset, n, result, scratch);
  }

  Status InvalidateCache(size_t offset, size_t length) override {
    std::unique_lock<std::mutex> lk(lock_);
    buffer_.Clear();
    return file_->InvalidateCache(offset, length);
  }

  bool use_direct_io() const override { return file_->use_direct_io(); }

 private:
  // Tries to read from buffer_ n bytes. If anything was read from the cache, it
  // sets cached_len to the number of bytes actually read, copies these number
  // of bytes to scratch and returns true.
  // If nothing was read sets cached_len to 0 and returns false.
  bool TryReadFromCache(size_t n, size_t* cached_len, char* scratch) {
    if (read_offset_ < buffer_offset_ ||
        read_offset_ >= buffer_offset_ + buffer_.CurrentSize()) {
      *cached_len = 0;
      return false;
    }
    uint64_t offset_in_buffer = read_offset_ - buffer_offset_;
    *cached_len = std::min(
        buffer_.CurrentSize() - static_cast<size_t>(offset_in_buffer), n);
    memcpy(scratch, buffer_.BufferStart() + offset_in_buffer, *cached_len);
    read_offset_ += *cached_len;
    return true;
  }

  // Reads into buffer_ the next n bytes from file_.
  // Can actually read less if EOF was reached.
  // Returns the status of the read operastion on the file.
  Status ReadIntoBuffer(size_t n) {
    if (n > buffer_.Capacity()) {
      n = buffer_.Capacity();
    }
    assert(IsFileSectorAligned(n, alignment_));
    Slice result;
    Status s = file_->Read(n, &result, buffer_.BufferStart());
    if (s.ok()) {
      buffer_offset_ = read_offset_;
      buffer_.Size(result.size());
      assert(result.size() == 0 || buffer_.BufferStart() == result.data());
    }
    return s;
  }

  const std::unique_ptr<SequentialFile> file_;
  const size_t alignment_;
  const size_t readahead_size_;

  std::mutex lock_;
  // The buffer storing the prefetched data
  AlignedBuffer buffer_;
  // The offset in file_, corresponding to data stored in buffer_
  uint64_t buffer_offset_;
  // The offset up to which data was read from file_. In fact, it can be larger
  // than the actual file size, since the file_->Skip(n) call doesn't return the
  // actual number of bytes that were skipped, which can be less than n.
  // This is not a problemm since read_offset_ is monotonically increasing and
  // its only use is to figure out if next piece of data should be read from
  // buffer_ or file_ directly.
  uint64_t read_offset_;
};
}  // namespace

Status FilePrefetchBuffer::Prefetch(RandomAccessFileReader* reader,
                                    uint64_t offset, size_t n,
                                    bool for_compaction) {
  size_t alignment = reader->file()->GetRequiredBufferAlignment();
  size_t offset_ = static_cast<size_t>(offset);
  uint64_t rounddown_offset = Rounddown(offset_, alignment);
  uint64_t roundup_end = Roundup(offset_ + n, alignment);
  uint64_t roundup_len = roundup_end - rounddown_offset;
  assert(roundup_len >= alignment);
  assert(roundup_len % alignment == 0);

  // Check if requested bytes are in the existing buffer_.
  // If all bytes exist -- return.
  // If only a few bytes exist -- reuse them & read only what is really needed.
  //     This is typically the case of incremental reading of data.
  // If no bytes exist in buffer -- full pread.

  Status s;
  uint64_t chunk_offset_in_buffer = 0;
  uint64_t chunk_len = 0;
  bool copy_data_to_new_buffer = false;
  if (buffer_.CurrentSize() > 0 && offset >= buffer_offset_ &&
      offset <= buffer_offset_ + buffer_.CurrentSize()) {
    if (offset + n <= buffer_offset_ + buffer_.CurrentSize()) {
      // All requested bytes are already in the buffer. So no need to Read
      // again.
      return s;
    } else {
      // Only a few requested bytes are in the buffer. memmove those chunk of
      // bytes to the beginning, and memcpy them back into the new buffer if a
      // new buffer is created.
      chunk_offset_in_buffer = Rounddown(static_cast<size_t>(offset - buffer_offset_), alignment);
      chunk_len = buffer_.CurrentSize() - chunk_offset_in_buffer;
      assert(chunk_offset_in_buffer % alignment == 0);
      assert(chunk_len % alignment == 0);
      assert(chunk_offset_in_buffer + chunk_len <=
             buffer_offset_ + buffer_.CurrentSize());
      if (chunk_len > 0) {
        copy_data_to_new_buffer = true;
      } else {
        // this reset is not necessary, but just to be safe.
        chunk_offset_in_buffer = 0;
      }
    }
  }

  // Create a new buffer only if current capacity is not sufficient, and memcopy
  // bytes from old buffer if needed (i.e., if chunk_len is greater than 0).
  if (buffer_.Capacity() < roundup_len) {
    buffer_.Alignment(alignment);
    buffer_.AllocateNewBuffer(static_cast<size_t>(roundup_len),
                              copy_data_to_new_buffer, chunk_offset_in_buffer,
                              static_cast<size_t>(chunk_len));
  } else if (chunk_len > 0) {
    // New buffer not needed. But memmove bytes from tail to the beginning since
    // chunk_len is greater than 0.
    buffer_.RefitTail(static_cast<size_t>(chunk_offset_in_buffer), static_cast<size_t>(chunk_len));
  }

  Slice result;
  s = reader->Read(rounddown_offset + chunk_len,
                   static_cast<size_t>(roundup_len - chunk_len), &result,
                   buffer_.BufferStart() + chunk_len, for_compaction);
  if (s.ok()) {
    buffer_offset_ = rounddown_offset;
    buffer_.Size(static_cast<size_t>(chunk_len) + result.size());
  }
  return s;
}

bool FilePrefetchBuffer::TryReadFromCache(uint64_t offset, size_t n,
                                          Slice* result, bool for_compaction) {
  if (track_min_offset_ && offset < min_offset_read_) {
    min_offset_read_ = static_cast<size_t>(offset);
  }
  if (!enable_ || offset < buffer_offset_) {
    return false;
  }

  // If the buffer contains only a few of the requested bytes:
  //    If readahead is enabled: prefetch the remaining bytes + readadhead bytes
  //        and satisfy the request.
  //    If readahead is not enabled: return false.
  if (offset + n > buffer_offset_ + buffer_.CurrentSize()) {
    if (readahead_size_ > 0) {
      assert(file_reader_ != nullptr);
      assert(max_readahead_size_ >= readahead_size_);
      Status s;
      if (for_compaction) {
        s = Prefetch(file_reader_, offset, std::max(n, readahead_size_), for_compaction);
      } else {
        s = Prefetch(file_reader_, offset, n + readahead_size_, for_compaction);
      }
      if (!s.ok()) {
        return false;
      }
      readahead_size_ = std::min(max_readahead_size_, readahead_size_ * 2);
    } else {
      return false;
    }
  }

  uint64_t offset_in_buffer = offset - buffer_offset_;
  *result = Slice(buffer_.BufferStart() + offset_in_buffer, n);
  return true;
}

std::unique_ptr<RandomAccessFile> NewReadaheadRandomAccessFile(
    std::unique_ptr<RandomAccessFile>&& file, size_t readahead_size) {
  std::unique_ptr<RandomAccessFile> result(
    new ReadaheadRandomAccessFile(std::move(file), readahead_size));
  return result;
}

std::unique_ptr<SequentialFile>
SequentialFileReader::NewReadaheadSequentialFile(
    std::unique_ptr<SequentialFile>&& file, size_t readahead_size) {
  if (file->GetRequiredBufferAlignment() >= readahead_size) {
    // Short-circuit and return the original file if readahead_size is
    // too small and hence doesn't make sense to be used for prefetching.
    return std::move(file);
  }
  std::unique_ptr<SequentialFile> result(
      new ReadaheadSequentialFile(std::move(file), readahead_size));
  return result;
}

Status NewWritableFile(Env* env, const std::string& fname,
                       std::unique_ptr<WritableFile>* result,
                       const EnvOptions& options) {
  Status s = env->NewWritableFile(fname, result, options);
  TEST_KILL_RANDOM("NewWritableFile:0", rocksdb_kill_odds * REDUCE_ODDS2);
  return s;
}

bool ReadOneLine(std::istringstream* iss, SequentialFile* seq_file,
                 std::string* output, bool* has_data, Status* result) {
  const int kBufferSize = 8192;
  char buffer[kBufferSize + 1];
  Slice input_slice;

  std::string line;
  bool has_complete_line = false;
  while (!has_complete_line) {
    if (std::getline(*iss, line)) {
      has_complete_line = !iss->eof();
    } else {
      has_complete_line = false;
    }
    if (!has_complete_line) {
      // if we're not sure whether we have a complete line,
      // further read from the file.
      if (*has_data) {
        *result = seq_file->Read(kBufferSize, &input_slice, buffer);
      }
      if (input_slice.size() == 0) {
        // meaning we have read all the data
        *has_data = false;
        break;
      } else {
        iss->str(line + input_slice.ToString());
        // reset the internal state of iss so that we can keep reading it.
        iss->clear();
        *has_data = (input_slice.size() == kBufferSize);
        continue;
      }
    }
  }
  *output = line;
  return *has_data || has_complete_line;
}

}  // namespace rocksdb
