#include "io_uring_engine.h"

#include <atomic>
#include <fcntl.h>
#include <mutex>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <liburing.h>

class IoUringEngineImpl {
public:
  IoUringEngineImpl(int queue_depth) : next_handle_(0) {
    int ret = io_uring_queue_init(std::max(queue_depth, 2), &ring_, 0);
    if (ret < 0) {
      throw std::runtime_error("io_uring_queue_init failed: " +
                               std::to_string(-ret));
    }
  }

  ~IoUringEngineImpl() {
    wait_all();
    io_uring_queue_exit(&ring_);
  }

  bool has_io_uring() const { return true; }

  int64_t submit_read(const std::string &path, void *buf, int64_t file_offset,
                      int64_t nbytes) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("Cannot open file: " + path);

    int64_t handle = next_handle_++;

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      // SQ full — drain completions first, then retry
      drain_completions();
      sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        close(fd);
        throw std::runtime_error("io_uring SQ full after drain");
      }
    }

    io_uring_prep_read(sqe, fd, buf, nbytes, file_offset);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      fd_map_[handle] = fd;
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      close(fd);
      fd_map_.erase(handle);
      throw std::runtime_error("io_uring_submit failed: " +
                               std::to_string(-ret));
    }
    return handle;
  }

  int64_t submit_write(const std::string &path, const void *buf,
                       int64_t file_offset, int64_t nbytes) {
    int fd =
        open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0)
      throw std::runtime_error("Cannot open file for writing: " + path);

    int64_t handle = next_handle_++;

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      drain_completions();
      sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        close(fd);
        throw std::runtime_error("io_uring SQ full after drain");
      }
    }

    io_uring_prep_write(sqe, fd, buf, nbytes, file_offset);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      fd_map_[handle] = fd;
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      close(fd);
      fd_map_.erase(handle);
      throw std::runtime_error("io_uring_submit failed: " +
                               std::to_string(-ret));
    }
    return handle;
  }

  void wait(int64_t handle) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (completed_.count(handle)) {
        close_and_remove(handle);
        return;
      }
    }

    // Reap completions until we find our handle
    while (true) {
      struct io_uring_cqe *cqe;
      int ret = io_uring_wait_cqe(&ring_, &cqe);
      if (ret < 0)
        throw std::runtime_error("io_uring_wait_cqe failed: " +
                                 std::to_string(-ret));

      int64_t h = reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));
      io_uring_cqe_seen(&ring_, cqe);

      std::lock_guard<std::mutex> lock(mutex_);
      if (h == handle) {
        close_and_remove(h);
        return;
      }
      completed_.insert(h);
    }
  }

  void wait_all() {
    while (true) {
      std::vector<int64_t> pending;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &kv : fd_map_) {
          if (!completed_.count(kv.first))
            pending.push_back(kv.first);
        }
      }
      if (pending.empty())
        break;
      for (int64_t h : pending)
        wait(h);
    }
    // Close any remaining completed FDs
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &kv : fd_map_)
      close(kv.second);
    fd_map_.clear();
    completed_.clear();
  }

  bool poll(int64_t handle) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (completed_.count(handle))
        return true;
    }

    struct io_uring_cqe *cqe;
    int ret = io_uring_peek_cqe(&ring_, &cqe);
    if (ret == 0) {
      int64_t h = reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));
      io_uring_cqe_seen(&ring_, cqe);
      std::lock_guard<std::mutex> lock(mutex_);
      completed_.insert(h);
      return h == handle;
    }
    return false;
  }

  int pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (auto &kv : fd_map_)
      if (!completed_.count(kv.first))
        count++;
    return count;
  }

private:
  void drain_completions() {
    // Non-blocking drain of all available completions
    struct io_uring_cqe *cqe;
    while (io_uring_peek_cqe(&ring_, &cqe) == 0) {
      int64_t h = reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));
      io_uring_cqe_seen(&ring_, cqe);
      std::lock_guard<std::mutex> lock(mutex_);
      completed_.insert(h);
    }
  }

  void close_and_remove(int64_t handle) {
    // Must be called with mutex_ held
    auto it = fd_map_.find(handle);
    if (it != fd_map_.end()) {
      close(it->second);
      fd_map_.erase(it);
    }
    completed_.erase(handle);
  }

  struct io_uring ring_;
  std::atomic<int64_t> next_handle_;
  mutable std::mutex mutex_;
  std::unordered_map<int64_t, int> fd_map_;
  std::unordered_set<int64_t> completed_;
};

// ============================================================
// Public API
// ============================================================

IoUringEngine::IoUringEngine(int queue_depth)
    : impl(new IoUringEngineImpl(queue_depth)) {}

IoUringEngine::~IoUringEngine() { delete impl; }

bool IoUringEngine::has_io_uring() const { return impl->has_io_uring(); }

int64_t IoUringEngine::submit_read(const std::string &path, torch::Tensor dst,
                                   int64_t file_offset, int64_t nbytes) {
  AT_ASSERTM(!dst.is_cuda(), "submit_read: tensor must be on CPU");
  AT_ASSERTM(dst.is_contiguous(), "submit_read: tensor must be contiguous");
  return impl->submit_read(path, dst.data_ptr(), file_offset, nbytes);
}

int64_t IoUringEngine::submit_write(const std::string &path, torch::Tensor src,
                                    int64_t file_offset, int64_t nbytes) {
  AT_ASSERTM(!src.is_cuda(), "submit_write: tensor must be on CPU");
  AT_ASSERTM(src.is_contiguous(), "submit_write: tensor must be contiguous");
  return impl->submit_write(path, src.data_ptr(), file_offset, nbytes);
}

void IoUringEngine::wait(int64_t handle) { impl->wait(handle); }
void IoUringEngine::wait_all() { impl->wait_all(); }
bool IoUringEngine::poll(int64_t handle) { return impl->poll(handle); }
int IoUringEngine::pending() const { return impl->pending(); }
