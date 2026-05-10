#pragma once

#include <torch/extension.h>
#include <string>

// Forward declare
class IoUringEngineImpl;

// io_uring async I/O engine for host <-> NVMe storage transfers.
// Uses bundled liburing 2.8 from third_party/.
class IoUringEngine {
public:
  explicit IoUringEngine(int queue_depth = 64);
  ~IoUringEngine();

  // Submit async read: file -> CPU tensor
  // Returns handle for wait()/poll()
  int64_t submit_read(const std::string &path, torch::Tensor dst,
                      int64_t file_offset, int64_t nbytes);

  // Submit async write: CPU tensor -> file
  int64_t submit_write(const std::string &path, torch::Tensor src,
                       int64_t file_offset, int64_t nbytes);

  // Wait for specific operation to complete
  void wait(int64_t handle);

  // Wait for all pending operations
  void wait_all();

  // Check if operation completed (non-blocking)
  bool poll(int64_t handle);

  // Number of in-flight operations
  int pending() const;

  // Whether io_uring is available (vs POSIX fallback)
  bool has_io_uring() const;

private:
  IoUringEngineImpl *impl;
};
