#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

// Single-worker thread pool for serialized async operations.
// Used for H2D and D2H transfer queues.
class ThreadPool {
public:
  ThreadPool();
  ~ThreadPool();
  template <class F> void run(F &&f);
  void synchronize();

private:
  bool stop;
  std::mutex mutex;
  std::thread worker;
  std::condition_variable condition;
  std::queue<std::future<void>> results;
  std::queue<std::function<void()>> tasks;
};

inline ThreadPool::ThreadPool() : stop(false) {
  worker = std::thread([this] {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->condition.wait(
            lock, [this] { return this->stop || !this->tasks.empty(); });
        if (this->stop && this->tasks.empty())
          return;
        task = std::move(this->tasks.front());
        this->tasks.pop();
      }
      task();
    }
  });
}

inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutex);
    stop = true;
  }
  condition.notify_all();
  worker.join();
}

template <class F> void ThreadPool::run(F &&f) {
  auto task = std::make_shared<std::packaged_task<void()>>(
      std::bind(std::forward<F>(f)));
  results.emplace(task->get_future());
  {
    std::unique_lock<std::mutex> lock(mutex);
    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
}

inline void ThreadPool::synchronize() {
  if (results.empty())
    return;
  results.front().get();
  results.pop();
}
