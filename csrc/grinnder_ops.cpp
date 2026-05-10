#include <torch/extension.h>

#include "cpu/subgraph.h"
#include "cpu/io_uring_engine.h"
#include "cuda/async_transfer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Async GPU <-> Host transfers
  m.def("h2d_copy_async", &h2d_copy_async, "Async CPU-host to GPU copy");
  m.def("d2h_copy_async", &d2h_copy_async, "Async GPU to CPU-host copy");
  m.def("h2d_synchronize", &h2d_synchronize, "Wait for H2D thread pool");
  m.def("d2h_synchronize", &d2h_synchronize, "Wait for D2H thread pool");

  // Partition gather/scatter
  m.def("gather_partitions", &gather_partitions,
        "Gather features from host partitions to GPU tensor");
  m.def("scatter_partitions", &scatter_partitions,
        "Scatter GPU gradient to host partitions with accumulation");

  // Subgraph extraction
  m.def("build_subgraph", &build_subgraph,
        "Extract 1-hop subgraph with contiguous relabeling");

  // io_uring engine for host <-> storage
  py::class_<IoUringEngine>(m, "IoUringEngine")
      .def(py::init<int>(), py::arg("queue_depth") = 64)
      .def("submit_read", &IoUringEngine::submit_read)
      .def("submit_write", &IoUringEngine::submit_write)
      .def("wait", &IoUringEngine::wait)
      .def("wait_all", &IoUringEngine::wait_all)
      .def("poll", &IoUringEngine::poll)
      .def("pending", &IoUringEngine::pending)
      .def("has_io_uring", &IoUringEngine::has_io_uring);
}
