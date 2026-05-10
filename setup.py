"""Build script for GriNNder with C++/CUDA extensions."""

import os
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

# Check for liburing
def has_liburing():
    """Check if liburing is available for io_uring support."""
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "liburing"],
            capture_output=True,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    # Check common paths
    for lib_dir in ["/usr/lib", "/usr/lib64", "/usr/local/lib"]:
        if os.path.exists(os.path.join(lib_dir, "liburing.so")):
            return True
        if os.path.exists(os.path.join(lib_dir, "liburing.a")):
            return True
    return False


def get_extensions():
    """Build C++/CUDA extensions."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    except ImportError:
        print("PyTorch not found. Skipping C++ extensions.")
        return [], {}

    sources = [
        "csrc/grinnder_ops.cpp",
        "csrc/cpu/subgraph.cpp",
        "csrc/cpu/io_uring_engine.cpp",
        "csrc/cuda/async_transfer.cu",
    ]

    extra_cxx_flags = ["-O3", "-std=c++17"]
    extra_nvcc_flags = ["-O3"]
    extra_link_flags = []
    define_macros = []

    # Use bundled liburing from third_party/ (built from source, version 2.8)
    liburing_dir = os.path.join(os.path.dirname(__file__), "third_party", "liburing")
    liburing_include = os.path.join(liburing_dir, "src", "include")
    liburing_lib = os.path.join(liburing_dir, "src")

    if os.path.exists(os.path.join(liburing_lib, "liburing.a")):
        print(f"Using bundled liburing from {liburing_dir}")
        extra_cxx_flags.append(f"-I{liburing_include}")
        # Static link against bundled liburing for portability
        extra_link_flags.append(os.path.join(liburing_lib, "liburing.a"))
    elif has_liburing():
        print("Using system liburing")
        extra_link_flags.append("-luring")
    else:
        raise RuntimeError(
            "liburing not found. Run: cd third_party/liburing && ./configure && make"
        )

    # Let PyTorch's extension builder expand the full arch list, including
    # multi-arch values such as "8.0;8.6;9.0" or entries with "+PTX".
    cuda_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    if cuda_arch_list:
        print(f"Using TORCH_CUDA_ARCH_LIST={cuda_arch_list}")
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        print("TORCH_CUDA_ARCH_LIST not set; defaulting to 8.0")

    ext = CUDAExtension(
        name="grinnder._C",
        sources=sources,
        define_macros=define_macros,
        extra_compile_args={
            "cxx": extra_cxx_flags,
            "nvcc": extra_nvcc_flags,
        },
        extra_link_args=extra_link_flags,
    )

    return [ext], {"build_ext": BuildExtension.with_options(no_python_abi_tag=True)}


ext_modules, cmdclass = get_extensions()

setup(
    name="grinnder",
    version="0.1.0",
    author="Accelerated Intelligent Systems Lab (AISys), Seoul National University",
    description="GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading",
    long_description=Path("README.md").read_text() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/AIS-SNU/GriNNder",
    project_urls={
        "Paper": "https://openreview.net/forum?id=8SNPzGRldN",
    },
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.21.0",
        "ogb>=1.3.6",
        "grdpart",
    ],
    extras_require={
        "gds": ["kvikio-cu12"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
