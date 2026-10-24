# OpenNPU backend for ggml: matrix products on the AMD XDNA2 NPU (XRT 2.25 C++ API).
set(OPENNPU_XRT_INCLUDE "" CACHE PATH "XRT include dir (with xrt/xrt_kernel.h)")
set(OPENNPU_XRT_LIB "" CACHE PATH "dir with libxrt_coreutil.so")

ggml_add_backend_library(ggml-opennpu ggml-opennpu.cpp)
target_include_directories(ggml-opennpu PRIVATE ${OPENNPU_XRT_INCLUDE})
find_library(XRT_COREUTIL xrt_coreutil HINTS ${OPENNPU_XRT_LIB} REQUIRED)
target_link_libraries(ggml-opennpu PRIVATE ${XRT_COREUTIL})
