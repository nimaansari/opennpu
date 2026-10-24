#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// OpenNPU: matrix products (MUL_MAT, MUL_MAT_ID) on the AMD Ryzen AI (XDNA2) NPU through
// OpenNPU's int4 GEMV device; everything else stays on the CPU backend.
GGML_BACKEND_API ggml_backend_t ggml_backend_opennpu_init(void);
GGML_BACKEND_API bool ggml_backend_is_opennpu(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opennpu_reg(void);

#ifdef  __cplusplus
}
#endif
