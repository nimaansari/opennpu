#!/usr/bin/env bash
# Build llama.cpp with OpenNPU's NPU backend (ggml-opennpu): llama.cpp's matrix products run on
# the AMD Ryzen AI NPU, everything else on the CPU.
#
#   llama/build.sh [DEST]        default DEST: $OPENNPU_HOME/llama.cpp (~/.local/share/opennpu/llama.cpp)
#
# Needs: git, cmake, a C++17 compiler, the XRT 2.25 runtime (libxrt_coreutil.so.2). XRT's C++
# headers are fetched from Xilinx/XRT (tag 2.25.0) because distros ship only the runtime.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOME_DIR=${OPENNPU_HOME:-$HOME/.local/share/opennpu}
DEST=${1:-$HOME_DIR/llama.cpp}
LLAMA_REV=${LLAMA_REV:-4413a19}
XRT_TAG=2.25.0
SRC=$DEST/src
XRT=$DEST/xrt
mkdir -p "$DEST"

if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch -q origin "$LLAMA_REV" 2>/dev/null || true
git -C "$SRC" checkout -q "$LLAMA_REV"

if [ ! -f "$XRT/include/xrt/xrt_kernel.h" ]; then
  rm -rf "$XRT/src"
  git clone -q --depth 1 --branch "$XRT_TAG" --filter=blob:none --sparse https://github.com/Xilinx/XRT "$XRT/src"
  git -C "$XRT/src" sparse-checkout set src/runtime_src/core/include
  mkdir -p "$XRT/include" && cp -r "$XRT/src/src/runtime_src/core/include/xrt" "$XRT/include/"
  cat > "$XRT/include/xrt/detail/version-slim.h" <<'EOF'
/* XRT 2.25 version header (generated at XRT build time; written by OpenNPU's llama/build.sh) */
#ifndef XRT_VERSION_SLIM_H
#define XRT_VERSION_SLIM_H
#define XRT_VERSION(major, minor) ((major << 16) + (minor))
#define XRT_VERSION_CODE XRT_VERSION(2, 25)
#define XRT_MAJOR(code) ((code >> 16))
#define XRT_MINOR(code) (code - ((code >> 16) << 16))
#endif
EOF
fi
LIBDIR=$(dirname "$(ldconfig -p | awk '/libxrt_coreutil.so.2 /{print $NF; exit}')")
mkdir -p "$XRT/lib" && ln -sfn "$LIBDIR/libxrt_coreutil.so.2" "$XRT/lib/libxrt_coreutil.so"

# hook the backend into the tree (idempotent)
# (copies, not symlinks: CMake resolves symlinked source dirs, which breaks the ../ includes)
rm -rf "$SRC/ggml/src/ggml-opennpu" && cp -r "$ROOT/llama/ggml-opennpu" "$SRC/ggml/src/ggml-opennpu"
rm -f "$SRC/ggml/include/ggml-opennpu.h" && cp "$ROOT/llama/include/ggml-opennpu.h" "$SRC/ggml/include/ggml-opennpu.h"
grep -q "GGML_OPENNPU" "$SRC/ggml/CMakeLists.txt" ||
  sed -i 's/^option(GGML_BLAS  .*/&\noption(GGML_OPENNPU "ggml: use the OpenNPU (AMD XDNA2 NPU) backend" OFF)/' "$SRC/ggml/CMakeLists.txt"
grep -q "ggml_add_backend(OPENNPU)" "$SRC/ggml/src/CMakeLists.txt" ||
  sed -i 's/^ggml_add_backend(BLAS)$/&\nggml_add_backend(OPENNPU)/' "$SRC/ggml/src/CMakeLists.txt"
python3 - "$SRC/ggml/src/ggml-backend-reg.cpp" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
if "ggml-opennpu.h" not in s:
    s = s.replace('#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif\n',
                  '#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif\n#ifdef GGML_USE_OPENNPU\n#include "ggml-opennpu.h"\n#endif\n', 1)
    s = s.replace('        register_backend(ggml_backend_blas_reg());\n#endif\n',
                  '        register_backend(ggml_backend_blas_reg());\n#endif\n#ifdef GGML_USE_OPENNPU\n'
                  '        register_backend(ggml_backend_opennpu_reg());\n#endif\n', 1)
    open(p, "w").write(s)
PYEOF
grep -q "GGML_OPENNPU" "$SRC/ggml/CMakeLists.txt" && grep -q "ggml_add_backend(OPENNPU)" "$SRC/ggml/src/CMakeLists.txt" &&
  grep -q "ggml_backend_opennpu_reg()" "$SRC/ggml/src/ggml-backend-reg.cpp" || { echo "patching llama.cpp failed" >&2; exit 1; }

cmake -S "$SRC" -B "$DEST/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENNPU=ON -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DOPENNPU_XRT_INCLUDE="$XRT/include" -DOPENNPU_XRT_LIB="$XRT/lib" >/dev/null
cmake --build "$DEST/build" --target llama-server llama-cli llama-bench llama-perplexity -j "$(nproc)"
echo "built: $DEST/build/bin/llama-server (OpenNPU backend)"

# libonpu: the same NPU core as a C library for OpenNPU's 4-bit engine (opennpu/native/npu4)
mkdir -p "$HOME_DIR/lib"
g++ -std=c++17 -O3 -fopenmp-simd -fPIC -shared -pthread -march=native -I"$XRT/include" -I"$ROOT/llama/ggml-opennpu" \
  "$ROOT/llama/ggml-opennpu/onpu_api.cpp" "$ROOT/llama/ggml-opennpu/onpu_cpu.cpp" -o "$HOME_DIR/lib/libonpu.so" -L"$XRT/lib" -lxrt_coreutil \
  -Wl,-rpath,"$(dirname "$(readlink -f "$XRT/lib/libxrt_coreutil.so")")"
echo "built: $HOME_DIR/lib/libonpu.so"
