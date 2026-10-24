#!/usr/bin/env bash
# Build the MLIR-AIR fused-decode templates for any model in fused_decode.py's _MODELS.
#   build_decode.sh <decode-model-name> <MODEL_TYPE macro> <out-dir> [LBUILD=2048]
# Emits <out-dir>/decode_L$LBUILD.{xclbin,insts.bin} + decode_L$((LBUILD-1)).{...}
set -euo pipefail
NAME=$1; MT=$2; OUT=$3; LBUILD=${4:-2048}; LREF=$((LBUILD-1))
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AIR=${OPENNPU_AIR:-$(cd "$ROOT" && python3 -c "from opennpu import config; print(config.AIR)")}
AIRENV=${OPENNPU_AIRENV:-$(cd "$ROOT" && python3 -c "from opennpu import config; print(config.AIRENV)")}
DEC=$AIR/programming_examples/fused_decode
set +u; source $AIRENV >/dev/null; set -u
AIEOPT_DIR=$(realpath "$(dirname "$(which aie-opt)")/..")
ENV=(DECODE_MODEL=$NAME UNIFIED=1 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1)
GLU=$(env "${ENV[@]}" FUSED_DECODE_PRINT_CONST=GLU_SLICE python3 $DEC/fused_decode.py 2>/dev/null || true)
KB=(-std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined
    -Wno-empty-body -Wno-deprecated-declarations -DNDEBUG -DMODEL_TYPE=$MT -D__AIE_API_AIE_ADF_HPP__
    -I $AIEOPT_DIR/include -I $DEC/kernels -I $DEC/models -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    ${GLU:+-DGLU_SLICE_EXPECTED=$GLU} ${KERNEL_EXTRA:-})
mkdir -p "$OUT"; LOG="$OUT/build.log"; : > "$LOG"
for k in proj_qmm rms_residual glu rope; do
  echo ">>> $k.cc (-O2)"; $PEANO_INSTALL_DIR/bin/clang++ "${KB[@]}" -O2 -c $DEC/kernels/$k.cc -o $DEC/$k.o
done
for k in attn_qk attn_kv; do
  echo ">>> $k.ll (-O1)"; $PEANO_INSTALL_DIR/bin/clang++ "${KB[@]}" -O1 -DDECODE_INLINE_ATTN -S -emit-llvm $DEC/kernels/$k.cc -o $DEC/$k.ll
done
for L in $LBUILD $LREF; do
  echo ">>> template decode_L$L"
  rm -f $DEC/decode.xclbin $DEC/decode.insts.bin
  (cd $DEC && env "${ENV[@]}" DECODE_GOLDEN_L=$L python3 fused_decode.py) >>"$LOG" 2>&1 || { tail -40 "$LOG"; exit 1; }
  [ -f $DEC/decode.xclbin ] || { echo "no xclbin"; tail -40 "$LOG"; exit 1; }
  mv -f $DEC/decode.xclbin "$OUT/decode_L$L.xclbin"; mv -f $DEC/decode.insts.bin "$OUT/decode_L$L.insts.bin"
done
env "${ENV[@]}" FUSED_DECODE_PRINT_CONST=MODEL_CONFIG_STAMP python3 $DEC/fused_decode.py > "$OUT/.decode_flags"
echo "built $NAME -> $OUT"
