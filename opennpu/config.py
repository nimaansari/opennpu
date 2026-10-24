"""Where OpenNPU keeps things. Every path has an environment override; defaults follow XDG.

  OPENNPU_HOME     data root (templates, kernels, sessions)    ~/.local/share/opennpu
  OPENNPU_CACHE    converted weight cache                       ~/.cache/opennpu
  OPENNPU_MODELS   checkpoints the server lists                 ~/models if it exists, else $OPENNPU_HOME/models
  OPENNPU_AIR      MLIR-AIR checkout (native engine builder)     $OPENNPU_HOME/toolchain/mlir-air
  OPENNPU_AIRENV   env script activating pyxrt + mlir-air         $OPENNPU_HOME/toolchain/airenv/env.sh
  OPENNPU_URL      server the CLI talks to                       http://127.0.0.1:11540
  OPENNPU_PREFILL_AIR  MLIR-AIR checkout with the fused prefill     $OPENNPU_HOME/toolchain/mlir-air-prefill
  OPENNPU_PREFILL_ENV  env script of its build toolchain            $OPENNPU_HOME/toolchain/airprefill/env.sh

  OPENNPU_DEV_TREE  optional developer tree; its mlir-air, airenv/env.sh, mlir-air-up and
                    airnext/env.sh are used when the toolchain is not installed under OPENNPU_HOME
"""
import os
from pathlib import Path

_home = Path.home()
_xdg_data = Path(os.environ.get("XDG_DATA_HOME", _home / ".local/share"))
_xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", _home / ".cache"))
_dev = Path(os.environ["OPENNPU_DEV_TREE"]).expanduser() if os.environ.get("OPENNPU_DEV_TREE") else None

HOME = Path(os.environ.get("OPENNPU_HOME", _xdg_data / "opennpu"))
CACHE = Path(os.environ.get("OPENNPU_CACHE", _xdg_cache / "opennpu"))
PORT = 11540
URL = os.environ.get("OPENNPU_URL", os.environ.get("NPU_URL", f"http://127.0.0.1:{PORT}"))
SERVICE = "opennpu-npu"


def _first(env, *cands):
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return Path(cands[0])


MODELS = _first("OPENNPU_MODELS", _home / "models", HOME / "models")
if not os.environ.get("OPENNPU_MODELS") and not MODELS.exists():
    MODELS = HOME / "models"
AIR = _first("OPENNPU_AIR", HOME / "toolchain/mlir-air", _dev and _dev / "mlir-air")
AIRENV = _first("OPENNPU_AIRENV", HOME / "toolchain/airenv/env.sh", _dev and _dev / "airenv/env.sh")
PREFILL_AIR = _first("OPENNPU_PREFILL_AIR", HOME / "toolchain/mlir-air-prefill", _dev and _dev / "mlir-air-up")
PREFILL_ENV = _first("OPENNPU_PREFILL_ENV", HOME / "toolchain/airprefill/env.sh", _dev and _dev / "airnext/env.sh")
NATIVE_TEMPLATES = HOME / "native"
MOE_TEMPLATES = Path(os.environ.get("OPENNPU_MOE_TEMPLATES", HOME / "native-moe"))
NATIVE_CACHE = CACHE / "native"
