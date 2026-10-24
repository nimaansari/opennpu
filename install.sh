#!/usr/bin/env bash
# OpenNPU installer: local models on AMD Ryzen AI (XDNA2) NPUs under Linux.
#
#   ./install.sh                 universal path + web UI/server + `npu` CLI   (~2 GB)
#   ./install.sh --native        also the own fused NPU engines (dense 8-14B, Qwen3-30B-A3B MoE)
#   ./install.sh --gen           also image / video / voice generation (diffusers, ~250 MB)
#   ./install.sh --check         only check the system, change nothing
#   ./install.sh --no-service    install files only (no ~/.local/bin links, no systemd unit)
#
# Installs into $OPENNPU_HOME (default ~/.local/share/opennpu), a user systemd service
# `opennpu-npu` (127.0.0.1:11540, localhost only) and `npu` / `opennpu` in ~/.local/bin.
# Never uses sudo: system prerequisites it cannot fix are printed with the exact command.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
HOME_DIR=${OPENNPU_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/opennpu}
BIN=$HOME/.local/bin
PYVER=3.12                                   # pyxrt (python3-xrt) and the AIE wheels are built for 3.12
NATIVE=0; CHECK=0; SERVICE=1; GEN=0
for a in "$@"; do
  case $a in
    --native) NATIVE=1 ;;
    --gen) GEN=1 ;;
    --check) CHECK=1 ;;
    --no-service) SERVICE=0 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

c_ok=$'\e[32m✔\e[0m'; c_bad=$'\e[31m✘\e[0m'; c_warn=$'\e[33m!\e[0m'
say() { printf '%s\n' "$*"; }
step() { printf '\n\e[1m%s\e[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. system check
step "Checking the system"
problems=0
need() {  # need <ok?> <what> <fix>
  if [ "$1" = 1 ]; then say " $c_ok $2"; else say " $c_bad $2"; say "     fix: $3"; problems=$((problems + 1)); fi
}
kv=$(uname -r | cut -d. -f1-2)
need "$(awk -v v="$kv" 'BEGIN{split(v,a,"."); print (a[1]>6 || (a[1]==6 && a[2]>=14)) ? 1 : 0}')" \
  "Linux kernel $(uname -r) (need >= 6.14 for the in-kernel amdxdna driver)" \
  "Ubuntu: sudo apt install linux-oem-24.04   (or any distro kernel >= 6.14)"
need "$([ -e /dev/accel/accel0 ] && echo 1 || echo 0)" "NPU device /dev/accel/accel0" \
  "sudo modprobe amdxdna; check: sudo dmesg | grep -i amdxdna   (firmware missing? sudo apt install linux-firmware)"
need "$(command -v xrt-smi >/dev/null && echo 1 || echo 0)" "XRT runtime (xrt-smi)" \
  "sudo add-apt-repository ppa:amd-team/xrt && sudo apt install libxrt2 libxrt-npu2 libxrt-utils libxrt-utils-npu"
need "$(id -nG | tr ' ' '\n' | grep -qx render && echo 1 || echo 0)" "user in the 'render' group" \
  "sudo usermod -aG render \$USER   (then log out and back in)"
ml=$(ulimit -l)
if [ "$ml" = unlimited ]; then say " $c_ok memlock unlimited"
else say " $c_warn memlock $ml KB in this shell (the service sets LimitMEMLOCK=infinity itself; for CLI use add"
     say "     '* soft memlock unlimited' + '* hard memlock unlimited' to /etc/security/limits.conf, then re-login)"; fi
if command -v xrt-smi >/dev/null; then
  arch=$(xrt-smi examine 2>/dev/null | grep -oE 'aie2p?' | head -1 || true)
  [ "$arch" = aie2p ] && say " $c_ok NPU is XDNA2 (aie2p)" || say " $c_warn NPU arch '${arch:-unknown}': prebuilt kernels target XDNA2 (Strix/Krackan/Strix Halo)"
fi
if [ $CHECK = 1 ]; then
  [ $problems = 0 ] && say $'\nSystem ready.' || say $'\n'"$problems problem(s) to fix first."
  exit $([ $problems = 0 ] && echo 0 || echo 1)
fi
[ $problems = 0 ] || { say $'\n'"Fix the $problems problem(s) above, then run ./install.sh again."; exit 1; }

# ---------------------------------------------------------------- 2. python 3.12 (private, survives OS upgrades)
step "Python $PYVER"
if ! command -v uv >/dev/null; then
  say " installing uv (Python package manager) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH=$HOME/.local/bin:$PATH
fi
# a uv-managed interpreter inside OPENNPU_HOME: an OS upgrade that replaces /usr/bin/python3.x
# can't pull it out from under the venv (that is how v0.2 installs broke on Ubuntu 26.04)
UV_PYTHON_INSTALL_DIR=$HOME_DIR/python uv python install -q $PYVER
PY=$(UV_PYTHON_INSTALL_DIR=$HOME_DIR/python uv python find --managed-python $PYVER)
say " $c_ok $PY"

# ---------------------------------------------------------------- 3. venv + packages
step "Python environment ($HOME_DIR/env)"
ENV=$HOME_DIR/env
[ -x "$ENV/bin/python" ] && "$ENV/bin/python" -c pass 2>/dev/null || uv venv -q --clear --python "$PY" "$ENV"
pip() { uv pip install -q --python "$ENV/bin/python" "$@"; }
# torch + torchvision together from the CPU index: a PyPI torchvision next to a +cpu torch
# fails at import ("operator torchvision::nms does not exist")
pip --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match torch torchvision
pip -e "$ROOT[universal,native,formats]"
# AMD IRON's open flm.GEMM (default NPU GEMM backend, ~4x the bundled overlays) + the AIE
# toolchain it compiles with. The native engines are validated on these same pins.
AIE_PINS=("mlir_aie==1.4.4.dev53+gd53582d" "llvm-aie==22.0.0.2026091701+773413fb")
AIE_LINKS=(-f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
           -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly)
pip --index-strategy unsafe-best-match "${AIE_PINS[@]}" "${AIE_LINKS[@]}"
pip --no-deps "iron @ git+https://github.com/amd/iron@821244f"
say " $c_ok opennpu $("$ENV/bin/python" -c 'import opennpu; print(opennpu.__version__)')"

# ---------------------------------------------------------------- 4. pyxrt for this Python
step "XRT Python bindings (pyxrt)"
SP=$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
SO=pyxrt.cpython-312-x86_64-linux-gnu.so
if ! "$ENV/bin/python" -c 'import pyxrt' 2>/dev/null; then
  if [ -f /usr/lib/python3/dist-packages/$SO ]; then
    ln -sfn /usr/lib/python3/dist-packages/$SO "$SP/$SO"
  else
    # distro Python is not 3.12 (e.g. Ubuntu 26.04 ships 3.14): take the binding from AMD's
    # noble build of the *installed* XRT version and keep a private copy
    xv=$(dpkg-query -W -f='${Version}' libxrt2 2>/dev/null | sed 's/^1://; s/~.*//' || true)
    [ -n "$xv" ] || { say " $c_bad cannot tell the XRT version; install python3-xrt for Python 3.12 by hand"; exit 1; }
    tmp=$(mktemp -d)
    url="https://ppa.launchpadcontent.net/amd-team/xrt/ubuntu/pool/main/x/xrt/python3-xrt_${xv}~noble1_amd64.deb"
    say " fetching $url"
    curl -fsSL "$url" -o "$tmp/px.deb" && dpkg-deb -x "$tmp/px.deb" "$tmp/x"
    mkdir -p "$HOME_DIR/pyxrt"
    cp "$tmp/x/usr/lib/python3/dist-packages/$SO" "$HOME_DIR/pyxrt/"
    ln -sfn "$HOME_DIR/pyxrt/$SO" "$SP/$SO"
    rm -rf "$tmp"
  fi
fi
"$ENV/bin/python" -c 'import pyxrt; pyxrt.device(0)' && say " $c_ok pyxrt opens the NPU"

if [ $GEN = 1 ]; then
  step "Generation (images / video / voice)"
  pip -e "$ROOT[gen]"
  pip --no-deps "diffusers>=0.35"        # see pyproject: its hub<2 pin is stale
  "$ENV/bin/python" -c 'import diffusers, imageio_ffmpeg, soundfile' && say " $c_ok diffusers, ffmpeg and soundfile import"
fi

# ---------------------------------------------------------------- 5. optional: native fused engines
mkdir -p "$HOME_DIR/toolchain/airenv"
ENVSH=$HOME_DIR/toolchain/airenv/env.sh
if [ $NATIVE = 1 ]; then
  step "Native fused engines (MLIR-AIR toolchain, ~3 GB)"
  # pinned to the versions OpenNPU's fused-decode patch is validated against
  # mlir_air on the same mlir_aie/llvm-aie pins as IRON (fused-decode output verified identical)
  pip --index-strategy unsafe-best-match "mlir_air==0.0.1.2026100104" "${AIE_PINS[@]}" \
    -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels "${AIE_LINKS[@]}"
  AIR=$HOME_DIR/toolchain/mlir-air
  if [ ! -d "$AIR/.git" ]; then
    git clone -q --filter=blob:none https://github.com/Xilinx/mlir-air "$AIR"
  fi
  git -C "$AIR" checkout -q b755300
  if git -C "$AIR" apply --check "$ROOT/opennpu/native/patches/mlir-air-fused-decode-dense-and-moe.patch" 2>/dev/null; then
    git -C "$AIR" apply "$ROOT/opennpu/native/patches/mlir-air-fused-decode-dense-and-moe.patch"
  fi
  LL=$(command -v llvm-link-18 || command -v llvm-link || true)
  [ -n "$LL" ] || { say " $c_bad llvm-link missing: sudo apt install llvm-18"; exit 1; }
  mkdir -p "$HOME_DIR/toolchain/llvmlink"; ln -sfn "$LL" "$HOME_DIR/toolchain/llvmlink/llvm-link"
  say " $c_ok mlir-air @ b755300 + OpenNPU patch"

  # Batched prompt prefill: MLIR-AIR's fused prefill device (int4 GEMMs + NPU attention). Its
  # sources need a newer toolchain than the decode pin, so they get their own checkout and a
  # build-only venv; the devices it compiles run in the main env. Built per model on first use.
  PRE=$HOME_DIR/toolchain/mlir-air-prefill
  if [ ! -d "$PRE/.git" ]; then
    git clone -q --filter=blob:none https://github.com/Xilinx/mlir-air "$PRE"
  fi
  git -C "$PRE" checkout -q 6e81ce12104de385c693d56dbce6d7912c22b91a
  PENV=$HOME_DIR/toolchain/airprefill
  [ -x "$PENV/bin/python" ] || uv venv -q --python "$PY" "$PENV"
  uv pip install -q --python "$PENV/bin/python" --index-strategy unsafe-best-match numpy ml_dtypes \
    "mlir_air==0.0.1.2026100717" "mlir_aie==1.4.4.dev82+gbe5b809" "llvm-aie==22.0.0.2026092101+0006955e" \
    -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels "${AIE_LINKS[@]}"
  PSP=$("$PENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  ESP=$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  for f in "$ESP"/pyxrt*.so; do [ -e "$f" ] && ln -sfn "$f" "$PSP/"; done
  {
    echo "# generated by OpenNPU install.sh: the toolchain that builds the fused prefill devices"
    echo "source \"$PENV/bin/activate\""
    echo "export MLIR_AIR_INSTALL_DIR=$PSP/mlir_air MLIR_AIE_INSTALL_DIR=$PSP/mlir_aie PEANO_INSTALL_DIR=$PSP/llvm-aie"
    echo "export PATH=$PSP/mlir_air/bin:$PSP/mlir_aie/bin:$HOME_DIR/toolchain/llvmlink:\$PATH"
    echo "export PYTHONPATH=$PSP/mlir_air/python:$PSP/mlir_aie/python:\${PYTHONPATH:-}"
    echo "export LD_LIBRARY_PATH=$PSP/mlir_air/lib:$PSP/mlir_aie/lib:\${LD_LIBRARY_PATH:-}"
  } > "$PENV/env.sh"
  bash -c "source '$PENV/env.sh' && python3 -c 'import aie, air'" && say " $c_ok fused prefill toolchain (mlir-air @ 6e81ce1)"

  # llama.cpp with OpenNPU's NPU backend: GGUF models (incl. an LM Studio folder) on the NPU
  if command -v cmake >/dev/null && command -v ninja >/dev/null && command -v g++ >/dev/null; then
    OPENNPU_HOME="$HOME_DIR" "$ROOT/llama/build.sh" "$HOME_DIR/llama.cpp" >/dev/null 2>&1 \
      && say " $c_ok llama.cpp with the NPU backend (GGUF models)" \
      || say " $c_warn llama.cpp NPU backend build failed: run llama/build.sh to see why"
  else
    say " $c_warn llama.cpp NPU backend skipped: needs cmake, ninja and g++ (sudo apt install cmake ninja-build g++)"
  fi
fi
cat > "$ENVSH" <<EOF
# generated by OpenNPU install.sh: activates the env the server and fused engines run in
source "$ENV/bin/activate"
SP=\$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export MLIR_AIR_INSTALL_DIR=\$SP/mlir_air MLIR_AIE_INSTALL_DIR=\$SP/mlir_aie PEANO_INSTALL_DIR=\$SP/llvm-aie
export PATH=\$MLIR_AIR_INSTALL_DIR/bin:\$MLIR_AIE_INSTALL_DIR/bin:$HOME_DIR/toolchain/llvmlink:\$PATH
export PYTHONPATH=\$MLIR_AIR_INSTALL_DIR/python:\$MLIR_AIE_INSTALL_DIR/python:\${PYTHONPATH:-}
export LD_LIBRARY_PATH=\$MLIR_AIR_INSTALL_DIR/lib:\$MLIR_AIE_INSTALL_DIR/lib:\${LD_LIBRARY_PATH:-}
EOF

# ---------------------------------------------------------------- 6. commands + service
if [ $SERVICE = 0 ]; then step "Done (files only)"; say "  $ENV/bin/opennpu doctor"; exit 0; fi
step "Commands and service"
mkdir -p "$BIN"
ln -sfn "$ENV/bin/npu" "$BIN/npu"
ln -sfn "$ENV/bin/opennpu" "$BIN/opennpu"
UNIT=$HOME/.config/systemd/user/opennpu-npu.service
mkdir -p "$(dirname "$UNIT")"
sed -e "s|@ROOT@|$ROOT|g" -e "s|@AIRENV@|$ENVSH|g" "$ROOT/packaging/opennpu-npu.service" > "$UNIT"
# OPENNPU_AIRENV pins the CLI/doctor to this install even when a dev tree also exists
mkdir -p "$UNIT.d"
printf '[Service]\nEnvironment=OPENNPU_HOME=%s\nEnvironment=OPENNPU_AIRENV=%s\n' "$HOME_DIR" "$ENVSH" > "$UNIT.d/install.conf"
systemctl --user daemon-reload
systemctl --user enable -q --now opennpu-npu
systemctl --user restart opennpu-npu
for _ in $(seq 30); do curl -fs http://127.0.0.1:11540/health >/dev/null && break; sleep 1; done
curl -fs http://127.0.0.1:11540/health >/dev/null && say " $c_ok server on http://127.0.0.1:11540" \
  || say " $c_warn server not up yet: journalctl --user -u opennpu-npu -n 50"
case ":$PATH:" in *":$BIN:"*) ;; *) say " $c_warn add $BIN to your PATH";; esac

# ---------------------------------------------------------------- 7. desktop app (GTK 4 + libadwaita)
step "Desktop app"
# runs on the distro Python (PyGObject comes from the distro); it only talks HTTP to the server
SYSPY=$(command -v python3 || true)
# libadwaita >= 1.4 (ToolbarView, Breakpoint, NavigationSplitView): Ubuntu 24.04+, Fedora 39+, Debian 13
if [ -n "$SYSPY" ] && "$SYSPY" -c "import gi; gi.require_version('Gtk','4.0'); gi.require_version('Adw','1'); from gi.repository import Adw; assert (Adw.get_major_version(), Adw.get_minor_version()) >= (1, 4)" 2>/dev/null; then
  printf '#!/bin/sh\nPYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}" OPENNPU_HOME="%s" exec "%s" -m opennpu.desktop "$@"\n' \
    "$ROOT" "$HOME_DIR" "$SYSPY" > "$BIN/opennpu-desktop"
  chmod +x "$BIN/opennpu-desktop"
  ICONS=${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps
  APPS=${XDG_DATA_HOME:-$HOME/.local/share}/applications
  mkdir -p "$ICONS" "$APPS"
  cp "$ROOT/opennpu/desktop/icons/hicolor/scalable/apps/io.github.opennpu.OpenNPU.svg" "$ICONS/"
  sed "s|^Exec=opennpu-desktop|Exec=\"$BIN/opennpu-desktop\"|" "$ROOT/packaging/io.github.opennpu.OpenNPU.desktop" \
    > "$APPS/io.github.opennpu.OpenNPU.desktop"
  command -v update-desktop-database >/dev/null && update-desktop-database -q "$APPS" || true
  command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "${ICONS%/scalable/apps}" 2>/dev/null || true
  say " $c_ok OpenNPU is in your app launcher (or run: opennpu-desktop)"
else
  say " $c_warn desktop app needs GTK 4 + libadwaita >= 1.4 for the system Python (the web UI works without it):"
  say "     sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1   (Fedora: python3-gobject libadwaita)"
fi

step "Done"
say "  Desktop    OpenNPU in your app launcher   (opennpu-desktop)"
say "  Web UI     http://127.0.0.1:11540"
say "  Chat       npu chat Qwen/Qwen3-0.6B            (any Hugging Face model id)"
say "  Models     npu list   |   npu pull <hf repo>"
say "  Check      opennpu doctor"
