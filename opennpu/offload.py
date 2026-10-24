"""Automatic partial offload of any PyTorch model to the NPU (Core-ML-style partitioning).

`offload(model)` swaps every eligible linear layer for an NPU-backed one; everything else
(attention softmax, norms, activations, convolutions, embeddings) stays on CPU. Works on any
`torch.nn.Module`, including Hugging Face transformers (Linear and GPT-2-style Conv1D).
"""
import time
from dataclasses import dataclass, field

import torch

from .engine import get_engine


class _CPUFallback(torch.nn.Module):
    """Compact CPU copy for small batches (LLM decode): int8 via fbgemm, else bf16 (AVX512-BF16)."""

    def __init__(self, w_nk: torch.Tensor, bias, int8: bool):
        super().__init__()
        out_f, in_f = w_nk.shape
        self.q = None
        if int8 and "fbgemm" in torch.backends.quantized.supported_engines:
            lin = torch.nn.Linear(in_f, out_f, bias=bias is not None)
            with torch.no_grad():
                lin.weight.copy_(w_nk)
                if bias is not None:
                    lin.bias.copy_(bias)
            self.q = torch.ao.quantization.quantize_dynamic(torch.nn.Sequential(lin), {torch.nn.Linear}, torch.qint8)
        else:
            self.register_buffer("w", w_nk.detach().to(torch.bfloat16).contiguous(), persistent=False)
            self.register_buffer("b", None if bias is None else bias.detach().float().clone(), persistent=False)

    def forward(self, x2):
        if self.q is not None:
            return self.q(x2.float())
        y = (x2.to(torch.bfloat16) @ self.w.t()).float()
        return y if self.b is None else y + self.b


class NPULinear(torch.nn.Module):
    AUTO = True        # per-row-count CPU/NPU timing, see forward()
    PROBE = 1

    def __init__(self, weight_nk: torch.Tensor, bias, min_tokens=16):
        super().__init__()
        e = get_engine()
        self._dtype = weight_nk.dtype
        w = weight_nk.detach().float().cpu()
        self.in_features = w.shape[1]
        self.out_features = w.shape[0]
        self.w = e.weight(w.numpy())
        self.register_buffer("bias", None if bias is None else bias.detach().float().clone(), persistent=False)
        self.min_tokens = min_tokens
        self.fallback = _CPUFallback(w, bias, e.int8) if min_tokens > 1 else None
        self.choice = {}       # rows -> "npu" | "cpu"
        self.stats = {}

    def _npu(self, x2):
        e = get_engine()
        if hasattr(e, "matmul_t"):
            y = e.matmul_t(x2, self.w)
        else:
            y = torch.from_numpy(e.matmul(x2.detach().float().cpu().numpy(), self.w))
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features)
        M = x2.shape[0]
        if self.fallback is None:
            y = self._npu(x2)
        elif M < self.min_tokens:
            y = self.fallback(x2)
        elif not self.AUTO or self.choice.get(M) == "npu":
            y = self._npu(x2)
        elif self.choice.get(M) == "cpu":
            y = self.fallback(x2)
        else:
            st = self.stats.setdefault(M, [0.0, 0.0, 0])
            t = time.perf_counter(); y = self._npu(x2); st[0] += time.perf_counter() - t
            t = time.perf_counter(); self.fallback(x2); st[1] += time.perf_counter() - t
            st[2] += 1
            if st[2] > self.PROBE:
                self.choice[M] = "npu" if st[0] <= st[1] else "cpu"
        return y.to(x.dtype).reshape(*shp[:-1], self.out_features)

    @property
    def weight(self):
        """CPU view of the weight for code that reads .weight directly (WavLM's attention calls
        F.linear(x, proj.weight)): in the layer's original dtype, rebuilt from the compact copy."""
        fb = self.fallback
        if fb is not None and fb.q is None:
            return fb.w.to(self._dtype)
        if fb is not None:
            return fb.q[0].weight().dequantize().to(self._dtype)
        return torch.empty(self.out_features, self.in_features, device="meta")

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, npu=xdna2/{get_engine().precision}"


class AutoPlace(torch.nn.Module):
    """Per-shape CPU-vs-NPU placement: the first `probe` calls of each input shape run both paths and
    time them; afterwards only the faster one runs. Correct for any layer, never slower than CPU."""

    def __init__(self, npu_mod, cpu_mod, probe=2):
        super().__init__()
        self.npu_mod, self.cpu_mod = npu_mod, cpu_mod
        self.probe = probe
        self.stats = {}       # shape -> [npu_s, cpu_s, n]
        self.choice = {}      # shape -> "npu" | "cpu"

    def forward(self, x):
        key = tuple(x.shape)
        c = self.choice.get(key)
        if c is not None:
            return self.npu_mod(x) if c == "npu" else self.cpu_mod(x)
        st = self.stats.setdefault(key, [0.0, 0.0, 0])
        t = time.perf_counter(); y = self.npu_mod(x); st[0] += time.perf_counter() - t
        t = time.perf_counter(); self.cpu_mod(x); st[1] += time.perf_counter() - t
        st[2] += 1
        if st[2] > self.probe:                     # first call includes warm-up; decide after probe+1
            self.choice[key] = "npu" if st[0] <= st[1] else "cpu"
        return y

    @property
    def weight(self):                 # some models read .weight/.bias for dtype/device checks
        return self.cpu_mod.weight

    @property
    def bias(self):
        return getattr(self.cpu_mod, "bias", None)

    def extra_repr(self):
        return f"choices={self.choice}"


@dataclass
class OffloadReport:
    offloaded: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    npu_params: int = 0
    total_params: int = 0

    def __str__(self):
        pct = 100 * self.npu_params / max(self.total_params, 1)
        return (f"OpenNPU: {len(self.offloaded)} layers on NPU, {len(self.skipped)} kept on CPU, "
                f"{self.npu_params/1e6:.1f}M / {self.total_params/1e6:.1f}M params ({pct:.0f}%) on NPU")


class NPUConv2d(torch.nn.Module):
    """Conv2d (groups=1) as im2col + NPU GEMM: (B*L, Cin*kh*kw) @ W(Cout, Cin*kh*kw)^T."""

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        self.kernel_size, self.stride = conv.kernel_size, conv.stride
        self.padding, self.dilation = conv.padding, conv.dilation
        self.out_channels = conv.out_channels
        self.lin = NPULinear(conv.weight.detach().reshape(conv.out_channels, -1), conv.bias, min_tokens=1)

    def forward(self, x):
        B, _, H, W = x.shape
        kh, kw = self.kernel_size
        Ho = (H + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        Wo = (W + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        cols = torch.nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        y = self.lin(cols.transpose(1, 2))                         # (B, L, Cout)
        return y.transpose(1, 2).reshape(B, self.out_channels, Ho, Wo)


def _conv1d_cls():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except Exception:  # noqa: BLE001
        return None


def offload(model: torch.nn.Module, min_features=256, min_tokens=16, skip=(), conv=True, auto=True,
            max_out=65536) -> OffloadReport:
    """Swap eligible layers in-place. Returns a report.

    Eligible: nn.Linear / HF Conv1D with in and out features >= min_features, not owned by
    nn.MultiheadAttention (which reads .weight directly), and whose qualified name contains none of `skip`.

    Calls with fewer than `min_tokens` rows (e.g. LLM decode steps) run on the CPU copy: the NPU
    tile is 128 rows, so single-token GEMV is faster on CPU until a dedicated GEMV overlay exists.
    This gives the NPU-prefill / CPU-decode hybrid split by default.

    Layers with more than `max_out` outputs (LLM vocabulary heads, 150k-260k rows) stay on the CPU:
    generation only ever feeds them the last token (CPU path anyway), and packing a 200k x 3k head
    for the NPU takes >16 GB of host RAM.
    """
    NPULinear.AUTO = auto
    conv1d = _conv1d_cls()
    rep = OffloadReport(total_params=sum(p.numel() for p in model.parameters()))
    _offload_onnx_matmuls(model, rep, min_features, min_tokens)
    for pname, parent in list(model.named_modules()):
        if isinstance(parent, torch.nn.MultiheadAttention):
            continue
        for cname, child in list(parent.named_children()):
            qn = f"{pname}.{cname}" if pname else cname
            if isinstance(child, torch.nn.Linear):
                w, b = child.weight, child.bias                   # (out, in)
            elif conv1d is not None and isinstance(child, conv1d):
                w, b = child.weight.t(), child.bias               # Conv1D stores (in, out)
            elif (conv and type(child) is torch.nn.Conv2d and child.groups == 1
                  and isinstance(child.padding, tuple) and child.padding_mode == "zeros"):
                k_in = child.in_channels * child.kernel_size[0] * child.kernel_size[1]
                if k_in < min_features or child.out_channels < 64 or any(s in qn for s in skip):
                    rep.skipped.append((qn, (k_in, child.out_channels)))
                    continue
                setattr(parent, cname, AutoPlace(NPUConv2d(child), child) if auto else NPUConv2d(child))
                rep.offloaded.append((qn, (k_in, child.out_channels)))
                rep.npu_params += child.weight.numel()
                continue
            else:
                continue
            out_f, in_f = w.shape
            if in_f < min_features or out_f < min_features or out_f > max_out or any(s in qn for s in skip):
                rep.skipped.append((qn, (in_f, out_f)))
                continue
            setattr(parent, cname, NPULinear(w, b, min_tokens=min_tokens))
            rep.offloaded.append((qn, (in_f, out_f)))
            rep.npu_params += w.numel()
            del child, w, b                                       # free the fp32 original
    return rep


class _ConstMatMul(torch.nn.Module):
    """Replaces onnx2torch OnnxMatMul(x, W[K,N]) -- called with (x, W) -- by an NPU GEMM on x."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x, _w=None):
        return self.lin(x)


def _offload_onnx_matmuls(model, rep, min_features, min_tokens):
    """onnx2torch graphs keep weights as initializers fed to OnnxMatMul / OnnxGemm nodes."""
    import torch.fx
    if not isinstance(model, torch.fx.GraphModule):
        return
    for node in model.graph.nodes:
        if node.op != "call_module" or len(node.args) < 2:
            continue
        mod = model.get_submodule(node.target)
        name = type(mod).__name__
        if name not in ("OnnxMatMul", "OnnxGemm"):
            continue
        w_node = node.args[1]
        if getattr(w_node, "op", None) != "get_attr":
            continue
        w = model.get_parameter(w_node.target) if w_node.target in dict(model.named_parameters()) \
            else model.get_buffer(w_node.target)
        if w.dim() != 2:
            continue
        bias = None
        if name == "OnnxGemm":
            if getattr(mod, "alpha", 1.0) != 1.0 or getattr(mod, "trans_a", 0):
                continue
            w_nk = w if getattr(mod, "trans_b", 0) else w.t()
            if len(node.args) > 2 and getattr(node.args[2], "op", None) == "get_attr":
                bias = model.get_buffer(node.args[2].target) if node.args[2].target in dict(model.named_buffers()) \
                    else model.get_parameter(node.args[2].target)
                bias = bias * getattr(mod, "beta", 1.0)
                node.args = node.args[:2]
        else:
            w_nk = w.t()                                           # ONNX MatMul: x[.., K] @ W[K, N]
        out_f, in_f = w_nk.shape
        if in_f < min_features or out_f < min_features:
            rep.skipped.append((node.target, (in_f, out_f)))
            continue
        model.add_submodule(node.target, _ConstMatMul(NPULinear(w_nk.contiguous(), bias, min_tokens=min_tokens)))
        rep.offloaded.append((node.target, (in_f, out_f)))
        rep.npu_params += w.numel()
        rep.total_params += w.numel()
    model.recompile()


def parity(ref_fn, npu_fn, n=3):
    """Run both callables, return (max relative error of outputs, ref ms, npu ms)."""
    def first_tensor(o):
        if isinstance(o, torch.Tensor):
            return o
        if hasattr(o, "logits"):
            return o.logits
        if hasattr(o, "last_hidden_state"):
            return o.last_hidden_state
        if isinstance(o, (tuple, list)):
            return first_tensor(o[0])
        if isinstance(o, dict):                  # other ModelOutputs (CLIP: logits_per_image, ...)
            for v in o.values():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    return v
        raise TypeError(type(o))

    def timed(f):
        f()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            o = f()
            ts.append(time.perf_counter() - t)
        return first_tensor(o).float(), sorted(ts)[n // 2] * 1e3

    r, rt = timed(ref_fn)
    y, yt = timed(npu_fn)
    r, y = r.double(), y.double()          # fp32 sums over ~1e7 logits drift past cos=1
    rel = ((y - r).norm() / r.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), r.flatten(), dim=0).item()
    return rel, cos, rt, yt
