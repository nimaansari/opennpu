"""Extra NPU layers for generation models that opennpu.offload does not cover yet.

Conv1d is the backbone of most TTS models (VITS/MMS, HiFi-GAN-style vocoders, SpeechT5's
post-net). It runs as im2col + one NPU GEMM, like NPUConv2d, inside AutoPlace so a layer only
stays on the NPU where that is measured to be faster than the CPU.
"""
import torch


class NPUConv1d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv1d):
        super().__init__()
        from opennpu.offload import NPULinear
        self.k, self.s = conv.kernel_size[0], conv.stride[0]
        self.d = conv.dilation[0]
        self.p = conv.padding[0] if isinstance(conv.padding, tuple) else _same_pad(conv)
        self.out_channels = conv.out_channels
        w = conv.weight.detach().reshape(conv.out_channels, -1)
        self.lin = NPULinear(w, conv.bias, min_tokens=1)

    def forward(self, x):
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)
        B, C, L = x.shape
        cols = torch.nn.functional.unfold(x.unsqueeze(2), (1, self.k), dilation=(1, self.d),
                                          padding=(0, self.p), stride=(1, self.s))   # (B, C*k, Lout)
        y = self.lin(cols.transpose(1, 2)).transpose(1, 2)                         # (B, Cout, Lout)
        return y[0] if unbatched else y


def _same_pad(conv):
    if conv.padding == "valid":
        return 0
    return conv.dilation[0] * (conv.kernel_size[0] - 1) // 2      # "same" with an odd kernel


def _eligible(c, min_features):
    if type(c) is not torch.nn.Conv1d or c.groups != 1 or c.padding_mode != "zeros":
        return False
    if c.padding == "same" and c.kernel_size[0] % 2 == 0:
        return False
    return c.in_channels * c.kernel_size[0] >= min_features and c.out_channels >= 64


def offload_conv1d(model, mode="auto", min_features=256):
    """Swap eligible nn.Conv1d layers in place; returns (layers, params) moved."""
    from opennpu.offload import AutoPlace
    n = params = 0
    for _, parent in list(model.named_modules()):
        for cname, child in list(parent.named_children()):
            if not _eligible(child, min_features):
                continue
            npu = NPUConv1d(child)
            setattr(parent, cname, AutoPlace(npu, child) if mode == "auto" else npu)
            n += 1
            params += child.weight.numel()
    return n, params
