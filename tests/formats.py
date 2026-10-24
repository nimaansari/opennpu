import copy
import sys
import torch
sys.path.insert(0, ".")
from opennpu import offload, parity
from opennpu.loaders import load
torch.set_num_threads(4)
cases = [("timm/resnet50.a1_in1k", lambda: dict(x=torch.randn(1, 3, 224, 224))),
         ("timm/convnext_tiny.fb_in1k", lambda: dict(x=torch.randn(1, 3, 224, 224))),
         ("Xenova/all-MiniLM-L6-v2:onnx/model.onnx", None)]
for ref_id, inp in cases:
    try:
        if ref_id.endswith(".onnx"):
            ref, _, kind = load(ref_id)
            ids = torch.randint(100, 2000, (1, 128))
            args = (ids, torch.ones_like(ids), torch.zeros_like(ids))
            f_ref = lambda m: m(*args)
        else:
            ref, _, kind = load(ref_id)
            x = inp()["x"]
            f_ref = lambda m: m(x)
        npu = copy.deepcopy(ref); rep = offload(npu, min_features=128)
        with torch.no_grad():
            [f_ref(npu) for _ in range(4)]   # let auto-placement probe
            rel, cos, rt, yt = parity(lambda: f_ref(ref), lambda: f_ref(npu))
        print(f"{ref_id:40s} [{kind}] {rep}\n    cos={cos:.4f} rel={rel:.3f}  cpu {rt:.1f}ms npu {yt:.1f}ms", flush=True)
    except Exception as e:
        print(f"{ref_id:40s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
