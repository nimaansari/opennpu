"""Throw many unrelated models at OpenNPU's universal path and record what happens.

    python tests/sweep.py [MODEL ...] [--out results.jsonl] [--timeout 900]

Each model runs in its own subprocess (a crash or hang cannot take the sweep down) through
the same code a user hits: loaders.load -> offload -> forward, checked against CPU fp32.
With no models given it runs ZOO, a deliberately mixed bag of families and modalities.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ZOO = [
    # text encoders
    "FacebookAI/roberta-base", "distilbert/distilbert-base-uncased", "albert/albert-base-v2",
    "google/electra-small-discriminator", "microsoft/deberta-v3-small", "answerdotai/ModernBERT-base",
    "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-small-v2", "BAAI/bge-m3",
    # seq2seq
    "google-t5/t5-small", "google/flan-t5-base", "facebook/bart-base", "Helsinki-NLP/opus-mt-en-de",
    # decoder LLMs (prefill parity)
    "HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen3-0.6B", "EleutherAI/pythia-160m", "facebook/opt-125m",
    "bigscience/bloom-560m", "LiquidAI/LFM2-350M", "state-spaces/mamba-130m-hf", "allenai/OLMo-2-0425-1B",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct", "microsoft/phi-1_5", "ibm-granite/granite-3.3-2b-instruct",
    # vision
    "microsoft/resnet-50", "facebook/convnext-tiny-224", "facebook/dinov2-small",
    "microsoft/swin-tiny-patch4-window7-224", "google/efficientnet-b0", "apple/mobilevit-small",
    "nvidia/segformer-b0-finetuned-ade-512-512", "facebook/deit-small-patch16-224", "microsoft/beit-base-patch16-224",
    "timm/resnet18.a1_in1k", "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    # vision-language
    "openai/clip-vit-base-patch32", "google/siglip-base-patch16-224",
    # audio
    "openai/whisper-tiny", "facebook/wav2vec2-base-960h", "facebook/hubert-base-ls960", "microsoft/wavlm-base",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    # other formats
    "Xenova/all-MiniLM-L6-v2:onnx/model.onnx",
]

CHILD = r'''
import json, sys, time, copy, traceback
import torch
torch.set_num_threads(4)
ref_id = sys.argv[1]
res = {"model": ref_id}
try:
    from opennpu import offload, parity
    from opennpu.loaders import load
    t = time.time()
    model, tok, kind = load(ref_id)
    res["kind"], res["load_s"] = kind, round(time.time() - t, 1)
    res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
    need = res["params_M"] * 1e6 * 2 * 2                # NPU bfp16 copy + bf16 CPU fallback copy
    if need > avail * 0.8:
        raise MemoryError(f"needs ~{need / 2**30:.0f} GiB more, {avail / 2**30:.0f} GiB available; skipped")
    repo = ref_id.split(":")[0]
    text = "OpenNPU runs models on AMD Ryzen AI NPUs under Linux. " * 6
    name = getattr(model, "main_input_name", "input_ids")
    cfg = getattr(model, "config", None)

    def image_inputs():
        from transformers import AutoImageProcessor
        import numpy as np
        img = (np.random.RandomState(0).rand(256, 256, 3) * 255).astype("uint8")
        return dict(AutoImageProcessor.from_pretrained(repo)(images=img, return_tensors="pt"))

    def audio_inputs():
        from transformers import AutoFeatureExtractor
        import numpy as np
        wav = (np.random.RandomState(0).randn(16000 * 5) * 0.05).astype("float32")
        return dict(AutoFeatureExtractor.from_pretrained(repo)(wav, sampling_rate=16000, return_tensors="pt"))

    if kind in ("onnx", "timm", "torch"):
        if kind == "onnx":
            ids = torch.randint(100, 2000, (1, 64))
            names = [n.name for n in __import__("onnx").load(__import__("huggingface_hub").hf_hub_download(repo, ref_id.split(":", 1)[1])).graph.input]
            inp = {n: (ids if "input" in n else torch.ones_like(ids) if "mask" in n else torch.zeros_like(ids)) for n in names}
            call = lambda m: m(*inp.values())
        else:
            x = torch.randn(1, 3, 224, 224)
            call = lambda m: m(x)
    else:
        mm = cfg is not None and hasattr(cfg, "vision_config") and hasattr(cfg, "text_config")
        if mm:
            from transformers import AutoProcessor
            import numpy as np
            img = (np.random.RandomState(0).rand(224, 224, 3) * 255).astype("uint8")
            inp = dict(AutoProcessor.from_pretrained(repo)(text=["a photo of a cat"], images=img, return_tensors="pt", padding="max_length"))
        elif name == "pixel_values":
            inp = image_inputs()
        elif name in ("input_values", "input_features"):
            inp = audio_inputs()
            if kind == "whisper" or getattr(cfg, "is_encoder_decoder", False):
                inp["decoder_input_ids"] = torch.tensor([[cfg.decoder_start_token_id or 0] * 4])
        else:
            inp = dict(tok(text, return_tensors="pt"))
            inp.pop("token_type_ids", None) if "token_type_ids" not in model.forward.__code__.co_varnames else None
            if getattr(cfg, "is_encoder_decoder", False):
                inp["decoder_input_ids"] = torch.full((1, 8), cfg.decoder_start_token_id or 0)
        call = lambda m: m(**inp)
    # CPU reference first, then offload the SAME model in place: a second fp32 copy doubled
    # peak RAM and got 3-4B models OOM-killed (taking other processes with them)
    with torch.no_grad():
        ref_out, rt = None, None
        call(model)
        t = time.perf_counter(); ref_out = call(model); rt = (time.perf_counter() - t) * 1e3
    rep = offload(model)
    res["npu_params_M"] = round(rep.npu_params / 1e6, 1)
    with torch.no_grad():
        for _ in range(3):
            call(model)                     # let per-layer auto-placement settle
        rel, cos, _, yt = parity(lambda: ref_out, lambda: call(model))
    res.update(cos=round(cos, 5), rel=round(rel, 4), cpu_ms=round(rt, 1), npu_ms=round(yt, 1),
               speedup=round(rt / yt, 2), status="ok" if cos > 0.99 else "MISMATCH")
except Exception as e:
    res["status"] = "ERROR"
    res["error"] = f"{type(e).__name__}: {e}"[:400]
    res["where"] = traceback.format_exc().strip().splitlines()[-3][:200]
print("RESULT " + json.dumps(res), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--out", default="sweep_results.jsonl")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["model"] for line in open(a.out) if line.strip()}
    for m in a.models or ZOO:
        if m in done:
            continue
        t = time.time()
        try:
            p = subprocess.run([sys.executable, "-c", CHILD, m], capture_output=True, text=True,
                               timeout=a.timeout, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
            res = json.loads(line[7:]) if line else {"model": m, "status": "CRASH", "rc": p.returncode,
                                                     "error": (p.stderr or "")[-400:]}
        except subprocess.TimeoutExpired:
            res = {"model": m, "status": "TIMEOUT"}
        res["wall_s"] = round(time.time() - t, 1)
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
