"""Tier 2 model loaders: get *any* model into a torch.nn.Module so offload() can partition it.

Supported references:
  org/name                         Hugging Face transformers checkpoint (safetensors / bin)
  org/name-GGUF[:file.gguf]        GGUF of any architecture transformers can dequantize
  path/model.gguf                  local GGUF (needs the matching HF repo for config: --base)
  path/model.onnx | org/name.onnx  ONNX graph, converted with onnx2torch
  timm/<name>                      timm vision model
  path/model.pt                    a pickled torch.nn.Module (torch.save(model))
"""
import os

import torch


def _hf_files(repo):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo)


def kind_of(ref):
    low = ref.lower()
    if low.endswith(".onnx"):
        return "onnx"
    if low.endswith(".gguf") or "gguf" in low.split(":")[0].lower():
        return "gguf"
    if low.startswith("timm/"):
        return "timm"
    if low.endswith((".pt", ".pth")):
        return "torch"
    if os.path.isdir(ref) and any(f.endswith(".onnx") for f in os.listdir(ref)) \
            and not os.path.isfile(os.path.join(ref, "config.json")):
        return "onnx"
    return "hf"


def gguf_file(repo, prefer=("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "F16")):
    files = [f for f in _hf_files(repo) if f.endswith(".gguf")]
    for p in prefer:
        for f in files:
            if p.lower() in f.lower():
                return f
    if not files:
        raise FileNotFoundError(f"no .gguf in {repo}")
    return files[0]


def gguf_arch(path_or_repo_file):
    from gguf import GGUFReader
    r = GGUFReader(path_or_repo_file)
    f = r.fields["general.architecture"]
    return bytes(f.parts[f.data[0]]).decode()


def param_count(ref):
    """Parameters of an HF checkpoint without loading it (None if unknown)."""
    try:
        if os.path.isdir(ref):
            size = sum(os.path.getsize(os.path.join(ref, f)) for f in os.listdir(ref) if f.endswith(".safetensors"))
            return size // 2 if size else None            # checkpoints are almost always 16-bit
        from huggingface_hub import model_info
        st = model_info(ref.split(":")[0]).safetensors
        return int(st.total) if st else None
    except Exception:  # noqa: BLE001
        return None


def auto_dtype(ref, limit=1.5e9):
    """fp32 for small models (exact CPU parts), bf16 above ~1.5B parameters: the NPU computes in
    bf16 anyway, and fp32 master weights for a 4B model (16 GB) plus the NPU copies don't fit next
    to everything else on a typical 32-64 GB laptop."""
    n = param_count(ref)
    return torch.bfloat16 if n and n > limit else torch.float32


def load(ref, task="auto", dtype=torch.float32):
    """Returns (model, tokenizer_or_processor_or_None, kind). dtype="auto": see auto_dtype."""
    kind = kind_of(ref)
    if dtype == "auto":
        dtype = auto_dtype(ref) if kind in ("hf", "gguf") else torch.float32
    if kind == "onnx":
        import onnx
        from onnx2torch import convert
        path = ref
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            repo, _, fname = ref.partition(":")
            if not fname:
                fname = next(f for f in _hf_files(repo) if f.endswith(".onnx"))
            path = hf_hub_download(repo, fname)
        if os.path.isdir(path):
            path = os.path.join(path, next(f for f in os.listdir(path) if f.endswith(".onnx")))
        return convert(onnx.load(path)).eval(), None, kind
    if kind == "timm":
        import timm
        return timm.create_model(ref.split("/", 1)[1], pretrained=True).eval(), None, kind
    if kind == "torch":
        return torch.load(ref, weights_only=False).eval(), None, kind

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
    kw = dict(dtype=dtype)
    repo = ref
    if kind == "gguf":
        repo, _, fname = ref.partition(":")
        kw["gguf_file"] = fname or gguf_file(repo)
    cfg = AutoConfig.from_pretrained(repo, **({"gguf_file": kw["gguf_file"]} if "gguf_file" in kw else {}))
    archs = cfg.architectures or []
    causal = task in ("generate", "chat") or (task == "auto" and (
        kind == "gguf" or any(a.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")) for a in archs)))
    if "Whisper" in "".join(archs) or cfg.model_type == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        return WhisperForConditionalGeneration.from_pretrained(repo, **kw).eval(), WhisperProcessor.from_pretrained(repo), "whisper"
    if getattr(cfg, "is_encoder_decoder", False) and kind != "gguf":
        # T5 / BART / Marian / ...: "*ForConditionalGeneration" here means seq2seq, not causal
        from transformers import AutoModelForSeq2SeqLM
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(repo, **kw).eval()
        except ValueError:
            model = AutoModel.from_pretrained(repo, **kw).eval()
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception:  # noqa: BLE001
            tok = None
        return model, tok, "seq2seq"
    cls = AutoModelForCausalLM if causal else AutoModel
    model = cls.from_pretrained(repo, **kw).eval()
    try:
        tok = AutoTokenizer.from_pretrained(repo, **({"gguf_file": kw["gguf_file"]} if "gguf_file" in kw else {}))
    except Exception:  # noqa: BLE001
        tok = None
    return model, tok, ("causal" if causal else "encoder")
