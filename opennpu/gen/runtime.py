"""Load and run generation models with their heavy network on the NPU.

Memory is checked before anything is loaded: the parameter count is read from the safetensors
headers (no weights touched), and the load is refused with a clear message if it would not fit
in the RAM that is free right now. On a laptop shared with a desktop and a chat model, running
out of memory freezes the whole machine, so refusing is better than trying.
"""
import io
import json
import os
import shutil
import struct
import time
from pathlib import Path

from .catalog import resolve

# Weight memory multipliers on top of the bf16 weights (estimates, checked by the memory test in
# tests/gen_smoke.py): auto placement keeps a CPU copy next to the NPU buffers to time both; NPU-only does not.
_AUTO_FACTOR = 2.3
_NPU_ONLY_FACTOR = 1.4
_HEADROOM = 2 << 30               # leave 2 GB for the desktop
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


class NotEnoughMemory(RuntimeError):
    pass


class GenCancelled(RuntimeError):
    """Raised from the step callback after POST /v1/gen/cancel."""


# Shared with the HTTP layer: what is running right now, and a flag to stop it. Read and set
# without the model lock, because a running generation holds that lock.
import threading as _threading
CANCEL = _threading.Event()
PROGRESS = {"task": None, "model": None, "phase": "idle", "step": 0, "steps": 0, "started": None}


def set_progress(**kw):
    PROGRESS.update(kw)


class ModelUnavailable(RuntimeError):
    """The repo does not exist, or it is gated and the user has not accepted its licence."""


def available_ram():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _safetensors_params(path):
    """Parameter count of one .safetensors file, from its JSON header only."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    total = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        cnt = 1
        for d in v["shape"]:
            cnt *= d
        total += cnt
    return total


def weights_params(folder, components=None):
    """Total parameters of the safetensors under `folder` (optionally only these subfolders).

    When a folder holds both a full and an .fp16 copy of the same weights, count it once.
    """
    folder = Path(folder)
    total = 0
    files = list(folder.rglob("*.safetensors"))
    for f in files:
        rel = f.relative_to(folder).parts
        if components is not None and (len(rel) < 2 or rel[0] not in components):
            continue
        if ".fp16." not in f.name and any(g.parent == f.parent and ".fp16." in g.name for g in files):
            continue
        total += _safetensors_params(f)
    return total


def _repo_files(repo):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    if repo.count("/") != 1:
        raise ModelUnavailable(f"{repo!r} is not a Hugging Face repo id (owner/name) or a local folder")
    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except GatedRepoError:
        raise ModelUnavailable(f"{repo} is gated: accept its licence on https://huggingface.co/{repo} "
                               f"and log in with `hf auth login`")
    except RepositoryNotFoundError:
        raise ModelUnavailable(f"{repo} was not found on Hugging Face (private repos need `hf auth login`)")
    except Exception:   # noqa: BLE001 - offline: fall back to whatever is cached
        return None
    return {s.rfilename: (s.size or 0) for s in info.siblings}


def _check_disk(nbytes, where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(where).free
    if nbytes and free < nbytes + (2 << 30):
        raise NotEnoughMemory(f"not enough disk space: the download is {nbytes / 2**30:.1f} GB and "
                              f"only {free / 2**30:.1f} GB is free in {where}")


def _hf_cache():
    return Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / (
        "" if os.environ.get("HF_HUB_CACHE") else "hub")


def download_size(repo, files, variant):
    """Bytes the pipeline download needs: only the component folders in model_index.json."""
    from huggingface_hub import hf_hub_download
    try:
        index = json.loads(Path(hf_hub_download(repo, "model_index.json")).read_text())
        comps = {k for k, v in index.items() if isinstance(v, list) and v and v[0]}
    except Exception:   # noqa: BLE001
        comps = None
    total = 0
    for f, size in files.items():
        parts = f.split("/")
        if comps is not None and (len(parts) < 2 or parts[0] not in comps):
            continue
        if not f.endswith((".safetensors", ".bin", ".json", ".txt", ".model")):
            continue
        if f.endswith((".safetensors", ".bin")):
            is_var = ".fp16." in f
            if variant and not is_var and any(g.startswith(parts[0] + "/") and ".fp16." in g for g in files):
                continue                     # this component has an fp16 file; skip the full one
            if not variant and is_var:
                continue
            if f.endswith(".bin") and any(g.startswith(parts[0] + "/") and g.endswith(".safetensors") for g in files):
                continue                     # safetensors preferred over .bin
        total += size
    return total


def fetch_diffusers(repo):
    """Download only what the pipeline needs (fp16 variant when the repo has one)."""
    if Path(repo).is_dir():
        return Path(repo), None
    files = _repo_files(repo)
    variant = "fp16" if files and any(".fp16." in f for f in files) else None
    if files:
        _check_disk(download_size(repo, files, variant), _hf_cache())
    from diffusers import DiffusionPipeline
    with _gated(repo):
        path = DiffusionPipeline.download(repo, variant=variant)
    return Path(path), variant


class _gated:
    """Turn a gated-repo download error into ModelUnavailable with the fix."""

    def __init__(self, repo):
        self.repo = repo

    def __enter__(self):
        return self

    def __exit__(self, et, e, tb):
        from huggingface_hub.errors import GatedRepoError
        if et is not None and (issubclass(et, GatedRepoError) or isinstance(e.__cause__, GatedRepoError)):
            raise ModelUnavailable(f"{self.repo} is gated: accept its licence on "
                                   f"https://huggingface.co/{self.repo} and log in with `hf auth login`") from e
        return False


def fetch_transformers(repo):
    if Path(repo).is_dir():
        return Path(repo)
    files = _repo_files(repo)
    if files:
        _check_disk(sum(s for f, s in files.items() if f.endswith((".safetensors", ".bin", ".json", ".model", ".txt"))),
                    _hf_cache())
    from huggingface_hub import snapshot_download
    has_st = files is None or any(f.endswith(".safetensors") for f in files)
    allow = ["*.json", "*.txt", "*.model", "*.safetensors", "*.npz", "*.npy"] + ([] if has_st else ["*.bin"])
    with _gated(repo):
        return Path(snapshot_download(repo, allow_patterns=allow))


def plan_memory(params, label, offloaded=None):
    """'auto' (CPU copy + NPU, picks the faster per layer) or 'npu' (lean), or raise.

    `offloaded` is the part of `params` that gets NPU buffers (default: all of it); the rest
    (text encoders, VAE) is loaded once in bf16.
    """
    off = params if offloaded is None else offloaded
    base = (params - off) * 2                  # bf16, CPU only
    avail = available_ram()
    if avail is None:
        return "auto"
    if base + off * 2 * _AUTO_FACTOR + _HEADROOM <= avail:
        return "auto"
    if base + off * 2 * _NPU_ONLY_FACTOR + _HEADROOM <= avail:
        return "npu"
    need = (base + off * 2 * _NPU_ONLY_FACTOR + _HEADROOM) / 2**30
    raise NotEnoughMemory(
        f"{label} needs about {need:.0f} GB of free RAM ({params / 1e9:.1f}B parameters) but only "
        f"{avail / 2**30:.0f} GB is free. Close other apps, unload the chat model "
        f"(POST /unload), or pick a smaller model (`opennpu gen list`).")


def _threads():
    import torch
    n = int(os.environ.get("OPENNPU_GEN_THREADS", os.environ.get("OPENNPU_CPU_THREADS", "6")))
    torch.set_num_threads(max(1, n))


def _dtype_anchors(module, before):
    """Model code often reads `next(submodule.parameters()).dtype` (Wan's time embedder does).
    Offloaded layers keep their weights in NPU buffers, so a submodule made only of them has no
    parameters left; give it an empty one of the original dtype so that code keeps working."""
    import torch
    n = 0
    for qn, m in module.named_modules():
        dt = before.get(qn)
        if dt is not None and next(m.parameters(), None) is None:
            m.register_parameter("_opennpu_dtype", torch.nn.Parameter(torch.empty(0, dtype=dt), requires_grad=False))
            n += 1
    return n


def _offload(module, mode, log):
    from opennpu.offload import offload
    before = {}
    for qn, m in module.named_modules():
        p = next(m.parameters(), None)
        if p is not None:
            before[qn] = p.dtype
    try:
        rep = offload(module, min_tokens=16 if mode == "auto" else 1, auto=(mode == "auto"))
        _dtype_anchors(module, before)
    except Exception as e:   # noqa: BLE001 - no NPU (or driver missing): run on the CPU, say so
        log(f"NPU offload unavailable ({type(e).__name__}: {e}); running on the CPU")
        return None
    log(str(rep))
    return rep


def _placement_file(name, comp):
    from opennpu import config
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{name}-{comp}")
    return config.CACHE / "gen-placement" / f"{safe}.json"


def _placed_layers(module):
    import sys
    O = sys.modules["opennpu.offload"]
    for qn, m in module.named_modules():
        if isinstance(m, O.AutoPlace) or (isinstance(m, O.NPULinear) and m.fallback is not None):
            yield qn, m


def save_placement(module, name, comp):
    """Per-layer CPU/NPU decisions, so the timing warm-up happens once per model, not per start."""
    data = {qn: [[list(k) if isinstance(k, tuple) else k, v] for k, v in m.choice.items()]
            for qn, m in _placed_layers(module) if m.choice}
    if not data:
        return 0
    f = _placement_file(name, comp)
    f.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(f.read_text()) if f.exists() else {}
    for qn, items in data.items():
        merged = {json.dumps(k): v for k, v in old.get(qn, [])}
        merged.update({json.dumps(k): v for k, v in items})
        old[qn] = [[json.loads(k), v] for k, v in merged.items()]
    f.write_text(json.dumps(old))
    return len(data)


def load_placement(module, name, comp):
    f = _placement_file(name, comp)
    if not f.exists():
        return 0
    try:
        data = json.loads(f.read_text())
    except ValueError:
        return 0
    n = 0
    for qn, m in _placed_layers(module):
        for k, v in data.get(qn, []):
            m.choice[tuple(k) if isinstance(k, list) else k] = v
            n += 1
    return n


def _say(msg):
    print(msg, flush=True)


class _Base:
    task = None

    def close(self):
        self.pipe = None
        import gc
        gc.collect()

    def chat(self, *a, **k):      # the chat server calls chat() on whatever is loaded
        raise ValueError(f"this is a {self.task} generation model; it has no chat")

    def embed(self, *a, **k):
        raise ValueError(f"this is a {self.task} generation model; it has no embeddings")


class DiffusionEngine(_Base):
    """Any diffusers text-to-image / text-to-video pipeline; its denoiser runs on the NPU."""

    def __init__(self, model, task=None, log=_say, npu=True, offload_all=False):
        t, repo, self.defaults, self.name = resolve(model, task)
        if t not in ("image", "video"):
            raise ValueError(f"{model} is a {t} model, not image/video")
        self.task = t
        import torch
        _threads()
        path, variant = fetch_diffusers(repo)
        index = json.loads((path / "model_index.json").read_text())
        comps = [k for k, v in index.items() if isinstance(v, list) and v and v[0]]
        params = weights_params(path, comps)
        targets = ["transformer", "unet", "prior", "decoder"]
        if offload_all:
            targets += ["text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
        off = weights_params(path, [c for c in comps if c in targets])
        mode = plan_memory(params, self.name, off) if npu else "cpu"
        log(f"{self.name}: {params / 1e9:.2f}B parameters ({off / 1e9:.2f}B on the NPU), placement={mode}")
        if t == "video":
            from diffusers import DiffusionPipeline as Auto
        else:
            from diffusers import AutoPipelineForText2Image as Auto
        kw = dict(torch_dtype=torch.bfloat16)
        if variant:
            kw["variant"] = variant
        try:
            self.pipe = Auto.from_pretrained(path, **kw)
        except ValueError:
            from diffusers import DiffusionPipeline
            self.pipe = DiffusionPipeline.from_pretrained(path, **kw)
        self.pipe.set_progress_bar_config(disable=True)
        self.report = []
        if npu:
            for n in targets:
                m = getattr(self.pipe, n, None)
                if isinstance(m, torch.nn.Module):
                    r = _offload(m, mode, log)
                    if r:
                        self.report.append((n, r))
                        if mode == "auto" and load_placement(m, self.name, n):
                            log(f"{n}: reusing saved CPU/NPU placement (no warm-up needed)")
        self.torch = torch

    def generate(self, prompt, negative=None, steps=None, guidance=None, width=None, height=None,
                 seed=None, frames=None, fps=None, on_step=None):
        d = self.defaults
        steps = int(steps or d.get("steps", 25))
        kw = dict(prompt=prompt, num_inference_steps=steps)
        g = guidance if guidance is not None else d.get("guidance")
        if g is not None:
            kw["guidance_scale"] = float(g)
        if negative and (g or 0) > 1:
            kw["negative_prompt"] = negative
        # Image pipelines need multiples of 8; Wan needs 16 and LTX-Video 32, so video uses 32.
        mult = 32 if self.task == "video" else 8
        for k, v in (("width", width or d.get("width")), ("height", height or d.get("height"))):
            if v:
                kw[k] = max(mult, int(v) // mult * mult)
        if self.task == "video":
            n = int(frames or d.get("frames", 17))
            kw["num_frames"] = (max(n, 5) - 1) // 8 * 8 + 1 if "ltx" in self.name.lower() else (max(n, 5) - 1) // 4 * 4 + 1
        if seed is not None:
            kw["generator"] = self.torch.Generator("cpu").manual_seed(int(seed))

        def cb(pipe, i, t, kwargs):
            # some schedulers run more timesteps than num_inference_steps
            total = max(steps, getattr(pipe, "num_timesteps", 0) or 0)
            set_progress(phase="generating", step=i + 1, steps=total)
            if on_step:
                on_step(i + 1, total)
            if CANCEL.is_set():
                raise GenCancelled("generation cancelled")
            return kwargs
        kw["callback_on_step_end"] = cb
        set_progress(phase="generating", step=0, steps=steps)
        if CANCEL.is_set():
            raise GenCancelled("generation cancelled")
        t0 = time.time()
        with self.torch.no_grad():
            out = self.pipe(**kw)
        dt = time.time() - t0
        for n, _ in self.report:
            save_placement(getattr(self.pipe, n), self.name, n)
        st = {"seconds": round(dt, 2), "steps": steps, "s_per_step": round(dt / max(steps, 1), 2)}
        if self.task == "image":
            return out.images, st
        return out.frames[0], dict(st, fps=int(fps or d.get("fps", 16)))


class SpeechEngine(_Base):
    """Any transformers text-to-speech model (VITS/MMS, Bark, SpeechT5, ...) with the model on the NPU."""

    task = "speech"

    def __init__(self, model, log=_say, npu=True):
        t, repo, self.defaults, self.name = resolve(model, "speech")
        if t != "speech":
            raise ValueError(f"{model} is a {t} model, not speech")
        import torch
        from transformers import pipeline
        _threads()
        path = fetch_transformers(repo)
        params = weights_params(path)
        mode = plan_memory(params, self.name) if npu else "cpu"
        log(f"{self.name}: {params / 1e6:.0f}M parameters, placement={mode}")
        self.pipe = pipeline("text-to-speech", model=str(path), device="cpu")
        self.path = str(path)
        self.speaker = None
        if "speecht5" in repo.lower():
            self.speaker = self._speecht5_voice(torch)
        self.report = []
        if npu:
            r = _offload(self.pipe.model, mode, log)
            if r and os.environ.get("OPENNPU_GEN_CONV1D") == "1":
                # Off by default: TTS convs are small, so the NPU measured 3.6x slower than the CPU
                # on MMS/VITS, and bf16 rounding shifts VITS's predicted durations.
                from .layers import offload_conv1d
                n, np_ = offload_conv1d(self.pipe.model, mode)
                if n:
                    r.offloaded += [("conv1d", None)] * n
                    r.npu_params += np_
                    log(f"  + {n} Conv1d layers ({np_ / 1e6:.1f}M params) on the NPU")
            if r:
                self.report.append(("model", r))
                if mode == "auto":
                    load_placement(self.pipe.model, self.name, "model")
        self.torch = torch

    @staticmethod
    def _speecht5_voice(torch):
        # A fixed neutral voice so SpeechT5 works without the external x-vector dataset.
        g = torch.Generator().manual_seed(7306)
        v = torch.randn(1, 512, generator=g) * 0.05
        return torch.nn.functional.normalize(v, dim=-1) * 3.0

    def generate(self, text, voice=None, seed=None):
        kw = {}
        if self.speaker is not None:
            kw["forward_params"] = {"speaker_embeddings": self.speaker}
        v = voice or self.defaults.get("voice")
        if seed is not None:
            self.torch.manual_seed(int(seed))
        set_progress(phase="generating", step=0, steps=1)
        if CANCEL.is_set():                    # TTS has no step hook: cancel only before it starts
            raise GenCancelled("generation cancelled")
        import numpy as np
        t0 = time.time()
        with self.torch.no_grad():
            if "bark" in self.name.lower():
                # Bark voice presets are tensors loaded by its processor, not a string argument
                from transformers import AutoProcessor
                proc = getattr(self, "_proc", None) or AutoProcessor.from_pretrained(self.path)
                self._proc = proc
                inputs = proc(text, voice_preset=v) if v else proc(text)
                audio = self.pipe.model.generate(**inputs, do_sample=True).cpu().numpy()
                sr = int(self.pipe.model.generation_config.sample_rate)
            else:
                out = self.pipe(text, **kw)
                audio, sr = out["audio"], int(out["sampling_rate"])
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        dt = time.time() - t0
        if self.report:
            save_placement(self.pipe.model, self.name, "model")
        dur = len(audio) / sr
        return (audio, sr), {"seconds": round(dt, 2), "audio_seconds": round(dur, 2),
                             "realtime_factor": round(dur / max(dt, 1e-9), 2)}


class TranscribeEngine(_Base):
    """Whisper (or any transformers speech-recognition model); the encoder runs on the NPU.

    Decoding is token by token, so those small GEMMs stay on the CPU by AutoPlace's choice."""

    task = "transcribe"
    SR = 16000

    def __init__(self, model, log=_say, npu=True):
        t, repo, self.defaults, self.name = resolve(model, "transcribe")
        if t != "transcribe":
            raise ValueError(f"{model} is a {t} model, not speech-to-text")
        import torch
        from transformers import pipeline
        _threads()
        path = fetch_transformers(repo)
        params = weights_params(path)
        mode = plan_memory(params, self.name) if npu else "cpu"
        log(f"{self.name}: {params / 1e6:.0f}M parameters, placement={mode}")
        self.pipe = pipeline("automatic-speech-recognition", model=str(path), device="cpu",
                             dtype=torch.float32 if params < 1e9 else torch.bfloat16)
        self.report = []
        if npu:
            r = _offload(self.pipe.model, mode, log)
            if r:
                self.report.append(("model", r))
                if mode == "auto":
                    load_placement(self.pipe.model, self.name, "model")
        self.torch = torch

    def _language(self, audio):
        """Detected language code (e.g. "en") from the first 30 s, or None."""
        m = self.pipe.model
        if not hasattr(m, "detect_language"):
            return None
        fe = self.pipe.feature_extractor
        feats = fe(audio[: 30 * self.SR], sampling_rate=self.SR, return_tensors="pt").input_features
        with self.torch.no_grad():
            ids = m.detect_language(feats.to(m.dtype))
        tok = self.pipe.tokenizer.convert_ids_to_tokens(int(ids[0]))
        return tok.strip("<|>") if tok else None

    def transcribe(self, audio, language=None, prompt=None, want_language=False):
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        set_progress(phase="generating", step=0, steps=1)
        if CANCEL.is_set():
            raise GenCancelled("generation cancelled")
        gk = {"task": "transcribe"}
        if language:
            gk["language"] = language
        if prompt:
            gk["prompt_ids"] = self.torch.as_tensor(self.pipe.tokenizer.get_prompt_ids(prompt))
        t0 = time.time()
        with self.torch.no_grad():
            out = self.pipe({"raw": audio, "sampling_rate": self.SR}, chunk_length_s=30, batch_size=4,
                            generate_kwargs=gk)
            lang = language or (self._language(audio) if want_language else None)
        if self.report:
            save_placement(self.pipe.model, self.name, "model")
        text = out["text"].strip()
        if prompt and text.startswith(prompt.strip()):          # some versions echo the prompt
            text = text[len(prompt.strip()):].strip()
        dur = len(audio) / self.SR
        dt = time.time() - t0
        return {"text": text, "language": lang, "duration": round(dur, 2)}, {
            "seconds": round(dt, 2), "audio_seconds": round(dur, 2), "realtime_factor": round(dur / max(dt, 1e-9), 2)}


def decode_audio(data, sr=16000, max_seconds=None):
    """Any container ffmpeg reads (webm/opus, ogg, wav, mp3, m4a, flac) -> mono float32 at `sr`."""
    import subprocess
    import numpy as np
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(sr)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    r = subprocess.run(cmd + ["-f", "f32le", "pipe:1"], input=data, capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        raise ValueError("could not decode the audio: " + r.stderr.decode(errors="replace").strip()[:200])
    return np.frombuffer(r.stdout, dtype=np.float32)


def load(model, task=None, **kw):
    t = resolve(model, task)[0]
    if t == "transcribe":
        return TranscribeEngine(model, **kw)
    return SpeechEngine(model, **kw) if t == "speech" else DiffusionEngine(model, task=t, **kw)


# ------------------------------------------------------------------ encoders for files / HTTP
def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def wav_bytes(audio, sr):
    import numpy as np
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm


def mp4_bytes(frames, fps):
    import tempfile
    from diffusers.utils import export_to_video
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "video.mp4"
        export_to_video(frames, str(out), fps=fps)
        return out.read_bytes()
