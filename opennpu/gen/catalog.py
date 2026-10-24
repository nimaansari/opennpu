"""Curated generation models (the ones we have run end to end) plus any Hugging Face repo.

Curated entries give short names (`sd-turbo`, `wan-1.3b`, ...) and sensible defaults. Any other
diffusers or text-to-speech repo id works too: `image:<repo>`, `video:<repo>`, `speech:<repo>`, `transcribe:<repo>`, or
a bare repo id with `--task`.
"""

TASKS = ("image", "video", "speech", "transcribe")

# name -> dict(task, repo, defaults, note). `ram_gb` is an estimate of peak RAM with the
# denoiser on the NPU (bf16), shown before download; the real check reads the weight headers.
CATALOG = {
    # text-to-image
    "sd-turbo": dict(task="image", repo="stabilityai/sd-turbo", ram_gb=4,
                     defaults=dict(steps=1, guidance=0.0, width=512, height=512),
                     note="Stable Diffusion 2.1 Turbo: 1 step, 512 px, fastest"),
    "sdxl-turbo": dict(task="image", repo="stabilityai/sdxl-turbo", ram_gb=10,
                       defaults=dict(steps=2, guidance=0.0, width=512, height=512),
                       note="SDXL Turbo: 1-4 steps, better detail"),
    "sd-1.5": dict(task="image", repo="stable-diffusion-v1-5/stable-diffusion-v1-5", ram_gb=5,
                   defaults=dict(steps=25, guidance=7.5, width=512, height=512),
                   note="the classic; huge LoRA / fine-tune ecosystem"),
    "sdxl": dict(task="image", repo="stabilityai/stable-diffusion-xl-base-1.0", ram_gb=12,
                 defaults=dict(steps=25, guidance=6.0, width=1024, height=1024),
                 note="SDXL base, 1024 px"),
    "sana-600m": dict(task="image", repo="Efficient-Large-Model/Sana_600M_512px_diffusers", ram_gb=8,
                      defaults=dict(steps=20, guidance=4.5, width=512, height=512),
                      note="Sana 600M: linear-attention DiT, NPU-friendly"),
    "sd3.5-medium": dict(task="image", repo="stabilityai/stable-diffusion-3.5-medium", ram_gb=22,
                         defaults=dict(steps=28, guidance=4.5, width=1024, height=1024),
                         note="SD 3.5 Medium (gated: accept the licence on Hugging Face)"),
    "flux-schnell": dict(task="image", repo="black-forest-labs/FLUX.1-schnell", ram_gb=40,
                         defaults=dict(steps=4, guidance=0.0, width=1024, height=1024),
                         note="FLUX.1 schnell 12B: best quality, needs ~40 GB RAM"),
    # text-to-video
    "wan-1.3b": dict(task="video", repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", ram_gb=20,
                     defaults=dict(steps=30, guidance=5.0, width=480, height=320, frames=17, fps=16),
                     note="Wan 2.1 1.3B text-to-video"),
    "ltx-video": dict(task="video", repo="Lightricks/LTX-Video", ram_gb=22,
                      defaults=dict(steps=30, guidance=3.0, width=512, height=320, frames=25, fps=24),
                      note="LTX-Video 2B: fastest video; use long, descriptive prompts"),
    # text-to-speech
    "mms-tts-eng": dict(task="speech", repo="facebook/mms-tts-eng", ram_gb=1,
                        defaults={}, note="Meta MMS (VITS) English: fast. 1100+ languages: mms-tts-<iso>"),
    "bark-small": dict(task="speech", repo="suno/bark-small", ram_gb=4,
                       defaults=dict(voice="v2/en_speaker_6"), note="Bark small: expressive, slower"),
    "speecht5": dict(task="speech", repo="microsoft/speecht5_tts", ram_gb=2,
                     defaults={}, note="SpeechT5 with a built-in neutral voice"),
    # speech-to-text (Whisper; encoder on the NPU)
    "whisper-small": dict(task="transcribe", repo="openai/whisper-small", ram_gb=2,
                          defaults={}, note="Whisper small: good accuracy, 99 languages (default)"),
    "whisper-base": dict(task="transcribe", repo="openai/whisper-base", ram_gb=1,
                         defaults={}, note="Whisper base: faster, less accurate"),
    "whisper-tiny": dict(task="transcribe", repo="openai/whisper-tiny", ram_gb=1,
                         defaults={}, note="Whisper tiny: fastest, for quick voice commands"),
    "whisper-large-v3-turbo": dict(task="transcribe", repo="openai/whisper-large-v3-turbo", ram_gb=5,
                                   defaults={}, note="Whisper large v3 turbo: best accuracy"),
}


def by_task(task):
    return {k: v for k, v in CATALOG.items() if v["task"] == task}


def resolve(model, task=None):
    """'sd-turbo' | 'image:<repo>' | '<repo>' (+task) -> (task, repo, defaults, name)."""
    if model in CATALOG:
        e = CATALOG[model]
        return e["task"], e["repo"], dict(e["defaults"]), model
    if ":" in model and model.split(":", 1)[0] in TASKS:
        t, repo = model.split(":", 1)
        return t, repo, _defaults_for_repo(repo), model
    for name, e in CATALOG.items():                # a curated repo given by its id
        if e["repo"] == model:
            return e["task"], model, dict(e["defaults"]), name
    if task in TASKS:
        return task, model, _defaults_for_repo(model), f"{task}:{model}"
    raise KeyError(f"unknown generation model {model!r}: use a name from `opennpu gen list` "
                   f"or image:/video:/speech:<huggingface repo>")


def _defaults_for_repo(repo):
    r = repo.lower()
    if "turbo" in r or "schnell" in r or "lightning" in r:
        return dict(steps=4, guidance=0.0)
    return {}
