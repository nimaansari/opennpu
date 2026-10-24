"""Image, video and speech generation on the NPU.

Any Hugging Face diffusers pipeline (text-to-image, text-to-video) or transformers
text-to-speech model runs here: the heavy network (UNet / diffusion transformer / TTS model)
goes through `opennpu.offload`, so its linear and conv layers run on the NPU, and the rest
stays on the CPU.

This package imports without torch; the engines import it lazily.
"""
from .catalog import CATALOG, TASKS, by_task, resolve  # noqa: F401
