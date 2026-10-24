"""`opennpu image | video | speak | gen list` — generation from the terminal."""
import sys
import time
from pathlib import Path

from .catalog import CATALOG, TASKS, by_task


def add_parsers(sub):
    common = dict(help="curated name (`opennpu gen list`) or any Hugging Face repo id")
    i = sub.add_parser("image", help="text-to-image (any diffusers model) on the NPU")
    i.add_argument("prompt"); i.add_argument("-m", "--model", default="sd-turbo", **common)
    i.add_argument("-o", "--output", help="PNG path (default: image-<time>.png)")
    i.add_argument("-n", "--count", type=int, default=1)
    v = sub.add_parser("video", help="text-to-video (any diffusers model) on the NPU")
    v.add_argument("prompt"); v.add_argument("-m", "--model", default="wan-1.3b", **common)
    v.add_argument("-o", "--output", help="MP4 path (default: video-<time>.mp4)")
    v.add_argument("--frames", type=int); v.add_argument("--fps", type=int)
    for p in (i, v):
        p.add_argument("--steps", type=int); p.add_argument("--guidance", type=float)
        p.add_argument("--width", type=int); p.add_argument("--height", type=int)
        p.add_argument("--seed", type=int); p.add_argument("--negative", help="negative prompt")
        p.add_argument("--offload-all", action="store_true",
                       help="also put text encoders and the VAE on the NPU (more RAM)")
    s = sub.add_parser("speak", help="text-to-speech (any transformers TTS model) on the NPU")
    s.add_argument("text"); s.add_argument("-m", "--model", default="mms-tts-eng", **common)
    s.add_argument("-o", "--output", help="WAV path (default: speech-<time>.wav)")
    s.add_argument("--voice", help="voice preset (Bark: v2/en_speaker_0..9)")
    s.add_argument("--seed", type=int)
    for p in (i, v, s):
        p.add_argument("--cpu", action="store_true", help="run on the CPU only (for comparison)")
    g = sub.add_parser("gen", help="generation models: list")
    g.add_argument("action", choices=["list"])
    g.add_argument("--task", choices=TASKS)


def _out(path, stem, ext):
    if path:
        return Path(path)
    return Path(f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}.{ext}")


def _progress(i, n):
    print(f"\r  step {i}/{n}", end="" if i < n else "\n", file=sys.stderr, flush=True)


def run(a):
    if a.cmd == "gen":
        for task in ([a.task] if a.task else TASKS):
            print(f"{task}:")
            for name, e in by_task(task).items():
                print(f"  {name:<16} ~{e['ram_gb']:>2} GB  {e['repo']:<52} {e['note']}")
        print("\nAny other Hugging Face repo works too: -m image:<repo>, video:<repo>, speech:<repo>")
        return 0
    from .runtime import ModelUnavailable, NotEnoughMemory, mp4_bytes, png_bytes, wav_bytes
    log = lambda m: print(f"[opennpu] {m}", file=sys.stderr)  # noqa: E731
    try:
        if a.cmd == "speak":
            from .runtime import SpeechEngine
            eng = SpeechEngine(a.model, log=log, npu=not a.cpu)
            (audio, sr), st = eng.generate(a.text, voice=a.voice, seed=a.seed)
            out = _out(a.output, "speech", "wav")
            out.write_bytes(wav_bytes(audio, sr))
            log(f"{st['audio_seconds']}s of audio in {st['seconds']}s ({st['realtime_factor']}x real time)")
        else:
            from .runtime import DiffusionEngine
            eng = DiffusionEngine(a.model, task=a.cmd, log=log, npu=not a.cpu, offload_all=a.offload_all)
            kw = dict(negative=a.negative, steps=a.steps, guidance=a.guidance, width=a.width,
                      height=a.height, seed=a.seed, on_step=_progress)
            if a.cmd == "video":
                frames, st = eng.generate(a.prompt, frames=a.frames, fps=a.fps, **kw)
                out = _out(a.output, "video", "mp4")
                out.write_bytes(mp4_bytes(frames, st["fps"]))
                log(f"{len(frames)} frames in {st['seconds']}s")
            else:
                outs = []
                for k in range(a.count):
                    if a.seed is not None:
                        kw["seed"] = a.seed + k
                    images, st = eng.generate(a.prompt, **kw)
                    o = _out(a.output, "image", "png")
                    if a.count > 1:
                        o = o.with_name(f"{o.stem}-{k + 1}{o.suffix}")
                    o.write_bytes(png_bytes(images[0]))
                    outs.append(o)
                    log(f"image {k + 1}/{a.count} in {st['seconds']}s")
                out = outs[-1] if len(outs) == 1 else Path(", ".join(map(str, outs)))
    except (NotEnoughMemory, ModelUnavailable) as e:
        log(str(e))
        return 2
    except KeyError as e:
        log(e.args[0])
        return 2
    print(out)
    return 0
