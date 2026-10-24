"""Long-context check of the native NPU decode against the CPU reference.

    python -m opennpu.native.verify <ckpt_dir> <decode_model> <template_dir> [n_new]

Generates greedily on the NPU after a ~250-token prompt (so attention runs over many
16-key blocks), then teacher-forces the same sequence through the CPU reference and
reports, for every generated position, whether the NPU token is the reference top-1 /
inside its top-5. Exit code 1 if top-5 agreement < 90%.
"""
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from opennpu.native.reference import forward
from opennpu.native.runtime import NativeDecoder

TEXT = (
    "The history of the Roman Empire spans more than a thousand years, from the founding of the "
    "city in the eighth century BC to the fall of Constantinople in 1453. During the Republic, "
    "Rome expanded across the Italian peninsula and then around the Mediterranean, defeating "
    "Carthage in the Punic Wars and absorbing the Hellenistic kingdoms of the east. Julius Caesar "
    "crossed the Rubicon in 49 BC, and after his assassination his heir Octavian defeated Mark "
    "Antony and Cleopatra at Actium. Octavian took the name Augustus and became the first emperor. "
    "The following two centuries, known as the Pax Romana, saw relative peace, the construction of "
    "roads, aqueducts and cities, and the spread of Latin and Roman law. In the third century the "
    "empire nearly collapsed under civil wars, plague and invasions, before Diocletian divided "
    "administration between east and west. Constantine later founded a new capital on the Bosporus. "
    "The western half fell in 476 when Odoacer deposed the last emperor, Romulus Augustulus. "
    "Summarize the three most important turning points in this history, and explain why each mattered:"
)


def main():
    ck, dm, td = sys.argv[1:4]
    n_new = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    tok = AutoTokenizer.from_pretrained(ck)
    ids = tok(TEXT)["input_ids"]
    d = NativeDecoder(ck, dm, td)
    out, tp, tdec = d.generate(ids, n_new)
    print(f"[verify] prompt {len(ids)} tok, NPU prefill {tp:.1f}s, decode {len(out)} tok "
          f"{len(out) / tdec:.2f} tok/s")
    print("[verify] NPU:", repr(tok.decode(out)))
    del d
    torch.set_num_threads(8)
    lg = forward(ck, ids + out[:-1])[len(ids) - 1:]
    top1 = lg.argmax(-1)
    top5 = np.argsort(-lg, -1)[:, :5]
    a1 = np.mean([t == r for t, r in zip(out, top1)])
    a5 = np.mean([t in r for t, r in zip(out, top5)])
    print(f"[verify] vs CPU reference: top-1 {a1 * 100:.0f}%  top-5 {a5 * 100:.0f}%  over {len(out)} tokens")
    sys.exit(0 if a5 >= 0.9 else 1)


if __name__ == "__main__":
    main()
