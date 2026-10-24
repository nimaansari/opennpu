"""python -m opennpu.native.smoke <ckpt_dir> <decode_model> <template_dir> [prompt]"""
import sys
from transformers import AutoTokenizer
from opennpu.native.runtime import NativeDecoder

ck, dm, td = sys.argv[1:4]
prompt = sys.argv[4] if len(sys.argv) > 4 else "The capital of France is"
tok = AutoTokenizer.from_pretrained(ck)
ids = tok(prompt)["input_ids"]
d = NativeDecoder(ck, dm, td)
n = d.ATTN_MAXL - len(ids)
out, tp, td_ = d.generate(ids, n)
print("IDS", ids + out)
print(repr(prompt + tok.decode(out)))
print(f"prefill {len(ids)} tok {tp:.2f}s, decode {len(out)} tok {td_:.2f}s = {len(out)/max(td_,1e-9):.2f} tok/s")
