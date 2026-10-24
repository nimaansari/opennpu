import copy
import sys
import time
import torch
sys.path.insert(0, ".")
from opennpu import offload, parity, get_engine
torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "Qwen/Qwen2.5-0.5B-Instruct"
lt = AutoTokenizer.from_pretrained(mid)
ref = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval(); npu = copy.deepcopy(ref)
rep = offload(npu); print(rep, "| precision:", get_engine().precision)
text = "OpenNPU runs any model on AMD Ryzen AI NPUs under Linux. " * 6
inp = dict(lt(text, return_tensors="pt"))
with torch.no_grad():
    rel, cos, rt, yt = parity(lambda: ref(**inp), lambda: npu(**inp))
    print(f"prefill {inp['input_ids'].shape[1]} tok: cos={cos:.4f} rel={rel:.3f} cpu {rt:.0f}ms npu {yt:.0f}ms")
    msgs = lt.apply_chat_template([{"role":"user","content":"Name three planets."}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    for name, m in (("CPU", ref), ("NPU", npu)):
        t = time.perf_counter(); o = m.generate(**msgs, max_new_tokens=32, do_sample=False); dt = time.perf_counter() - t
        n = o.shape[1] - msgs["input_ids"].shape[1]
        print(f"{name} {n/dt:5.1f} tok/s:", repr(lt.decode(o[0, msgs['input_ids'].shape[1]:], skip_special_tokens=True)))
