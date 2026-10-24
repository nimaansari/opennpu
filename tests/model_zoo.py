"""Run a zoo of different model families through OpenNPU and check parity vs CPU."""
import copy
import sys
import torch
sys.path.insert(0, ".")
from opennpu import offload, parity
torch.set_num_threads(4)
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModelForImageClassification, WhisperForConditionalGeneration, WhisperProcessor

def run(name, build, inputs_fn):
    ref = build().eval(); npu = copy.deepcopy(ref)
    rep = offload(npu)
    inp = inputs_fn()
    with torch.no_grad():
        for _ in range(3): npu(**inp)          # let per-layer auto-placement settle
        rel, cos, rt, yt = parity(lambda: ref(**inp), lambda: npu(**inp))
    print(f"{name:28s} {rep.npu_params/1e6:6.0f}M/{rep.total_params/1e6:5.0f}M on NPU  cos={cos:.4f} rel={rel:.3f}  cpu {rt:7.1f}ms  npu {yt:7.1f}ms", flush=True)
    return ref, npu

text = "OpenNPU runs any model on AMD Ryzen AI NPUs under Linux. " * 6
# 1. BERT-family encoder (embeddings)
tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
run("bge-base (BERT encoder)", lambda: AutoModel.from_pretrained("BAAI/bge-base-en-v1.5"), lambda: dict(tok(text, return_tensors="pt")))
# 2. Vision transformer
proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
img = torch.rand(3, 224, 224)
run("ViT-base (vision)", lambda: AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224"), lambda: dict(proc(images=img, return_tensors="pt", do_rescale=False)))
# 3. Whisper (speech) encoder+decoder forward
wp = WhisperProcessor.from_pretrained("openai/whisper-small")
feats = wp(torch.randn(16000*10).numpy()*0.05, sampling_rate=16000, return_tensors="pt").input_features
run("whisper-small (speech)", lambda: WhisperForConditionalGeneration.from_pretrained("openai/whisper-small"), lambda: dict(input_features=feats, decoder_input_ids=torch.tensor([[50258, 50259, 50359, 50363]])))
# 4. Decoder LLM prefill
lt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref, npu = run("Qwen2.5-0.5B (LLM prefill)", lambda: AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32), lambda: dict(lt(text, return_tensors="pt")))
msgs = lt.apply_chat_template([{"role":"user","content":"Name three planets."}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
with torch.no_grad():
    a = ref.generate(**msgs, max_new_tokens=20, do_sample=False); b = npu.generate(**msgs, max_new_tokens=20, do_sample=False)
print("   CPU gen:", repr(lt.decode(a[0, msgs['input_ids'].shape[1]:], skip_special_tokens=True)))
print("   NPU gen:", repr(lt.decode(b[0, msgs['input_ids'].shape[1]:], skip_special_tokens=True)))
# 5. GPT-2 (HF Conv1D layers)
gt = AutoTokenizer.from_pretrained("gpt2")
run("gpt2 (Conv1D layers)", lambda: AutoModelForCausalLM.from_pretrained("gpt2"), lambda: dict(gt(text, return_tensors="pt")))
