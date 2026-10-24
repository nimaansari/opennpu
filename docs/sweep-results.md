# Universal path sweep, 2026-10-12

Ryzen AI 7 PRO 350 (XDNA2), XRT 2.25, IRON flm.GEMM backend. Each model in its own process via
`tests/sweep.py`; cosine of the model output on the NPU vs the same model on CPU fp32;
speed is one forward pass (prefill for LLMs) vs 4 CPU threads.

| Model | Kind | Params | On NPU | Cosine | CPU ms | NPU ms | Speedup |
|---|---|---|---|---|---|---|---|
| FacebookAI/roberta-base | encoder | 124.6M | 85.5M | 0.99989 | 73.5 | 36.9 | 1.99x |
| distilbert/distilbert-base-uncased | encoder | 66.4M | 42.5M | 0.99991 | 33.2 | 14.0 | 2.38x |
| albert/albert-base-v2 | encoder | 11.7M | 7.7M | 0.99943 | 89.9 | 45.9 | 1.96x |
| google/electra-small-discriminator | encoder | 13.5M | 9.4M | 0.99996 | 10.5 | 7.1 | 1.48x |
| microsoft/deberta-v3-small | encoder | 141.3M | 42.5M | 0.99984 | 72.3 | 35.1 | 2.06x |
| answerdotai/ModernBERT-base | encoder | 149.0M | 110.3M | 0.9998 | 101.6 | 50.7 | 2.0x |
| sentence-transformers/all-mpnet-base-v2 | encoder | 109.5M | 85.5M | 0.99982 | 73.3 | 30.6 | 2.39x |
| intfloat/e5-small-v2 | encoder | 33.4M | 21.4M | 0.99999 | 21.7 | 9.4 | 2.31x |
| BAAI/bge-m3 | encoder | 567.8M | 303.0M | 0.99976 | 213.8 | 96.4 | 2.22x |
| google-t5/t5-small | seq2seq | 60.5M | 60.5M | 0.99999 | 28.2 | 15.0 | 1.88x |
| google/flan-t5-base | seq2seq | 247.6M | 222.9M | 0.99999 | 135.7 | 62.9 | 2.16x |
| facebook/bart-base | seq2seq | 139.4M | 137.7M | 0.99994 | 72.7 | 33.5 | 2.17x |
| Helsinki-NLP/opus-mt-en-de | seq2seq | 74.4M | 73.8M | 1.00001 | 35.1 | 16.5 | 2.13x |
| HuggingFaceTB/SmolLM2-135M-Instruct | causal | 134.5M | 127.8M | 1.00001 | 103.8 | 51.4 | 2.02x |
| Qwen/Qwen3-0.6B | causal | 596.0M | 596.0M | 1.00021 | 414.5 | 164.3 | 2.52x |
| EleutherAI/pythia-160m | causal | 162.3M | 123.6M | 0.99886 | 101.2 | 41.7 | 2.43x |
| facebook/opt-125m | causal | 125.2M | 123.5M | 0.99993 | 83.2 | 44.9 | 1.86x |
| bigscience/bloom-560m | causal | 559.2M | 558.9M | 1.00309 | 509.5 | 154.0 | 3.31x |
| LiquidAI/LFM2-350M | causal | 354.5M | 354.4M | 0.99976 | 271.0 | 87.8 | 3.09x |
| state-spaces/mamba-130m-hf | causal | 129.1M | 123.5M | 0.99988 | 251.1 | 202.3 | 1.24x |
| allenai/OLMo-2-0425-1B | causal | 1484.9M | 1279.3M | 0.99961 | 1094.0 | 229.8 | 4.76x |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | causal | 494.0M | 488.5M | 0.99991 | 364.2 | 130.0 | 2.8x |
| microsoft/phi-1_5 | causal | 1418.3M | 1312.8M | 0.99995 | 7524.3 | 1049.6 | 7.17x |
| ibm-granite/granite-3.3-2b-instruct | causal | 2533.5M | 2533.4M | 0.99906 | 2371.2 | 469.5 | 5.05x |
| microsoft/resnet-50 | encoder | 23.5M | 23.1M | 0.99986 | 38.5 | 43.8 | 0.88x |
| facebook/convnext-tiny-224 | encoder | 27.8M | 26.3M | 0.99999 | 52.6 | 33.6 | 1.56x |
| facebook/dinov2-small | encoder | 22.1M | 21.5M | 0.99984 | 62.6 | 34.4 | 1.82x |
| microsoft/swin-tiny-patch4-window7-224 | encoder | 27.5M | 26.2M | 0.99996 | 57.5 | 38.6 | 1.49x |
| google/efficientnet-b0 | encoder | 4.0M | 0.0M | 1.0 | 12.8 | 11.2 | 1.14x |
| apple/mobilevit-small | encoder | 4.9M | 1.6M | 1.0 | 33.1 | 30.3 | 1.09x |
| nvidia/segformer-b0-finetuned-ade-512-512 | encoder | 3.3M | 2.4M | 0.99998 | 75.3 | 70.0 | 1.08x |
| facebook/deit-small-patch16-224 | encoder | 21.8M | 21.7M | 0.99971 | 46.0 | 22.0 | 2.09x |
| microsoft/beit-base-patch16-224 | encoder | 85.8M | 85.5M | 0.99678 | 180.9 | 87.6 | 2.06x |
| timm/resnet18.a1_in1k | timm | 11.7M | 11.6M | 1.0 | 17.4 | 16.7 | 1.04x |
| timm/vit_small_patch16_224.augreg_in21k_ft_in1k | timm | 22.1M | 21.9M | 0.99997 | 47.7 | 22.9 | 2.09x |
| openai/whisper-tiny | whisper | 37.8M | 36.4M | 0.99753 | 208.7 | 127.1 | 1.64x |
| facebook/wav2vec2-base-960h | encoder | 94.4M | 85.3M | 0.99433 | 296.7 | 175.9 | 1.69x |
| facebook/hubert-base-ls960 | encoder | 94.4M | 85.3M | 0.99933 | 327.2 | 182.5 | 1.79x |
| MIT/ast-finetuned-audioset-10-10-0.4593 | encoder | 86.2M | 85.1M | 0.99975 | 1301.8 | 728.7 | 1.79x |
| Xenova/all-MiniLM-L6-v2:onnx/model.onnx | onnx | 0.0M | 10.6M | 0.99996 | 14.9 | 10.2 | 1.45x |
| openai/clip-vit-base-patch32 | encoder | 151.3M | 125.7M | 1.0 | 96.9 | 41.7 | 2.32x |
| microsoft/wavlm-base | encoder | 94.4M | 85.3M | 0.99938 | 303.5 | 215.2 | 1.41x |
| google/siglip-base-patch16-224 | encoder | 203.2M | 175.2M | 1.0 | 231.0 | 100.2 | 2.31x |
| tiiuae/Falcon3-1B-Instruct | causal | 1669.4M | 1400.9M | 0.99995 | 1008.2 | 215.5 | 4.68x |
| google/vit-base-patch16-224 | encoder | 86.4M | 86.1M | 0.99803 | 162.9 | 49.6 | 3.28x |
| facebook/detr-resnet-50 | seq2seq | 41.4M | 40.9M | 0.9999 | 642.8 | 637.9 | 1.01x |

Larger LLMs (3-4B, e.g. Phi-4-mini, SmolLM3-3B) are loaded in bf16 by the server and CLI;
Phi-4-mini's greedy output on the NPU is token-identical to CPU bf16.
