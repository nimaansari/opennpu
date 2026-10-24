"""Host-side logic of the batched prefill (no NPU)."""
import pytest

from opennpu.native import prefill

QWEN3_8B = dict(model_type="qwen3", hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
                head_dim=128, intermediate_size=12288, num_hidden_layers=36, vocab_size=151936)


def test_desc_and_tag():
    f = prefill.desc_fields(QWEN3_8B)
    assert (f["d"], f["dh"], f["heads"], f["kv_heads"], f["inter"]) == (4096, 128, 32, 8, 12288)
    assert f["qk_norm"] and not f["qkv_bias"]
    assert prefill.geometry_tag(f) == "d4096_h32x128_kv8_i12288_l36_v151936_qkn"


def test_down_split():
    assert prefill.down_split(12288) == 1
    assert prefill.down_split(17408) == 2          # Qwen3-14B: deeper than the device takes
    assert 17408 // 2 <= prefill.MAX_K
    with pytest.raises(ValueError):
        prefill.down_split(16384 * 9)
    tag = prefill.geometry_tag(prefill.desc_fields(dict(QWEN3_8B, intermediate_size=17408)))
    assert tag.endswith("_ds2")


@pytest.mark.parametrize("cfg,ok", [
    (QWEN3_8B, True),
    (dict(QWEN3_8B, model_type="mistral", sliding_window=None), True),
    (dict(QWEN3_8B, model_type="qwen2", sliding_window=4096, use_sliding_window=False), True),
    (dict(QWEN3_8B, model_type="qwen2", sliding_window=4096, use_sliding_window=True), False),
    (dict(QWEN3_8B, partial_rotary_factor=0.75), False),
    (dict(QWEN3_8B, rope_scaling={"rope_type": "yarn"}), False),
    (dict(QWEN3_8B, model_type="gpt_oss"), False),
])
def test_supported(cfg, ok):
    assert prefill.supported(cfg) is ok


def test_worth_it():
    p = prefill.Prefiller.__new__(prefill.Prefiller)
    p.max_len = 2048
    assert p.worth_it(500, 500)              # fresh long prompt
    assert not p.worth_it(1500, 3)           # follow-up turn: feed the few new tokens
    assert not p.worth_it(2048, 2048)        # no room left for the device's records
    assert p.worth_it(1500, 400)
