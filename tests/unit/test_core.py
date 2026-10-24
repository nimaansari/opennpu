"""Fast checks that need no NPU, torch or network:  pytest tests/unit"""
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _config(monkeypatch, **env):
    for k in list(os.environ):
        if k.startswith("OPENNPU_"):
            monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import opennpu.config
    return importlib.reload(opennpu.config)


def test_package_imports_without_torch():
    # the CLI and server front end must start in an env without torch
    import opennpu
    import opennpu.config  # noqa: F401
    assert opennpu.__version__


def test_config_env_overrides(monkeypatch, tmp_path):
    c = _config(monkeypatch, OPENNPU_HOME=tmp_path / "h", OPENNPU_MODELS=tmp_path / "m",
                OPENNPU_AIRENV=tmp_path / "env.sh")
    assert c.HOME == tmp_path / "h"
    assert c.MODELS == tmp_path / "m"            # honoured even before it exists
    assert c.AIRENV == tmp_path / "env.sh"
    assert c.MOE_TEMPLATES == tmp_path / "h" / "native-moe"


def test_config_prefers_installed_toolchain(monkeypatch, tmp_path):
    env = tmp_path / "h" / "toolchain" / "airenv" / "env.sh"
    env.parent.mkdir(parents=True)
    env.write_text("")
    c = _config(monkeypatch, OPENNPU_HOME=tmp_path / "h")
    assert c.AIRENV == env


def test_no_fastflowlm_settings_left(monkeypatch, tmp_path):
    c = _config(monkeypatch, OPENNPU_HOME=tmp_path / "h")
    assert not hasattr(c, "FLM") and not hasattr(c, "FLM_MODELS")


@pytest.fixture
def server(monkeypatch, tmp_path):
    _config(monkeypatch, OPENNPU_HOME=tmp_path, OPENNPU_MODELS=tmp_path / "models")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    import opennpu.native.server as srv
    srv = importlib.reload(srv)
    srv._CAT[:] = [0.0, None]
    return srv


def test_find_any_hf_repo_is_universal(server):
    key, (kind, info) = server.find("Qwen/Qwen3-0.6B")
    assert (key, kind, info) == ("universal:Qwen/Qwen3-0.6B", "universal", ("Qwen/Qwen3-0.6B",))


def test_find_unknown_name_raises(server):
    with pytest.raises(KeyError):
        server.find("definitely-not-a-model")


def test_universal_lists_local_dirs_and_hf_cache(server, tmp_path):
    d = tmp_path / "models" / "my-finetune"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors").write_text("")
    snap = tmp_path / "hub" / "models--org--tiny" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"architectures": ["GPT2LMHeadModel"]}))
    (snap / "model.safetensors").write_text("")
    enc = tmp_path / "hub" / "models--org--bert" / "snapshots" / "abc"
    enc.mkdir(parents=True)
    (enc / "config.json").write_text(json.dumps({"architectures": ["BertModel"]}))
    (enc / "model.safetensors").write_text("")
    u = server.universal_models(claimed=set())
    assert "my-finetune" in u
    assert "org/bert" not in u                       # not a causal LM


def _handler(server, origin=None):
    h = server.H.__new__(server.H)
    h.headers = {"Origin": origin} if origin else {}
    return h


@pytest.mark.parametrize("origin,ok", [
    (None, True), ("http://127.0.0.1:11540", True), ("http://localhost:3000", True),
    ("https://evil.example", False), ("http://127.0.0.1.evil.example", False),
])
def test_only_local_origins_may_call_the_api(server, origin, ok):
    assert server.H._origin_ok(_handler(server, origin)) is ok


def test_extra_origins_can_be_allowed(server, monkeypatch):
    monkeypatch.setattr(server, "ORIGINS", {"https://my.app"})
    assert server.H._origin_ok(_handler(server, "https://my.app"))


def test_params_openai_and_ollama(server):
    assert server.H._params({"max_tokens": 50, "temperature": 0.5}) == (50, 0.5, False, None)
    assert server.H._params({"options": {"num_predict": 7, "temperature": 0.2}, "think": True}) == (7, 0.2, True, None)
    assert server.H._params({"options": {"num_predict": -1}})[0] == 4096      # Ollama: -1 = unlimited
    assert server.H._params({"chat_template_kwargs": {"enable_thinking": True}})[2] is True


def test_web_ui_assets_present():
    web = ROOT / "opennpu" / "web"
    html = (web / "index.html").read_text()
    for ref in ("app.css", "app.js"):
        assert ref in html and (web / ref).exists()
    import re
    css = (web / "app.css").read_text()
    for icon in set(re.findall(r"url\(icons/([\w-]+\.svg)\)", css)):
        assert (web / "icons" / icon).exists(), icon
    # every icon class the markup or scripts use has a CSS rule (else it renders as a solid square)
    used = set(re.findall(r"\bi-([a-z0-9-]+)", html + "".join((web / f).read_text() for f in ("app.js", "gen.js", "hub.js"))))
    defined = set(re.findall(r"\.i-([a-z0-9-]+)\s*\{", css))
    assert used <= defined, sorted(used - defined)
