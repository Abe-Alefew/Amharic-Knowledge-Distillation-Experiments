"""Config-driven teacher loading. Switch teachers by pointing at a different YAML.

    from src.teacher_loader import Teacher
    t = Teacher("configs/teachers/gemma4_26b_a4b.yaml")          # or backend="hf"
    out = t.generate("ስለ ኢትዮጵያ ታሪክ በአጭሩ ንገረኝ።")
"""
import copy
import os
import time
from pathlib import Path

from .metrics import strip_think
from .utils import ROOT, load_yaml

DEFAULTS = {
    "backend": "gguf",
    "strip_think": True,
    "system_prompt": "",
    "prompt_suffix": "",
    "chat_template_kwargs": {},
    "hf": {"quant": "4bit"},
    "gguf": {"path": "", "repo": "", "file": "", "n_ctx": 4096, "n_gpu_layers": 0},
    "generation": {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    p = Path(path)
    if not p.exists():
        p = ROOT / path
    cfg = _merge(DEFAULTS, load_yaml(p))
    cfg.setdefault("name", p.stem)
    return cfg


def list_teacher_configs(directory="configs/teachers"):
    return sorted((ROOT / directory).glob("*.yaml"))


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.get("tokenizer_repo") or cfg["hf_repo"],
                                         trust_remote_code=True)


class Teacher:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
        self.backend = backend or self.cfg["backend"]
        if self.backend == "hf":
            self._init_hf()
        elif self.backend == "gguf":
            self._init_gguf()
        else:
            raise ValueError(f"unknown backend {self.backend}")

    # ---- backends -----------------------------------------------------
    def _init_hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        repo = self.cfg["hf_repo"]
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        kw = dict(device_map="auto", trust_remote_code=True)
        if self.cfg["hf"]["quant"] == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        else:
            kw["torch_dtype"] = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        except Exception:  # multimodal checkpoints
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(repo, **kw)
        self.model.eval()
        self._fn = self._gen_hf

    def _init_gguf(self):
        from llama_cpp import Llama
        gg = self.cfg["gguf"]
        if gg.get("path"):
            path = gg["path"]
        else:
            if not (gg.get("repo") and gg.get("file")):
                raise SystemExit(f"[{self.cfg['name']}] set gguf.path, or gguf.repo + gguf.file, in the YAML")
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(gg["repo"], gg["file"])
        self.llm = Llama(model_path=path, n_ctx=gg["n_ctx"], n_threads=os.cpu_count(),
                         n_gpu_layers=gg["n_gpu_layers"], verbose=False)
        self._fn = self._gen_gguf

    # ---- generation ---------------------------------------------------
    def _messages(self, prompt):
        msgs = []
        if self.cfg.get("system_prompt"):
            msgs.append({"role": "system", "content": self.cfg["system_prompt"]})
        msgs.append({"role": "user", "content": prompt + (self.cfg.get("prompt_suffix") or "")})
        return msgs

    def _gen_hf(self, prompt, g):
        text = self.tok.apply_chat_template(self._messages(prompt), tokenize=False,
                                            add_generation_prompt=True,
                                            **self.cfg["chat_template_kwargs"])
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        temp = g.get("temperature", 0.7)
        out = self.model.generate(**inputs, max_new_tokens=g["max_new_tokens"], do_sample=temp > 0,
                                  temperature=temp if temp > 0 else None, top_p=g.get("top_p", 0.9))
        new = out[0][inputs["input_ids"].shape[1]:]
        n = len(new)
        return (self.tok.decode(new, skip_special_tokens=True), n,
                "length" if n >= g["max_new_tokens"] else "stop")

    def _gen_gguf(self, prompt, g):
        r = self.llm.create_chat_completion(messages=self._messages(prompt),
                                            max_tokens=g["max_new_tokens"],
                                            temperature=g.get("temperature", 0.7),
                                            top_p=g.get("top_p", 0.9))
        ch = r["choices"][0]
        return ch["message"]["content"], r["usage"]["completion_tokens"], ch.get("finish_reason", "stop")

    def generate(self, prompt, **override):
        g = {**self.cfg["generation"], **override}
        t0 = time.time()
        raw, n, fin = self._fn(prompt, g)
        dt = time.time() - t0
        text = strip_think(raw) if self.cfg["strip_think"] else raw.strip()
        return {"text": text, "raw": raw, "out_tokens": n, "finish_reason": fin,
                "seconds": round(dt, 2), "tok_per_s": round(n / max(dt, 1e-6), 2)}
