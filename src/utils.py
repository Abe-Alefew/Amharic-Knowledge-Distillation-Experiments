import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HORNMT_AM = "https://raw.githubusercontent.com/asmelashteka/HornMT/main/data/amh.txt"
HORNMT_EN = "https://raw.githubusercontent.com/asmelashteka/HornMT/main/data/eng.txt"


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "amharic-distillation/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


def normalize_text(s):
    s = unicodedata.normalize("NFC", s).translate(_ZW)
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
