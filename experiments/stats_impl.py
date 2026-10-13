"""How often do the implementation approximations actually fire in the generated data?

- buffer steps : generation steps taken while a UTF-8 character is incomplete (top-64 slow path)
- jamo texts   : outputs containing conjoining jamo (where stream canonicalisation matters)
- SP byte-fallback tokens in the vocabulary (<0xHH>)
Writes ../out/stats_impl.json
"""
import glob
import json
import os
import re

from transformers import AutoTokenizer

from common import decode_partial, token_bytes_table

OUT = os.environ.get("KOMAJ_OUT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "out"))
TOKP = {"qwen25_7b": "Qwen/Qwen2.5-7B-Instruct", "exaone_8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "kanana_8b": "kakaocorp/kanana-1.5-8b-instruct-2505", "qwen38_27b": os.environ.get("KOMAJ_BIG_MODEL", "Qwen3.8-27B-FP8")}
JAMO = re.compile("[\u1100-\u11ff]")
res = {}
for m, p in TOKP.items():
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    tb = token_bytes_table(tok, len(tok))
    sp = sum(1 for i in range(len(tok)) if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", tok.convert_ids_to_tokens(i) or ""))
    R = {"sp_byte_tokens": sp, "tags": {}}
    for f in sorted(glob.glob(f"{OUT}/gen_main/{m}/test/han*.jsonl")):
        steps = buf_steps = texts = jamo = 0
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            texts += 1
            jamo += bool(JAMO.search(r["text"]))
            buf = b""
            for t in r["token_ids"]:
                steps += 1
                buf_steps += bool(buf)  # this step was generated with an incomplete character pending
                _, buf = decode_partial(buf + (tb[t] if t < len(tb) else b""))
        R["tags"][os.path.basename(f)[:-6]] = {"steps": steps, "buf_steps": buf_steps, "buf_share": buf_steps / steps,
                                               "texts": texts, "jamo_texts": jamo}
    res[m] = R
    print(m, sp, {k: (round(v["buf_share"], 4), v["jamo_texts"]) for k, v in R["tags"].items()}, flush=True)
json.dump(res, open(f"{OUT}/stats_impl.json", "w"), indent=1)
