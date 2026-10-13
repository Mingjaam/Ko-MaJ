"""Korean Wikipedia prompts / human continuations.

dev   : 200 prompts (hyper-parameter choices only)
test  : 500 prompts (final tables, run once)
human : 3000 paragraphs from disjoint articles (FPR / null distribution)
"""
import json
import random
import re
import sys

from datasets import load_dataset

from common import syllables

OUT = sys.argv[1] if len(sys.argv) > 1 else "../data"
random.seed(0)
SENT_END = re.compile(r"(?<=[다요음함됨까])\.\s")


def good_para(p):
    if len(p) < 500 or p.count("|") > 2 or p.startswith(("*", "#", "=", "-")):
        return False
    s = syllables(p)
    return len(s) / max(1, len(p)) > 0.6


def split_prompt(p):
    """Prompt = first sentences up to ~150-300 chars; continuation = rest (>=300 syllables)."""
    cut = None
    for m in SENT_END.finditer(p):
        if m.end() >= 150:
            cut = m.end()
            break
    if cut is None or cut > 300:
        return None
    prompt, cont = p[:cut].strip(), p[cut:].strip()
    if len(syllables(cont)) < 300:
        return None
    return prompt, cont


ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
ds = ds.shuffle(seed=0, buffer_size=20000)
pairs, human = [], []
for art in ds:
    paras = [q.strip() for q in art["text"].split("\n") if good_para(q.strip())]
    if not paras:
        continue
    if len(pairs) < 700:
        sp = split_prompt(paras[0])
        if sp:
            pairs.append({"id": f"w{len(pairs):04d}", "title": art["title"], "prompt": sp[0], "human": sp[1]})
            continue
    if len(human) < 3000:
        p = max(paras, key=len)
        if len(syllables(p)) >= 300:
            human.append({"id": f"h{len(human):04d}", "title": art["title"], "text": p})
    if len(pairs) >= 700 and len(human) >= 3000:
        break

random.shuffle(pairs)
for name, rows in [("dev", pairs[:200]), ("test", pairs[200:700]), ("human", human)]:
    with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
