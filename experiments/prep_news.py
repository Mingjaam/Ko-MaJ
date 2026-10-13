"""Second domain: Korean news (daekeun-ml/naver-news-summarization-ko).

news_test  : 500 prompts (first sentences of an article) + its human continuation
news_human : 3000 other articles (null distribution for this domain), disjoint from news_test
"""
import json
import random
import re
import sys

from datasets import load_dataset

from common import syllables

OUT = sys.argv[1] if len(sys.argv) > 1 else "../data"
random.seed(0)
SENT_END = re.compile(r"(?<=[다요])\.\s")
ds = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
idx = list(range(len(ds)))
random.shuffle(idx)
pairs, human = [], []
for i in idx:
    doc = re.sub(r"\s+", " ", ds[i]["document"]).strip()
    if len(syllables(doc)) < 450 or len(syllables(doc)) / len(doc) < 0.6:
        continue
    if len(pairs) < 500:
        cut = next((m.end() for m in SENT_END.finditer(doc) if m.end() >= 120), None)
        if cut and cut <= 300 and len(syllables(doc[cut:])) >= 300:
            pairs.append({"id": f"n{len(pairs):04d}", "title": ds[i]["title"], "prompt": doc[:cut].strip(),
                          "human": doc[cut:].strip()})
            continue
    if len(human) < 3000:
        human.append({"id": f"nh{len(human):04d}", "title": ds[i]["title"], "text": doc})
    if len(pairs) >= 500 and len(human) >= 3000:
        break
for name, rows in [("news_test", pairs), ("news_human", human)]:
    with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
