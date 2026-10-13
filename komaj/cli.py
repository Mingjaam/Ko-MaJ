"""Command line: detect a Ko-MaJ watermark in text files or stdin.

    komaj detect --key 123456789 essay.txt
    cat essay.txt | komaj detect --key 123456789 --max-syllables 200
"""
import argparse
import json
import sys

from . import __version__, detect


def main(argv=None):
    ap = argparse.ArgumentParser(prog="komaj", description="Ko-MaJ: Korean syllable-pair watermark")
    ap.add_argument("--version", action="version", version=f"komaj {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="detect the watermark (key only: no model or tokenizer needed)")
    d.add_argument("files", nargs="*", help="text files (default: stdin)")
    d.add_argument("--key", type=int, required=True)
    d.add_argument("--gamma", type=float, default=0.25)
    d.add_argument("--variant", choices=["h1", "h2", "h2r"], default="h1")
    d.add_argument("--threshold", type=float, default=4.0)
    d.add_argument("--max-syllables", type=int, default=None)
    d.add_argument("--json", action="store_true", help="one JSON object per input")
    a = ap.parse_args(argv)
    inputs = [(f, open(f, encoding="utf-8").read()) for f in a.files] or [("<stdin>", sys.stdin.read())]
    code = 1
    for name, text in inputs:
        r = detect(text, a.key, a.gamma, a.variant, a.threshold, a.max_syllables)
        code = 0 if r.is_watermarked else code
        if a.json:
            print(json.dumps({"file": name, "z": round(r.z, 4), "n_pairs": r.n_pairs, "p_value": r.p_value,
                              "watermarked": r.is_watermarked}, ensure_ascii=False))
        else:
            print(f"{name}: {r}")
    return code


if __name__ == "__main__":
    sys.exit(main())
