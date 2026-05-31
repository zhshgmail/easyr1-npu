#!/usr/bin/env python3
"""npu-adapt-assist retrieval: input trace -> ranked KB candidates.

Loads all `docs/_meta/kb/porting_lessons/*.md`, parses YAML frontmatter,
scores each entry against the input trace, prints top-3 + the top-1
`correction` section.

Score:
  +3 per substring match of any `trigger` phrase in the input (case-insensitive)
  +2 per substring match of any `symptom_in_wild` phrase in the input
  +1 per token-overlap with title or layer

Threshold: top-1 must score >= 2 to be considered a match.

No external deps — stdlib only (re, pathlib, argparse, json, sys).
"""

import argparse
import json
import re
import sys
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "at", "of", "for", "and", "or",
    "to", "from", "with", "as", "it", "this", "that", "be", "by", "if",
    "not", "but", "no", "so", "do", "does", "has", "have", "are", "was",
    "were", "will", "can", "could", "would", "should", "i", "you", "we",
    "my", "your", "our",
}

MATCH_THRESHOLD = 2


def find_kb_dir(start: Path) -> Path:
    """Walk upward from start to find docs/_meta/kb/porting_lessons."""
    cur = start.resolve()
    while cur != cur.parent:
        candidate = cur / "docs" / "_meta" / "kb" / "porting_lessons"
        if candidate.is_dir():
            return candidate
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find docs/_meta/kb/porting_lessons by walking up from "
        f"{start}. Run from inside the easyr1-npu repo."
    )


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML-ish frontmatter parser.

    Supports:
      key: value
      key: |
        block
      key:
        - list
        - list
      key: >
        folded
        text
    Stops at second `---`.
    """
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    fm_text = text[4:end]
    body = text[end + 5:]

    fm: dict = {}
    lines = fm_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key = m.group(1)
        rest = m.group(2)
        if rest in ("", ">", "|", ">-", "|-"):
            # multi-line: collect either folded text or list items
            items: list[str] = []
            block_text: list[str] = []
            j = i + 1
            base_indent = None
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():
                    j += 1
                    continue
                indent = len(next_line) - len(next_line.lstrip())
                if base_indent is None:
                    if indent == 0:
                        break
                    base_indent = indent
                elif indent < base_indent:
                    break
                content = next_line[base_indent:]
                if content.startswith("- "):
                    items.append(content[2:].strip().strip('"').strip("'"))
                else:
                    block_text.append(content)
                j += 1
            if items:
                fm[key] = items
            else:
                fm[key] = " ".join(block_text).strip()
            i = j
        else:
            fm[key] = rest.strip().strip('"').strip("'")
            i += 1
    return fm, body


def extract_section(body: str, heading_substr: str) -> str:
    """Extract markdown heading section by substring match."""
    pattern = re.compile(
        rf"^#{{1,4}}.*{re.escape(heading_substr)}.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(body)
    if not m:
        return ""
    start = m.start()
    # find next heading of same-or-shallower level
    next_h = re.search(r"^#{1,4} ", body[m.end():], re.MULTILINE)
    end = m.end() + next_h.start() if next_h else len(body)
    return body[start:end].strip()


def tokenize(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) >= 3}


def score_entry(fm: dict, body: str, input_text: str, input_tokens: set[str]) -> int:
    score = 0
    input_lc = input_text.lower()

    for trig in fm.get("trigger", []) or []:
        if not trig:
            continue
        trig_lc = trig.lower()
        if trig_lc in input_lc:
            score += 3
            continue
        # also token overlap >= 3 distinct tokens of trigger
        trig_toks = tokenize(trig_lc)
        if trig_toks and len(trig_toks & input_tokens) >= 3:
            score += 2

    for sym in fm.get("symptom_in_wild", []) or []:
        if not sym:
            continue
        sym_lc = sym.lower()
        if sym_lc in input_lc:
            score += 2
            continue
        sym_toks = tokenize(sym_lc)
        if sym_toks and len(sym_toks & input_tokens) >= 3:
            score += 1

    title = (fm.get("title") or "").lower()
    layer = (fm.get("layer") or "").lower()
    score += len(tokenize(title) & input_tokens)
    if layer and layer in input_lc:
        score += 1
    return score


def load_kb(kb_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name in ("index.md", "_schema.md"):
            continue
        text = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if "id" not in fm:
            continue
        entries.append({
            "path": str(path),
            "id": fm["id"],
            "layer": fm.get("layer", ""),
            "title": fm.get("title", ""),
            "frontmatter": fm,
            "body": body,
        })
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieve KB entries matching a trace.")
    ap.add_argument("--trace-file", type=Path, help="Read input from file")
    ap.add_argument("--input", type=str, help="Inline input string")
    ap.add_argument("--kb-dir", type=Path, help="Override KB dir")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    ap.add_argument("--top", type=int, default=3, help="Top N to surface")
    args = ap.parse_args()

    if args.trace_file:
        input_text = args.trace_file.read_text(encoding="utf-8")
    elif args.input is not None:
        input_text = args.input
    else:
        input_text = sys.stdin.read()

    if not input_text.strip():
        print("Error: no input (use --trace-file or --input or pipe stdin)",
              file=sys.stderr)
        return 2

    kb_dir = args.kb_dir or find_kb_dir(Path(__file__).resolve())
    entries = load_kb(kb_dir)
    if not entries:
        print(f"No KB entries in {kb_dir}", file=sys.stderr)
        return 2

    input_tokens = tokenize(input_text)
    for e in entries:
        e["score"] = score_entry(e["frontmatter"], e["body"], input_text, input_tokens)

    entries.sort(key=lambda e: -e["score"])
    top = entries[: args.top]

    result = {
        "matched": top[0]["score"] >= MATCH_THRESHOLD if top else False,
        "top": [
            {
                "id": e["id"],
                "layer": e["layer"],
                "title": e["title"],
                "score": e["score"],
                "path": e["path"],
            }
            for e in top
        ],
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not result["matched"]:
        print("UNKNOWN — no KB entry matched above threshold")
        print(f"  best score: {top[0]['score']} < threshold {MATCH_THRESHOLD}")
        print()
        print("Weak matches (for reference):")
        for e in top:
            print(f"  - {e['id']} (score {e['score']}): {e['title'][:80]}")
        print()
        print("Add a new KB entry: copy docs/_meta/kb/porting_lessons/_schema.md")
        return 1

    best = entries[0]
    print(f"Top match: {best['id']} - {best['title']}")
    print(f"  layer: {best['layer']}  score: {best['score']}")
    print(f"  file: {best['path']}")
    print()
    correction = extract_section(best["body"], "correction")
    if not correction:
        # fallback: pull from frontmatter
        corr_lines = best["frontmatter"].get("correction") or []
        if corr_lines:
            print("Correction:")
            for c in corr_lines:
                print(f"  - {c}")
        else:
            print("(No correction section found; open the cookbook file.)")
    else:
        print(correction)
    print()
    if len(top) > 1:
        print("Also consider:")
        for e in top[1:]:
            print(f"  - {e['id']} (score {e['score']}): {e['title'][:80]}")
    print()
    print(f"Full cookbook: {best['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
