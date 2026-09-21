"""
Build the headings-only sheet that people who have NOT read the corpus use to write
independent questions. They see section titles, never the text, so their questions
reflect what a reader would ask rather than the documents' own wording.

    cd backend && python -m eval.make_heading_sheet
    -> eval/datasets/HEADINGS_SHEET.md
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
OUT = os.path.join(HERE, "datasets", "HEADINGS_SHEET.md")
_HEADING_RE = re.compile(r"^\s*(?:(?:Chapter|CHAPTER)[- ]?\d+[:.]?\s*.*|\d+(?:\.\d+)*\s+[A-Z].{2,80}|[A-Z][A-Z &/,()-]{8,})\s*$")


def headings(path: str) -> list[str]:
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if 4 <= len(t) <= 90 and _HEADING_RE.match(t) and t not in seen and not t.startswith("Page |") and "Dept of CSE" not in t:
                seen.add(t); out.append(t)
    return out


def main():
    parts = ["# Question-writing sheet (headings only)\n",
             "You are writing questions about three documents you have **not** read. Below are only their section titles.\n",
             "For each document write **about seven questions** a curious reader would ask after seeing these titles — ",
             "the kind of thing you would type into an assistant. Do not guess answers. Vary the style: some short, some ",
             "specific ('what does X mean by Y'), some comparative. Write them in the box under each document.\n\n",
             "Return the file; the author will fill in gold answers and evidence from the text afterwards.\n"]
    for name in sorted(os.listdir(CORPUS)):
        if not name.endswith(".txt"):
            continue
        parts.append(f"\n## {name}\n")
        for h in headings(os.path.join(CORPUS, name)):
            parts.append(f"- {h}")
        parts.append("\n**Your questions (one per line, start each with `Q:`):**\n\n```\nQ:\nQ:\nQ:\nQ:\nQ:\nQ:\nQ:\n```\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
