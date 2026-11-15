#!/usr/bin/env python3
"""Synchronize paper/ncs/main_ncs.bbl in Nature citation order from main_ncs.tex and references.bib."""

from __future__ import annotations

import re
from pathlib import Path


def parse_bibtex(bib_text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)
    pos = 0
    while True:
        m = pattern.search(bib_text, pos)
        if not m:
            break
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        idx = m.end()
        depth = 1
        while idx < len(bib_text) and depth > 0:
            if bib_text[idx] == "{":
                depth += 1
            elif bib_text[idx] == "}":
                depth -= 1
            idx += 1
        body = bib_text[m.end() : idx - 1]
        pos = idx

        fields: dict[str, str] = {"_type": entry_type}
        f_idx = 0
        f_pat = re.compile(r"(\w+)\s*=\s*", re.M)
        while True:
            fm = f_pat.search(body, f_idx)
            if not fm:
                break
            fname = fm.group(1).lower()
            v_start = fm.end()
            while v_start < len(body) and body[v_start].isspace():
                v_start += 1
            if v_start >= len(body):
                break
            if body[v_start] == "{":
                d2 = 1
                v_end = v_start + 1
                while v_end < len(body) and d2 > 0:
                    if body[v_end] == "{":
                        d2 += 1
                    elif body[v_end] == "}":
                        d2 -= 1
                    v_end += 1
                val = body[v_start + 1 : v_end - 1]
                f_idx = v_end
            elif body[v_start] == '"':
                v_end = body.find('"', v_start + 1)
                val = body[v_start + 1 : v_end]
                f_idx = v_end + 1
            else:
                v_end = v_start
                while v_end < len(body) and body[v_end] not in ",\n\r":
                    v_end += 1
                val = body[v_start:v_end].strip()
                f_idx = v_end
            fields[fname] = re.sub(r"\s+", " ", val).strip()
        entries[key] = fields
    return entries


def _initials(given: str) -> str:
    tokens = [t.strip() for t in given.replace(".", ". ").split() if t.strip()]
    out = []
    for t in tokens:
        if "-" in t:
            sub = "-".join((s[0] + ".") for s in t.split("-") if s)
            out.append(sub)
        elif t.endswith(".") and len(t) <= 3:
            out.append(t)
        else:
            out.append(t[0] + ".")
    return " ".join(out)


def format_nature_authors(raw_author: str) -> str:
    """Format authors in Nature Computational Science style (Surname, I. J., ... or First et al.)."""
    raw = raw_author.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw[1:-1].strip()
    parts = [p.strip() for p in raw.split(" and ") if p.strip()]
    has_others = any(p.lower() == "others" for p in parts)
    parts = [p for p in parts if p.lower() != "others"]

    formatted = []
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            formatted.append(p[1:-1].strip())
            continue
        if "," in p:
            family, given = [x.strip() for x in p.split(",", 1)]
            init = _initials(given)
            formatted.append(f"{family}, {init}" if init else family)
        else:
            tokens = p.split()
            if len(tokens) > 1:
                family = tokens[-1]
                init = _initials(" ".join(tokens[:-1]))
                formatted.append(f"{family}, {init}")
            else:
                formatted.append(p)

    if not formatted:
        return "Unknown"
    if has_others or len(formatted) > 5:
        return f"{formatted[0]} et~al."
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} \\& {formatted[1]}"
    return ", ".join(formatted[:-1]) + f" \\& {formatted[-1]}"


def _clean_title(title: str) -> str:
    # Unwrap protective BibTeX inner braces around acronyms like {Uni-Mol2} -> Uni-Mol2
    cleaned = re.sub(r"\{([^{}]+)\}", r"\1", title)
    return cleaned.rstrip(".").strip()


def main() -> None:
    ncs_dir = Path(__file__).resolve().parent
    tex_text = (ncs_dir / "main_ncs.tex").read_text(encoding="utf-8")
    bib_entries = parse_bibtex((ncs_dir / "references.bib").read_text(encoding="utf-8"))

    order: list[str] = []
    for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex_text):
        for k in match.group(1).split(","):
            key = k.strip()
            if key not in bib_entries:
                raise KeyError(f"Missing BibTeX entry for cited key: {key}")
            if key not in order:
                order.append(key)

    lines = [
        f"\\begin{{thebibliography}}{{{len(order)}}}",
        "\\providecommand{\\url}[1]{\\texttt{#1}}",
        "\\expandafter\\ifx\\csname urlstyle\\endcsname\\relax\\else\\urlstyle{same}\\fi",
    ]

    for key in order:
        rec = bib_entries[key]
        authors = format_nature_authors(rec.get("author", "Unknown")).strip().rstrip(".")
        author_line = f"{authors}."

        title = _clean_title(rec.get("title", ""))
        year = rec.get("year", "2026")
        venue = (
            rec.get("journal")
            or rec.get("booktitle")
            or rec.get("howpublished")
            or rec.get("publisher")
            or ""
        )
        venue = re.sub(r"\{([^{}]+)\}", r"\1", venue).strip()
        vol = rec.get("volume", "").strip()
        pages = rec.get("pages", "").strip()
        doi = rec.get("doi", "").strip()
        url = rec.get("url", "").strip()

        venue_part = f"\\emph{{{venue}}}" if venue else ""
        if vol and pages:
            bib_loc = f"\\textbf{{{vol}}}, {pages}"
        elif vol:
            bib_loc = f"\\textbf{{{vol}}}"
        elif pages:
            bib_loc = pages
        else:
            bib_loc = ""

        detail_tokens = [t for t in [venue_part, bib_loc] if t]
        detail_str = " ".join(detail_tokens)
        pub_line = f"{detail_str} ({year})." if detail_str else f"({year})."
        if doi:
            doi_esc = doi.replace("_", "\\_")
            pub_line += f" \\href{{https://doi.org/{doi}}}{{https://doi.org/{doi_esc}}}."
        elif url:
            pub_line += f" \\url{{{url}}}."

        lines.append("")
        lines.append(f"\\bibitem{{{key}}}")
        lines.append(f"{author_line} {title}. {pub_line}")

    lines.append("")
    lines.append("\\end{thebibliography}")
    lines.append("")

    out_bbl = ncs_dir / "main_ncs.bbl"
    out_bbl.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sync_ncs_bbl] Synchronized {len(order)} Nature citations -> {out_bbl}")


if __name__ == "__main__":
    main()
