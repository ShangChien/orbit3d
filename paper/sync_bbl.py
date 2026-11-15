#!/usr/bin/env python3
"""Synchronize paper/main.bbl in citation order from paper/main.tex and paper/references.bib."""

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
        # Find matching closing brace
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


def format_authors(raw_author: str) -> tuple[str, str]:
    """Return (formatted_author_list, short_first_author_for_natbib)."""
    raw = raw_author.strip()
    if raw.startswith("{") and raw.endswith("}"):
        org = raw[1:-1]
        return org, org
    parts = [p.strip() for p in raw.split(" and ") if p.strip()]
    formatted = []
    first_surname = "Author"
    for idx, p in enumerate(parts):
        if p.lower() == "others":
            formatted.append("et~al.")
            continue
        if p.startswith("{") and p.endswith("}"):
            name = p[1:-1]
            if idx == 0:
                first_surname = name
            formatted.append(name)
            continue
        if "," in p:
            family, given = [x.strip() for x in p.split(",", 1)]
            if idx == 0:
                first_surname = family
            initials = " ".join((g[0] + ".") if len(g) > 1 and not g.endswith(".") else g for g in given.split())
            formatted.append(f"{initials}~{family}" if initials else family)
        else:
            tokens = p.split()
            if idx == 0 and tokens:
                first_surname = tokens[-1]
            formatted.append(p)
    short = (
        f"{first_surname} et~al."
        if (len(parts) > 2 or (len(parts) == 2 and parts[1].lower() == "others"))
        else (f"{first_surname} and {parts[1].split(',')[0].strip()}" if len(parts) == 2 else first_surname)
    )
    if len(formatted) > 6:
        author_str = ", ".join(formatted[:6]) + ", et~al."
    elif len(formatted) == 2 and formatted[1] == "et~al.":
        author_str = f"{formatted[0]} et~al."
    elif len(formatted) == 2:
        author_str = f"{formatted[0]} and {formatted[1]}"
    elif len(formatted) > 2:
        author_str = ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
    else:
        author_str = formatted[0] if formatted else "Unknown"
    return author_str.rstrip("."), short


def main() -> None:
    paper_dir = Path(__file__).resolve().parent
    tex_text = (paper_dir / "main.tex").read_text(encoding="utf-8")
    bib_entries = parse_bibtex((paper_dir / "references.bib").read_text(encoding="utf-8"))

    order: list[str] = []
    for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex_text):
        for k in match.group(1).split(","):
            key = k.strip()
            if key not in bib_entries:
                raise KeyError(f"Missing BibTeX entry for cited key: {key}")
            if key not in order:
                order.append(key)

    lines = [f"\\begin{{thebibliography}}{{{len(order)}}}"]
    lines.append("\\providecommand{\\natexlab}[1]{#1}")
    lines.append("\\providecommand{\\url}[1]{\\texttt{#1}}")
    lines.append("\\expandafter\\ifx\\csname urlstyle\\endcsname\\relax\\else\\urlstyle{same}\\fi")

    for key in order:
        rec = bib_entries[key]
        authors_full, short_auth = format_authors(rec.get("author", "Unknown"))
        year = rec.get("year", "2026")
        title = rec.get("title", "").rstrip(".")
        venue = (
            rec.get("journal")
            or rec.get("booktitle")
            or rec.get("howpublished")
            or rec.get("publisher")
            or ""
        )
        vol = rec.get("volume", "")
        pages = rec.get("pages", "")
        doi = rec.get("doi", "")
        url = rec.get("url", "")

        lines.append("")
        lines.append(f"\\bibitem[{short_auth}({year})]{{{key}}}")
        lines.append(f"{authors_full}.")
        lines.append(f"\\newblock {title}.")
        venue_bits = []
        if venue:
            venue_bits.append(f"\\emph{{{venue}}}")
        if vol:
            venue_bits.append(f"{vol}" + (f":{pages}" if pages else ""))
        elif pages:
            venue_bits.append(f"pages {pages}")
        venue_bits.append(year)
        lines.append("\\newblock " + ", ".join(venue_bits) + ".")
        if doi:
            doi_esc = doi.replace("_", "\\_")
            lines.append(f"\\newblock DOI: \\href{{https://doi.org/{doi}}}{{\\texttt{{{doi_esc}}}}}.")
        elif url:
            lines.append(f"\\newblock URL: \\url{{{url}}}.")

    lines.append("")
    lines.append("\\end{thebibliography}")
    lines.append("")

    out_bbl = paper_dir / "main.bbl"
    out_bbl.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sync_bbl] Synchronized {len(order)} citations -> {out_bbl}")


if __name__ == "__main__":
    main()
