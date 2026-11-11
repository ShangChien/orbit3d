#!/usr/bin/env python3
"""Generate the pilot table and LaTeX manuscript from the Markdown source.

Install pypandoc-binary for the Pandoc executable. This command does not train or
evaluate models and does not compile or publish the paper.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import pypandoc

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "ORBIT3D_arXiv_Paper.md"
DATA = PAPER / "data/route_b_dpa4_scaling_summary.json"


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(ch, ch) for ch in str(text))


def write_bibliography(latex):
    """Include a synchronized numeric BBL for environments without BibTeX."""
    records = json.loads(pypandoc.convert_file(str(PAPER / "references.bib"), "csljson", format="bibtex"))
    entries = {record["id"]: record for record in records}
    order = []
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", latex):
        for key in group.split(","):
            key = key.strip()
            if key not in entries:
                raise ValueError(f"Unknown citation key: {key}")
            if key not in order:
                order.append(key)
    lines = [r"\begin{thebibliography}{"+str(len(order))+"}"]
    for key in order:
        record = entries[key]
        authors = []
        for author in record.get("author", []):
            name = author.get("literal") or " ".join(
                author.get(field, "") for field in
                ["given", "dropping-particle", "non-dropping-particle", "family"]
                if author.get(field)
            )
            if author.get("suffix"):
                name += ", "+author["suffix"]
            authors.append("et al." if name == "others" else name)
        if len(authors) > 4:
            authors = authors[:3]+["et al."]
        year = record["issued"]["date-parts"][0][0]
        # BibTeX case-protection braces become CSL nocase spans. Their text
        # remains case-preserved; the HTML wrapper must not enter LaTeX.
        plain = lambda value: html.unescape(re.sub(r"<[^>]+>", "", value))
        venue = plain(record.get("container-title", record.get("publisher", "")))
        lines.extend(["", r"\bibitem{"+key+"}", tex_escape(", ".join(authors)).rstrip(".")+".", r"\newblock "+tex_escape(plain(record["title"]))+"."])
        details = f"{venue}, {year}"
        if record.get("volume"):
            details += f", {record['volume']}"
        if record.get("page"):
            details += ":"+record["page"]
        lines.append(r"\newblock "+tex_escape(details)+".")
        if record.get("note"):
            lines.append(r"\newblock "+tex_escape(record["note"])+".")
        if record.get("URL"):
            lines.append(r"\newblock \url{"+record["URL"]+"}.")
    lines.append(r"\end{thebibliography}")
    (PAPER / "ORBIT3D_arXiv_Paper.bbl").write_text("\n".join(lines)+"\n")


def pilot_table():
    data = json.loads(DATA.read_text())
    provenance = json.loads((PAPER / "data/provenance.json").read_text())
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != provenance["source_sha256"]:
        raise ValueError("Pilot source changed: update and verify provenance before rendering tables")
    tasks = [
        ("Gap", "gap_ood_interval"), ("HOMO", "homo_interval"),
        ("LUMO", "lumo_interval"), ("Joint", "joint_gap_homo_lumo"),
    ]
    lines = [
        "**Pilot targets (Hartree).** Gap: [0.320, 0.345]; HOMO: [-0.205, -0.190]; LUMO: [0.060, 0.080]. Joint: Gap [0.285, 0.320], HOMO [-0.245, -0.220], LUMO [0.055, 0.085].",
        "",
        "| Task | Refinement backend | GVP hit (%) | RMS force (eV/Å) | Wall time (s) |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for name, task in tasks:
        for label, key in [("DPA4-Mini", "4_dpa4_mini_omol25"), ("DPA4-Plus", "7_dpa4_plus_omol25")]:
            row = data[task][key]
            lines.append(f"| {name} | {label} | {100*row['joint_interval_hit_rate']:.2f} | {row['dpa_rms_force_eV_A']:.4f} | {row['sampling_time_sec']:.2f} |")
    lines.extend(["", "Table: Archived octree/GVP pilot, 48 retained candidates per task. Forces are evaluated by DPA4-Plus. Values describe selected outputs under the legacy protocol, not independent validation of the target architecture."])
    return "\n".join(lines)


def build():
    text = SOURCE.read_text()
    evidence = json.loads((PAPER / 'data/unified_evidence.json').read_text())
    current = {}
    for name, source in evidence.items():
        if hashlib.sha256(source['json'].encode()).hexdigest() != source['sha256']:
            raise ValueError(f'Current experiment checksum mismatch: {name}')
        current[name] = json.loads(source['json'])
    test = current['ocelot_4096_20261111.json']['metrics']['test']
    lines = ['| Held-out OCELOT property | Bootstrap DPA MAE (eV) | Train-mean MAE (eV) |',
             '| :--- | ---: | ---: |']
    for key in ['homo', 'lumo']:
        lines.append(f"| {key.upper()} | {test['mae_eV'][key]:.4f} | {test['train_mean_baseline_mae_eV'][key]:.4f} |")
    lines += ['', 'Table: 4096 training records, three independent bootstrap adapters, 256 held-out test records.', '',
              '| Candidate ranking | Mean joint success (%) | Across-seed SD (percentage points) |',
              '| :--- | ---: | ---: |']
    for name, row in current['ranking_ablation_20261111.json']['aggregate'].items():
        lines.append(f"| {name.replace('_', ' ')} | {100*row['mean_joint_success_rate']:.2f} | {100*row['seed_std']:.2f} |")
    lines += ['', 'Table: Three inference seeds and identical proposal pools, before applying the later applicability-domain filter. Joint success uses the optimizing surrogate and includes all 32 attempts in its denominator.']
    experiment = current['conditioning_factorial_20261112.json']
    lines += ['', '| Conditioner | Even-electron constraint | Chemical validity (%) | Joint success (%) |',
              '| :--- | :--- | ---: | ---: |']
    for row in experiment['aggregate']:
        valid = row['raw_valid']['mean'] / experiment['budget_per_run']['attempts'] * 100
        hit = row['joint_success_rate']['mean'] * 100
        lines.append(f"| {row['variant'].replace('_', ' ')} | {'on' if row['electron_parity'] == 0 else 'off'} | {valid:.2f} | {hit:.2f} |")
    lines += ['', 'Table: Matched conditioner comparison with domain filtering, 64 attempts per inference seed (three seeds). Both variants use the same structural prior and adaptation labels; v3 jointly changes frequency encoding and interval sampling. The validity gain does not establish improved joint success.']
    lines += ['', '| GRPO objective | Valid / 192 | Optimizing-oracle joint hits / 192 | Independent-surrogate joint hits / 192 |',
              '| :--- | ---: | ---: | ---: |']
    ablation = current['reward_ablation_20261112.json']
    for variant in ['full', 'no_kl', 'no_physics', 'no_replay']:
        runs = [row for row in ablation['runs'] if row['variant'] == variant]
        assert len(runs) == 3 and sum(row['attempted'] for row in runs) == 192
        lines.append(f"| {variant.replace('_', ' ')} | {sum(row['raw_valid'] for row in runs)} | "
                     f"{sum(row['qualified'] for row in runs)} | "
                     f"{sum(row['independent']['independent_joint_successes'] for row in runs)} |")
    lines += ['', 'Table: Fifty rollout iterations (two optimizer passes per group), training seed 42, three inference seeds, and identical float64 energy reporting. The independent evaluator is a separately trained fingerprint surrogate, not DFT truth.']
    stages = current['stage_and_training_seeds_20261112.json']
    lines += ['', '| Policy stage | Attempts | Optimizing joint hits | Independent joint hits |',
              '| :--- | ---: | ---: | ---: |']
    for variant, label in [('unconditional_prior', 'Unconditional prior'), ('interval_sft', 'Interval SFT')]:
        runs = [row for row in stages['stage_runs'] if row['variant'] == variant]
        assert len(runs) == 3 and sum(row['attempted'] for row in runs) == 192
        lines.append(f"| {label} | 192 | {sum(row['qualified'] for row in runs)} | "
                     f"{sum(row['independent']['independent_joint_successes'] for row in runs)} |")
    runs = stages['grpo_replicas']
    assert len(runs) == 9 and sum(row['attempted'] for row in runs) == 576
    lines.append(f"| SFT + full GRPO | 576 | {sum(row['qualified'] for row in runs)} | "
                 f"{sum(row['independent']['independent_joint_successes'] for row in runs)} |")
    lines += ['', 'Table: Three inference seeds per policy. GRPO additionally uses three training seeds (42-44), all initialized from one SFT checkpoint. The prior-to-SFT comparison changes both conditioning and the self-shortlist.']
    refinement = current['refinement_matched_20261112.json']
    assert refinement['attempted_per_rule'] == 192 and refinement['settings']['step_size'] == .004
    lines += ['', '| Refinement rule | Optimizing joint hits / 192 | Independent joint hits / 192 | Mean RMS force (eV/Å) |',
              '| :--- | ---: | ---: | ---: |']
    for key, label in [('no_refinement', 'None'), ('energy_only', 'Energy only'),
                       ('projection_only', 'Projection only'), ('full_acceptance', 'Full acceptance')]:
        row = refinement['rules'][key]
        lines.append(f"| {label} | {row['joint_successes']} | {row['independent_joint_successes']} | "
                     f"{row['mean_final_rms_force_eV_A']:.4f} |")
    lines += ['', 'Table: Identical 34 selected starting structures from 192 proposals; 30-step budget and production step size 0.004. Selection is fixed, but backtracking changes evaluator cost. The independent fingerprint surrogate favors energy-only refinement.']
    matched = current['ocelot8192_evaluation_20261112.json']['evaluation']['matched_previous_test']
    lines += ['', '| Matched OCELOT test (256 structures) | 4096 training records, MAE (eV) | 8192 training records, MAE (eV) |',
              '| :--- | ---: | ---: |']
    for key in ['homo', 'lumo']:
        lines.append(f"| {key.upper()} | {matched['previous_model_mae_eV'][key]:.4f} | {matched['new_model_mae_eV'][key]:.4f} |")
    lines += ['', 'Table: Exact same test structures and reconstructed geometries; no overlap with either training set. Expanding training improves HOMO and slightly worsens LUMO on this test.']
    text = re.sub(r'<!-- BEGIN CURRENT RESULTS -->.*?<!-- END CURRENT RESULTS -->',
                  lambda _: '<!-- BEGIN CURRENT RESULTS -->\n'+'\n'.join(lines)+'\n<!-- END CURRENT RESULTS -->', text, flags=re.S)
    text = re.sub(
        r"<!-- BEGIN GENERATED PILOT TABLE -->.*?<!-- END GENERATED PILOT TABLE -->",
        lambda _: "<!-- BEGIN GENERATED PILOT TABLE -->\n"+pilot_table()+"\n<!-- END GENERATED PILOT TABLE -->",
        text, flags=re.S,
    )
    SOURCE.write_text(text)
    title = text.splitlines()[0].removeprefix("# ")
    abstract = text.split("## Abstract\n", 1)[1].split("\n## 1.", 1)[0].strip()
    body = text[text.index("## 1. Introduction"):].split("\n## References", 1)[0]
    body = re.sub(r"^(#{2,3}) \d+(?:\.\d+)?\. ", lambda m: "#"*(len(m[1])-1)+" ", body, flags=re.M)
    body = body.replace("figures/fig0_orbit3d_system_architecture.png", "figures/fig0_orbit3d_system_architecture.pdf")
    latex = pypandoc.convert_text(body, "latex", format="markdown+tex_math_dollars+citations", extra_args=[
        "--standalone", "--natbib", f"--template={PAPER / 'manuscript_template.tex'}",
        "--metadata", f"title={title}", "--metadata", f"abstract={abstract}",
    ])
    # The accessibility alt key emitted by recent Pandoc requires a recent
    # graphicx; keep arXiv's older TeX installations compatible.
    latex = re.sub(r",alt=\{[^}]*\}(?=\])", "", latex)
    # Give the short task label less width than the model and force columns.
    pilot_widths = ["0.1000", "0.2700", "0.1900", "0.2500", "0.1900"]
    for width in pilot_widths:
        latex = latex.replace(r"\real{0.2000}", r"\real{"+width+"}", 1)
    (PAPER / "ORBIT3D_arXiv_Paper.tex").write_text(latex)
    write_bibliography(latex)
    print(f"Updated {SOURCE.name} table and ORBIT3D_arXiv_Paper.tex (Pandoc {pypandoc.get_pandoc_version()})")


if __name__ == "__main__":
    build()
