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


def compact_tables(latex):
    """Keep the manuscript's short result tables together as ordinary floats."""
    def convert(match):
        table = match.group(0)
        specification, remainder = table.removeprefix(r'\begin{longtable}[]').split(r'\caption', 1)
        caption, remainder = remainder.split(r'\tabularnewline', 1)
        header, remainder = remainder.split(r'\endfirsthead', 1)
        _, rows = remainder.split(r'\endlastfoot', 1)
        rows = rows.removesuffix(r'\end{longtable}')
        return (r'\begin{table}[!htbp]' + '\n' + r'\centering' + '\n' + r'\caption' + caption
                + '\n' + r'\begin{tabular}' + specification + header + rows
                + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}')
    return re.sub(r'\\begin\{longtable\}\[\].*?\\end\{longtable\}', convert, latex, flags=re.S)


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
    tables = {}
    lines = ['| Candidate ranking | Mean joint success (%) | Across-seed SD (percentage points) |',
             '| :--- | ---: | ---: |']
    for name, row in current['ranking_ablation_20261111.json']['aggregate'].items():
        lines.append(f"| {name.replace('_', ' ')} | {100*row['mean_joint_success_rate']:.2f} | {100*row['seed_std']:.2f} |")
    lines += ['', 'Table: Three inference seeds and identical proposal pools, before applying the later applicability-domain filter. Joint success uses the optimizing surrogate and includes all 32 attempts in its denominator.']
    tables['RANKING'] = '\n'.join(lines).strip()
    lines = []
    experiment = current['conditioning_factorial_20261112.json']
    lines += ['', '| Conditioner | Even-electron constraint | Chemical validity (%) | Joint success (%) |',
              '| :--- | :--- | ---: | ---: |']
    for row in experiment['aggregate']:
        valid = row['raw_valid']['mean'] / experiment['budget_per_run']['attempts'] * 100
        hit = row['joint_success_rate']['mean'] * 100
        lines.append(f"| {row['variant'].replace('_', ' ')} | {'on' if row['electron_parity'] == 0 else 'off'} | {valid:.2f} | {hit:.2f} |")
    lines += ['', 'Table: Matched conditioner comparison with domain filtering, 64 attempts per inference seed (three seeds). Both variants use the same structural prior and adaptation labels; v3 jointly changes frequency encoding and interval sampling. The validity gain does not establish improved joint success.']
    tables['CONDITIONING'] = '\n'.join(lines).strip()
    lines = []
    lines += ['', '| GRPO objective | Valid / 576 | Optimizing joint / 576 | Independent joint / 576 |',
              '| :--- | ---: | ---: | ---: |']
    ablation = current['grpo_factorial_20261112.json']['aggregate']
    for variant in ['full', 'no_kl', 'no_physics', 'no_replay', 'uncertainty_on']:
        row = ablation[variant]
        assert row['attempted'] == 576 and len(row['per_training_seed']) == 3
        lines.append(f"| {variant.replace('_', ' ')} | {row['raw_valid']} | "
                     f"{row['optimizing_joint']['total']} | {row['independent_joint']['total']} |")
    lines += ['', 'Table: Five objectives, three GRPO training seeds and three inference seeds. Each policy uses fifty rollout iterations with two optimizer passes per group and a common SFT initialization. The independent fingerprint evaluator is not DFT truth.']
    tables['GRPO'] = '\n'.join(lines).strip()
    lines = []
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
    tables['STAGES'] = '\n'.join(lines).strip()
    lines = []
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
    tables['REFINEMENT'] = '\n'.join(lines).strip()
    lines = []
    matched = current['ocelot8192_evaluation_20261112.json']['evaluation']['matched_previous_test']
    lines += ['', '| Matched OCELOT test (256 structures) | 4096 training records, MAE (eV) | 8192 training records, MAE (eV) |',
              '| :--- | ---: | ---: |']
    for key in ['homo', 'lumo']:
        lines.append(f"| {key.upper()} | {matched['previous_model_mae_eV'][key]:.4f} | {matched['new_model_mae_eV'][key]:.4f} |")
    lines += ['', 'Table: Exact same test structures and reconstructed geometries; no overlap with either training set. Expanding training improves HOMO and slightly worsens LUMO on this test.']
    tables['ORACLE_SCALE'] = '\n'.join(lines).strip()
    lines = []
    optical = current['optical_matched512_20261112.json']['models']
    lines += ['', '| Shared optical test (512 molecules) | DPA MAE (eV) | HyFF MAE (eV) | Fingerprint MAE (eV) |',
              '| :--- | ---: | ---: | ---: |']
    for key, label in [('homo', 'HOMO'), ('lumo', 'LUMO'), ('s0s1', 'Vertical S1'), ('s0t1', 'Vertical T1')]:
        lines.append(f"| {label} | {optical['dpa4']['mae_eV'][key]:.4f} | {optical['hyff_presto']['mae_eV'][key]:.4f} | {optical['fingerprint']['mae_eV'][key]:.4f} |")
    lines.append(f"| Paired vertical S1–T1 | {optical['dpa4']['vertical_singlet_triplet_gap']['mae_eV']:.4f} | "
                 f"{optical['hyff_presto']['vertical_singlet_triplet_gap']['mae_eV']:.4f} | {optical['fingerprint']['vertical_singlet_triplet_gap']['mae_eV']:.4f} |")
    lines += ['', 'Table: Official OCELOT labels on the same 512 scaffold-held-out molecules. DPA and HyFF share 8192 training structures and exact downstream geometries; the fingerprint model uses 20356 structures. Architecture and pretraining differ.  The last row averages per-molecule absolute gap errors.']
    tables['OPTICAL'] = '\n'.join(lines).strip()
    lines = []
    generated = current['optical_generation_20261112.json']['aggregate']
    lines += ['', '| Active targets | Valid / 192 | In domain / 192 | Optimizing joint / 192 | Independent joint / 192 |',
              '| :--- | ---: | ---: | ---: | ---: |']
    for key, label in [('orbitals','HOMO/LUMO'),('vertical','Vertical S1/T1'),('joint','All four')]:
        row=generated[key]
        assert row['attempted']==192
        lines.append(f"| {label} | {row['raw_valid']} | {row['domain_eligible']} | {row['qualified']} | {row['independent_joint_successes']} |")
    lines += ['', 'Table: One four-target generator, three inference seeds per window and unchanged ranking/refinement limits. Inactive targets are masked. This pilot trains and reloads the expanded interface but does not establish useful joint four-property generation.']
    tables['OPTICAL_GENERATION'] = '\n'.join(lines).strip()
    baseline = current['brics_comparison_20261112.json']['models']
    lines = ['| Proposal / target window | Valid / 192 | In domain / 192 | DPA joint / 192 | FP joint / 192 | HyFF joint / 192 |',
             '| :--- | ---: | ---: | ---: | ---: | ---: |']
    for window in ['original', 'central']:
        for method, label in [('octree_5000', 'Octree'), ('brics', 'BRICS')]:
            row = baseline[method][window]
            assert row['attempted'] == 192
            lines.append(f"| {label} / {window} | {row['raw_valid']} | {row['domain_eligible']} | {row['qualified']} | {row['fingerprint_joint']} | {row['hyff_joint_with_source_force']} |")
    lines += ['', 'Table: Same 8188 adaptation structures, target windows and proposal/ranking/refinement caps. Octree uses the 5000-step control; BRICS uses single-cut training fragments and ETKDG/UFF geometry. FP and HyFF independently rescore unchanged exports; all three joint columns use the source DPA force criterion. Actual evaluator calls and training compute differ.']
    tables['BRICS'] = '\n'.join(lines)
    qualified = current['qualified20k_quantum_20261112.json']
    lines = ['| Case / window | Source DPA H / L (eV) | Quantum H / L (eV) | Target check |',
             '| :--- | ---: | ---: | :--- |']
    for row in qualified['records']:
        prediction = row['source_prediction_eV']
        if row['status'] == 'completed':
            values = row['quantum_values_eV']
            reference = f"{values['homo']:.4f} / {values['lumo']:.4f}"
            missed = [name.upper() for name, hit in row['quantum_target_hits'].items() if not hit]
            outcome = ', '.join(missed) + ' miss' if missed else 'Hit'
        else:
            reference, outcome = 'Not completed', 'Failed'
        lines.append(f"| {row['quantum_index']} / {row['source_run'].split('_seed')[0]} | "
                     f"{prediction['homo']:.4f} / {prediction['lumo']:.4f} | {reference} | {outcome} |")
    lines += ['', 'Table: Preselected subset of eight surrogate-qualified designs from 384 attempts. One lowest-ID candidate is selected per nonempty window/inference-seed stratum before quantum calculation. All five preserve the source graph after S0 optimization; the three unselected qualified designs have unknown quantum outcomes. H/L denote HOMO/LUMO.']
    tables['QUALIFIED_QUANTUM'] = '\n'.join(lines)
    for name, content in tables.items():
        pattern = r'<!-- BEGIN TABLE '+name+r' -->.*?<!-- END TABLE '+name+r' -->'
        replacement = '<!-- BEGIN TABLE '+name+' -->\n'+content+'\n<!-- END TABLE '+name+' -->'
        text, count = re.subn(pattern, lambda _: replacement, text, flags=re.S)
        if count != 1:
            raise ValueError(f'Missing or duplicated generated-table region: {name}')
    text = re.sub(
        r"<!-- BEGIN GENERATED PILOT TABLE -->.*?<!-- END GENERATED PILOT TABLE -->",
        lambda _: "<!-- BEGIN GENERATED PILOT TABLE -->\n"+pilot_table()+"\n<!-- END GENERATED PILOT TABLE -->",
        text, flags=re.S,
    )
    SOURCE.write_text(text)
    title = text.splitlines()[0].removeprefix("# ")
    abstract = text.split("## Abstract\n", 1)[1].split("\n## 1.", 1)[0].strip()
    body = text[text.index("## 1. Introduction"):].split("\n## References", 1)[0]
    body = body.replace('## Appendix A.', '\\appendix\n\n## Appendix A.', 1)
    body = re.sub(r"^(#{2,3}) (?:(?:\d+(?:\.\d+)?|Appendix [A-Z])\. )?",
                  lambda m: "#"*(len(m[1])-1)+" ", body, flags=re.M)
    body = body.replace("figures/fig0_orbit3d_system_architecture.png", "figures/fig0_orbit3d_system_architecture.pdf")
    body = body.replace("figures/fig1_grpo_training_seeds.png", "figures/fig1_grpo_training_seeds.pdf")
    latex = pypandoc.convert_text(body, "latex", format="markdown+tex_math_dollars+citations", extra_args=[
        "--standalone", "--natbib", f"--template={PAPER / 'manuscript_template.tex'}",
        "--metadata", f"title={title}", "--metadata", f"abstract={abstract}",
    ])
    # The accessibility alt key emitted by recent Pandoc requires a recent
    # graphicx; keep arXiv's older TeX installations compatible.
    latex = re.sub(r",alt=\{[^}]*\}(?=\])", "", latex)
    latex = compact_tables(latex)
    latex = latex.replace(r'\subsection{', '\\FloatBarrier\n' + r'\subsection{')
    (PAPER / "ORBIT3D_arXiv_Paper.tex").write_text(latex)
    write_bibliography(latex)
    print(f"Updated {SOURCE.name} table and ORBIT3D_arXiv_Paper.tex (Pandoc {pypandoc.get_pandoc_version()})")


if __name__ == "__main__":
    build()
