#!/usr/bin/env python3
"""Generate result tables and the LaTeX manuscript from the Markdown source.

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


def build():
    text = SOURCE.read_text()
    evidence = json.loads((PAPER / 'build/data/unified_evidence.json').read_text())
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
    lines += ['', 'Table: One four-target generator, three inference seeds per window and unchanged ranking/refinement limits. Inactive targets are masked. This experiment trains and reloads the expanded interface but does not establish useful joint four-property generation.']
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
    duration = current['sft_duration_20261112.json']
    if not duration['complete_predeclared_three_seed_matrix']:
        raise ValueError('The manuscript duration table requires all three fixed continuation seeds')
    lines = ['| Total steps / window | Valid / 576 | In domain / 576 | DPA joint / 576 | FP joint / 576 |',
             '| :--- | ---: | ---: | ---: | ---: |']
    for steps in ['10000', '20000']:
        for window in ['original', 'central']:
            row = duration['aggregate'][steps][window]
            assert row['attempted'] == 576
            lines.append(f"| {steps} / {window} | {row['raw_valid']} | {row['domain_eligible']} | "
                         f"{row['qualified']} | {row['fingerprint_joint']} |")
    lines += ['', 'Table: Three stochastic SFT continuations from one common 5000-step checkpoint, with three inference seeds per window and fixed total-10000/20000 evaluations. Each continuation uses the same 8188 labelled structures and a fresh optimizer. This does not replicate the initial 5000-step training or upstream pretraining.']
    tables['SFT_DURATION'] = '\n'.join(lines)
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
    holdout=current['oe62_source_holdout_20261113.json']
    lines=['| Frozen model | HOMO MAE (eV) | LUMO MAE (eV) | Joint coverage (%) |',
           '| :--- | ---: | ---: | ---: |']
    for key,label in [('dpa','DPA'),('fingerprint_v1','Original fingerprint'),('fingerprint_canonical','Canonical fingerprint'),('hyff','HyFF')]:
        row=holdout['models'][key];assert row['evaluated']==holdout['n']==4762
        metrics=row['overall']['targets']
        lines.append(f"| {label} | {metrics['homo']['mae_eV']:.4f} | {metrics['lumo']['mae_eV']:.4f} | {100*row['calibrated_90pct_joint_coverage']:.2f} |")
    lines += ['', 'Table: Frozen predictors on all 4,762 source-coordinate scaffold-test molecules. The last column is empirical coverage of the previously fitted nominal 90% joint intervals; it is not a guarantee for generated structures.']
    tables['OE62_HOLDOUT']='\n'.join(lines)
    lines=['| Target / arm | Domain (%) | FP MAE | HyFF MAE | FP hit (%) | HyFF hit (%) |',
           '| :------------------- | ----------: | --------: | ---------: | ----------: | -----------: |']
    for target in ['homo','lumo']:
        fp=current[f'oe62_formal_{target}_repeats_20261113.json']
        hyff=current[f'oe62_formal_{target}_hyff_repeats_20261113.json']
        for report in [fp,hyff]:
            assert report['training_seeds']==[42,43,44] and report['attempts_per_arm_per_training_seed']==1920
        for arm,label in [('frozen','Frozen'),('continue400','SFT 400'),('grpo200','GRPO'),('continue_wall','Time SFT')]:
            f,h=fp['arms'][arm],hyff['arms'][arm]
            lines.append(f"| {target.upper()} / {label} | {100*f['domain_rate']['mean']:.2f} | {f['independent_target_mae_eV']['mean']:.4f} | {h['independent_target_mae_eV']['mean']:.4f} | {100*f['independent_hit_rate']['mean']:.2f} | {100*h['independent_hit_rate']['mean']:.2f} |")
    lines += ['', 'Table: Formal OE62 raw-pool comparison, 5,760 attempts per row. MAE (eV) averages eligible structures within each target, then equally weights five targets and three training seeds. Property hits use tolerance 0.25 eV and all attempts as denominator. FP is the original frozen fingerprint model.']
    tables['OE62_FORMAL']='\n'.join(lines)
    quantum=current['oe62_point_quantum_20261113.json']
    lines=['| Target / arm | Input MAE | Input hits / 6 | Accepted / 6 | Final MAE | Final hits / 6 |',
           '| :------------------- | --------: | ---------: | ---------: | --------: | ---------: |']
    for target in ['homo','lumo']:
        for arm,label in [('continue_wall','Time SFT'),('grpo200','GRPO')]:
            row=quantum['groups'][target+'/'+arm];assert row['planned']==6
            lines.append(f"| {target.upper()} / {label} | {row['input_target']['mae_eV']:.4f} | {row['input_target']['hits']['0.25']} | {row['accepted_optimized']} | {row['optimized_target']['mae_eV']:.4f} | {row['optimized_target']['hits']['0.25']} |")
    lines += ['', 'Table: Frozen formal-policy quantum cohort. MAE is in eV; hits use a 0.25 eV tolerance. Accepted endpoints require geometry convergence and graph preservation. Final MAE uses accepted endpoints only; all hit denominators retain six selected candidates per row.']
    tables['OE62_QUANTUM']='\n'.join(lines)
    geometry=current['oe62_geometry_comparison_20261113.json']
    lines=['| Method | Graph / 48 | Graph + force / 48 | Joint proxy / 48 | E/F calls |',
           '| :--- | ---: | ---: | ---: | ---: |']
    for method,label in [('none','None'),('full','Full'),('energy_current','Current energy'),('lbfgs','LBFGS'),('fire','FIRE')]:
        row=next(r for r in geometry['aggregate'] if r['target']=='all' and r['method']==method)
        assert row['attempted']==48
        lines.append(f"| {label} | {row['graph_preserved']} | {row['graph_and_rms_below_0.1']} | {row['joint_success']} | {row['energy_force_calls']} |")
    full=next(r for r in geometry['aggregate'] if r['target']=='all' and r['method']=='full')
    lines += ['', f'Table: Identical 48 generated starting structures. Force means DPA RMS force at most 0.1 eV/Angstrom; joint proxy success additionally requires both DPA and HyFF within 0.25 eV of the requested point. Full also uses {full["property_gradient_calls"]} property-gradient calls. Optimizer budgets differ and all graph failures remain in the denominator.']
    tables['OE62_GEOMETRY']='\n'.join(lines)
    terminal=current['oe62_terminal_full_quantum_20261114.json']
    labels={'point_frozen':'Frozen SFT','terminal_sft':'Time SFT','terminal_structure':'Structure','terminal_physical':'Structure + physical','terminal_property':'Structure + physical + target'}
    lines=['| Feedback arm | Terminal | DFT success | DFT hits / 384 | Target MAE |','| :--- | ---: | ---: | ---: | ---: |']
    for name,label in labels.items():
        row=terminal['arms'][name];cells=list(row['cells'].values())
        lines.append(f"| {label} | {row['terminal_selected']} | {sum(c['dft_completed'] for c in cells)} | {sum(c['hits'] for c in cells)} | {row['mae_equal_target_eV']:.3f} |")
    lines += ['', 'Table: Complete frozen terminal pools at three central HOMO requests, one training seed. Every arm has 384 original proposals. DFT single-point hits use 0.25 eV tolerance. MAE (eV) conditions on successful endpoints and weights requests equally; it is not a DFT geometry-optimization result.']
    tables['TERMINAL_QUANTUM']='\n'.join(lines)
    full=current['oe62_full_source_quantum_20261114.json']['reports']
    labels={'subset8192_5000':'8k / 5k','full43951_5000':'44k / 5k','full43951_27000':'44k / 27k'}
    lines=['| Property / training | Terminal | DFT hits / 640 | Off hit (%) | MAE | Slope |','| :--- | ---: | ---: | ---: | ---: | ---: |']
    for target in ['homo','lumo']:
        for arm,label in labels.items():
            row=full[f'e14_{target}_correct']['arms'][arm];off=full[f'e14_{target}_off']['arms'][arm]
            hits=sum(c['hits'] for c in row['cells'].values())
            lines.append(f"| {target[0].upper()} / {label} | {row['terminal_selected']} | {hits} | {100*off['yield_equal_target']:.2f} | {row['mae_equal_target_eV']:.3f} | {row['response_slope']:.3f} |")
    lines += ['', 'Table: Full native-data SFT controls, all 2,026 selected single points converged. Correct-condition budgets are 640 per row; each off pool has 128 unique proposals rescored at five targets. H/L denote HOMO/LUMO; 8k/44k denote 8192/43951 source structures and 27k denotes 5k plus restarted 22k updates. MAE (eV) and DFT mean-response slope use successful endpoints.']
    tables['FULL_SOURCE_QUANTUM']='\n'.join(lines)
    lines=['| Property head | Generated test H/L | Native test H/L | Fresh pool H/L |','| :--- | ---: | ---: | ---: |']
    for size in ['8192','43951']:
        fitted=current[f'oe62_terminal_calibration{size}_20261114.json']['metrics'];transfer=current[f'oe62_terminal_transfer{size}_20261114.json']['all']
        for model,label,key in [('baseline','parent','baseline_mae_eV'),('adapted','+ DFT','adapted_mae_eV')]:
            pair=lambda xs:f'{xs[0]:.3f} / {xs[1]:.3f}'
            lines.append(f"| {size} {label} | {pair(fitted['test_generated'][key])} | {pair(fitted['source_test'][key])} | {pair(transfer[model]['mae_eV'])} |")
    lines += ['', 'Table: Property-prediction MAE in eV, distinct from target MAE. Generated test: 164 scaffold-held-out terminal structures; native test: 4,762 source geometries; fresh pool: 2,026 E14 endpoints excluded from fitting and early stopping.']
    tables['TERMINAL_CALIBRATION']='\n'.join(lines)
    feedback=current['oe62_calibrated_feedback_20261114.json']
    lines=['| Policy | Terminal | DFT hits / 1280 | Yield (%) | MAE | Slope |',
           '| :--- | ---: | ---: | ---: | ---: | ---: |']
    for arm,label in [('frozen','Frozen SFT'),('continue_wall','Time SFT'),('native_critic','Source GRPO'),('calibrated_critic','Calibrated GRPO')]:
        q=feedback['arms'][arm]['quantum']
        assert q['proposals']==1280
        hits=sum(c['hits'] for c in q['cells'].values())
        lines.append(f"| {label} | {q['terminal_selected']} | {hits} | {100*q['yield_equal_target']:.2f} | {q['mae_equal_target_eV']:.3f} | {q['response_slope']:.3f} |")
    lines += ['', 'Table: Complete E16 HOMO pools, one training seed and four inference seeds. All 2475 accepted endpoints, including 116 shared-off endpoints, converge in DFT. MAE (eV) and response slope use accepted endpoints; yields retain every proposal. The separate 256-proposal shared-off pool yields 8.05% when rescored across five goals.']
    tables['CALIBRATED_FEEDBACK']='\n'.join(lines)
    for name, content in tables.items():
        pattern = r'<!-- BEGIN TABLE '+name+r' -->.*?<!-- END TABLE '+name+r' -->'
        replacement = '<!-- BEGIN TABLE '+name+' -->\n'+content+'\n<!-- END TABLE '+name+' -->'
        text, count = re.subn(pattern, lambda _: replacement, text, flags=re.S)
        if count != 1:
            raise ValueError(f'Missing or duplicated generated-table region: {name}')
    SOURCE.write_text(text)
    title = text.splitlines()[0].removeprefix("# ")
    abstract = text.split("## Abstract\n", 1)[1].split("\n## 1.", 1)[0].strip()
    body = text[text.index("## 1. Introduction"):].split("\n## References", 1)[0]
    body = body.replace('## Appendix A.', '\\appendix\n\n## Appendix A.', 1)
    body = re.sub(r"^(#{2,3}) (?:(?:\d+(?:\.\d+)?|Appendix [A-Z])\. )?",
                  lambda m: "#"*(len(m[1])-1)+" ", body, flags=re.M)
    body = body.replace("figures/fig0_orbit3d_system_architecture.png", "figures/fig0_orbit3d_system_architecture.pdf")
    body = body.replace("figures/fig1_grpo_training_seeds.png", "figures/fig1_grpo_training_seeds.pdf")
    body = body.replace("figures/fig2_oe62_paired_effects.png", "figures/fig2_oe62_paired_effects.pdf")
    latex = pypandoc.convert_text(body, "latex", format="markdown+tex_math_dollars+citations", extra_args=[
        "--standalone", "--natbib", f"--template={PAPER / 'manuscript_template.tex'}",
        "--metadata", f"title={title}", "--metadata", f"abstract={abstract}",
    ])
    # The accessibility alt key emitted by recent Pandoc requires a recent
    # graphicx; keep arXiv's older TeX installations compatible.
    latex = re.sub(r",alt=\{[^}]*\}(?=\])", "", latex)
    latex = latex.replace(r'\begin{figure}', r'\begin{figure}[!htbp]')
    latex = compact_tables(latex)
    # Keep conclusion paragraphs unbroken before the appendices.
    latex = latex.replace(r'\section{Conclusion}', r'\begin{samepage}' + '\n' + r'\section{Conclusion}')
    latex = latex.replace(r'\appendix', r'\end{samepage}' + '\n' + r'\appendix', 1)
    latex = latex.replace(r'\subsection{', '\\FloatBarrier\n' + r'\subsection{')
    (PAPER / "ORBIT3D_arXiv_Paper.tex").write_text(latex)
    write_bibliography(latex)
    print(f"Updated {SOURCE.name} table and ORBIT3D_arXiv_Paper.tex (Pandoc {pypandoc.get_pandoc_version()})")


if __name__ == "__main__":
    build()
