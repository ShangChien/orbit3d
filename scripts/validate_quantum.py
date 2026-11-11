"""Independent same-level OCELOT quantum validation of a fixed SDF sample set."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from rdkit import Chem
from orbit3d.quantum.ocelot import OcelotEvaluator, OcelotTheory


def evaluate_one(task):
    index, block, properties, output, settings, optimize = task
    workdir = Path(output) / f"molecule_{index:04d}"
    mol = Chem.MolFromMolBlock(block, removeHs=False)
    for k, v in properties.items():
        mol.SetProp(k, v)
    try:
        evaluator = OcelotEvaluator(OcelotTheory(**settings), workdir)
        result = evaluator.evaluate(mol, optimize_geometry=optimize)
        result.update(index=index, properties=properties, status="completed")
    except Exception as exc:
        result = {"index": index, "properties": properties, "status": "failed", "error": str(exc)}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "record.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--fixed-geometry", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    molecules = list(Chem.SDMolSupplier(str(a.sdf), removeHs=False))
    if a.limit:
        molecules = molecules[: a.limit]
    assert molecules and all(m is not None for m in molecules)
    settings = asdict(OcelotTheory(threads=a.threads))
    manifest = {
        "source_sha256": hashlib.sha256(a.sdf.read_bytes()).hexdigest(),
        "theory": settings,
        "sample_count": len(molecules),
        "optimize_geometry": not a.fixed_geometry,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = [
        (
            i,
            Chem.MolToMolBlock(m, forceV3000=True),
            {k: m.GetProp(k) for k in m.GetPropNames()},
            str(a.output),
            settings,
            not a.fixed_geometry,
        )
        for i, m in enumerate(molecules)
    ]
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(evaluate_one, t) for t in tasks]
        for future in as_completed(futures):
            r = future.result()
            records.append(r)
            print("completed", r["index"], r["status"], flush=True)
            (a.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    report = {
        "attempted": len(tasks),
        "completed": sum(r["status"] == "completed" for r in records),
        "elapsed_seconds": time.perf_counter() - start,
        "records": sorted(records, key=lambda r: r["index"]),
    }
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["completed"] != report["attempted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
