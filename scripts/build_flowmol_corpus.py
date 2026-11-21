"""Cloud-ready multi-source full-periodic-table 3D molecular corpus builder for FlowMol pretraining.

Fetches and unifies five open-access 3D all-atom organic & organometallic datasets:
  - QM9        (DeepChem S3 qm9.zip, B3LYP/6-31G(2df,p) DFT)
  - OE62       (TUM mediaTUM df_62k.json, PBE+vdW DFT, 16 elements incl. Li & Se/Te/As/Si/B)
  - PCQM4Mv2   (HF CDN pcqm4m-v2-train.sdf.tar.gz, PubChemQC B3LYP/6-31G* DFT, ~3.12M mols)
  - GEOM-Drugs (FlowMol3 official Pitt CSB pre-split train/val/test_data.pickle, CREST/xTB)
  - tmQM       (UiO tmQM_X1..3.{xyz,BO}.gz, TPSSh/def2-SVP DFT + Wiberg BOs, 86k 3d/4d/5d+La complexes)

Outputs the compact Flat-CSR binary format (`loom.flowmol.data`) across the full 118-element
periodic table, 11 formal charges (-4..+6), and 6 bond classes (including COORDINATE/DATIVE),
with zero conformer leakage across train/val/test splits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import tarfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.request
import zipfile

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from loom.flowmol.constants import (
    ATOM_TO_IDX,
    CHARGE_TO_IDX,
    FLOWMOL_ATOM_TYPES,
    NON_METAL_ATOMIC_NUMS,
    mol_to_sparse_record,
    tensors_to_rdkit_mol,
)
from loom.flowmol.data import write_flat_corpus
from loom.utils.oe62 import adapt_record as adapt_oe62_record

RDLogger.DisableLog("rdApp.*")

URLS: Dict[str, List[str]] = {
    "qm9": [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
    ],
    "oe62": [
        "https://dataserv.ub.tum.de/index.php/s/m1507656/download?path=%2F&files=df_62k.json",
    ],
    "pcqm4mv2": [
        "https://huggingface.co/datasets/liuganghuggingface/moldata-pcqm4mv2/resolve/main/structures/pcqm4m-v2-train.sdf.tar.gz",
    ],
    "geom_drugs_train": [
        "https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle",
    ],
    "geom_drugs_val": [
        "https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle",
    ],
    "geom_drugs_test": [
        "https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle",
    ],
    "tmqm_x1_xyz": ["https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_X1.xyz.gz"],
    "tmqm_x2_xyz": ["https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_X2.xyz.gz"],
    "tmqm_x3_xyz": ["https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_X3.xyz.gz"],
    "tmqm_x1_bo": ["https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_X1.BO.gz"],
    "tmqm_x2_bo": ["https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_X2.BO.gz"],
    "tmqm_x3_bo": ["https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_X3.BO.gz"],
}


def _assign_split(key: str, seed: int = 42, val_frac: float = 0.015, test_frac: float = 0.015) -> int:
    """Deterministic molecule-level split: 0=train, 1=val, 2=test."""
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:16]
    u = int(h, 16) / float(1 << 64)
    if u < test_frac:
        return 2
    if u < test_frac + val_frac:
        return 1
    return 0


def _probe_range_support(url: str) -> Tuple[int, bool, str]:
    """Return `(content_length, accepts_ranges, final_url)` via HEAD."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 FlowMolBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        cl = int(resp.headers.get("Content-Length") or 0)
        ar = "bytes" in (resp.headers.get("Accept-Ranges") or "").lower()
        return cl, ar, resp.geturl()


def _download_range_part(args: Tuple[str, Path, int, int]) -> int:
    url, part_path, start, end = args
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return expected
    cmd = [
        "curl", "-sfL", "--retry", "5", "--retry-delay", "3",
        "--connect-timeout", "30",
        "-H", f"Range: bytes={start}-{end}",
        "-A", "Mozilla/5.0 FlowMolBuilder/1.0",
        "-o", str(part_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    got = part_path.stat().st_size
    if got != expected:
        raise RuntimeError(f"Range part {part_path.name} size mismatch: {got} != {expected}")
    return got


def _download_file(urls: Sequence[str], dest: Path, *, min_bytes: int = 1024, n_conns: int = 8) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[download] reusing existing {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
        return dest

    last_err = None
    for url in urls:
        t0 = time.perf_counter()
        try:
            total_bytes = 0
            can_range = False
            final_url = url
            try:
                total_bytes, can_range, final_url = _probe_range_support(url)
            except Exception:
                pass

            used_parallel = False
            if can_range and total_bytes >= 64 * 1024 * 1024 and n_conns > 1:
                try:
                    print(
                        f"[download] parallel ({n_conns}x) {dest.name} ({total_bytes / 1e6:.1f} MB) from {final_url[:90]}",
                        flush=True,
                    )
                    part_dir = dest.parent / f".parts_{dest.name}"
                    part_dir.mkdir(parents=True, exist_ok=True)
                    chunk = (total_bytes + n_conns - 1) // n_conns
                    tasks = []
                    part_paths = []
                    for i in range(n_conns):
                        s = i * chunk
                        e = min(total_bytes - 1, (i + 1) * chunk - 1)
                        if s > e:
                            break
                        pp = part_dir / f"part_{i:02d}"
                        part_paths.append(pp)
                        tasks.append((final_url, pp, s, e))
                    with ThreadPoolExecutor(max_workers=len(tasks)) as tex:
                        list(tex.map(_download_range_part, tasks))
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with tmp.open("wb") as out_f:
                        for pp in part_paths:
                            with pp.open("rb") as in_f:
                                shutil.copyfileobj(in_f, out_f, length=4 * 1024 * 1024)
                    shutil.rmtree(part_dir, ignore_errors=True)
                    if tmp.stat().st_size != total_bytes:
                        raise RuntimeError(f"Merged size mismatch: {tmp.stat().st_size} != {total_bytes}")
                    tmp.replace(dest)
                    used_parallel = True
                except Exception as pe:
                    print(f"[download] parallel range fallback to single-stream for {dest.name}: {pe}", flush=True)

            if not used_parallel:
                print(f"[download] streaming {dest.name} from {url[:90]}", flush=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                cmd = [
                    "curl", "-fL", "--retry", "5", "--retry-delay", "5",
                    "--connect-timeout", "30", "-C", "-",
                    "-A", "Mozilla/5.0 FlowMolBuilder/1.0",
                    "-o", str(tmp), url,
                ]
                subprocess.run(cmd, check=True)
                if tmp.stat().st_size < min_bytes:
                    raise RuntimeError(f"Downloaded file too small: {tmp.stat().st_size} bytes")
                tmp.replace(dest)

            dt = max(time.perf_counter() - t0, 1e-3)
            mb = dest.stat().st_size / (1024 * 1024)
            print(f"[download] OK {dest.name}: {mb:.1f} MB in {dt:.1f}s ({mb / dt:.1f} MB/s)", flush=True)
            return dest
        except Exception as e:
            last_err = e
            print(f"[download] WARN failed {url}: {e}", flush=True)
    raise RuntimeError(f"All download URLs failed for {dest.name}: {last_err}")


def _parse_sdf_chunk(args: Tuple[List[str], int, int, int]) -> List[Tuple[Dict[str, np.ndarray], int]]:
    blocks, max_atoms, seed, source_tag = args
    RDLogger.DisableLog("rdApp.*")
    out: List[Tuple[Dict[str, np.ndarray], int]] = []
    for block in blocks:
        if not block.strip():
            continue
        try:
            mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                continue
            if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
                continue
            rec = mol_to_sparse_record(mol, atom_to_idx=ATOM_TO_IDX, max_atoms=max_atoms)
            if rec is None:
                continue
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=False)
            sp = _assign_split(f"{source_tag}:{smi}", seed=seed)
            out.append((rec, sp))
        except Exception:
            continue
    return out


def _iter_sdf_chunks(stream: io.TextIOBase, chunk_size: int = 2000, max_mols: Optional[int] = None) -> Iterable[List[str]]:
    buf: List[str] = []
    chunk: List[str] = []
    total = 0
    for line in stream:
        buf.append(line)
        if line.startswith("$$$$"):
            chunk.append("".join(buf))
            buf.clear()
            total += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if max_mols is not None and total >= max_mols:
                break
    if chunk:
        yield chunk


def process_qm9(
    work_dir: Path,
    *,
    max_atoms: int,
    workers: int,
    seed: int,
    max_mols: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    archive = _download_file(URLS["qm9"], work_dir / "qm9.zip", min_bytes=10_000_000, n_conns=4)
    sdf_path = work_dir / "gdb9.sdf"
    if not sdf_path.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            cand = [n for n in zf.namelist() if n.endswith(".sdf") and not n.startswith("__MACOSX")]
            if not cand:
                raise RuntimeError(f"No .sdf found inside {archive}")
            with zf.open(cand[0]) as src, sdf_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    recs: List[Dict[str, np.ndarray]] = []
    splits: List[int] = []
    t0 = time.perf_counter()
    with sdf_path.open("r", encoding="utf-8", errors="ignore") as f:
        chunks = [(c, max_atoms, seed, 0) for c in _iter_sdf_chunks(f, chunk_size=2000, max_mols=max_mols)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_parse_sdf_chunk, chunks):
            for r, sp in res:
                recs.append(r)
                splits.append(sp)
    print(f"[qm9] accepted {len(recs):,} molecules in {time.perf_counter() - t0:.1f}s", flush=True)
    return recs, splits


def _recover_oe62_fallback(row: Dict[str, Any], max_atoms: int) -> Optional[Chem.Mol]:
    """Recover valid 3D OE62 molecules when strict HOMO/LUMO benchmark filters reject Li/charged/unlabeled rows."""
    xyz_block = row.get("xyz_pbe_relaxed")
    smi_raw = str(row.get("canonical_smiles") or "").strip()
    if not xyz_block or not smi_raw:
        return None
    xyz_mol = Chem.MolFromXYZBlock(xyz_block)
    if xyz_mol is None:
        return None
    na = int(xyz_mol.GetNumAtoms())
    if na < 2 or na > max_atoms:
        return None
    coords = np.asarray(xyz_mol.GetConformer().GetPositions(), dtype=np.float64)
    if not np.isfinite(coords).all():
        return None

    src = Chem.MolFromSmiles(smi_raw)
    if src is None or any(a.GetNumRadicalElectrons() > 0 for a in src.GetAtoms()):
        return None
    q_tot = int(Chem.GetFormalCharge(src))

    # Case 1: Single-fragment molecule (e.g. non-zero net charge or missing PBE0 frontier label)
    if len(Chem.GetMolFrags(src)) == 1:
        tmpl = Chem.AddHs(src)
        if tmpl.GetNumAtoms() == na:
            mc = Chem.Mol(xyz_mol)
            try:
                rdDetermineBonds.DetermineBondOrders(
                    mc, charge=q_tot, allowChargedFragments=True, embedChiral=False, maxIterations=250
                )
                Chem.SanitizeMol(mc)
                if not any(a.GetNumRadicalElectrons() > 0 for a in mc.GetAtoms()):
                    return mc
            except Exception:
                pass

    # Case 2: Lithium / alkali coordination complexes in OE62 (written as `[Li+].<anion>` in SMILES)
    metal_idx = [i for i, a in enumerate(xyz_mol.GetAtoms()) if int(a.GetAtomicNum()) not in NON_METAL_ATOMIC_NUMS]
    if len(metal_idx) == 1:
        m_i = metal_idx[0]
        m_sym = xyz_mol.GetAtomWithIdx(m_i).GetSymbol()
        m_q = 1 if m_sym in ("Li", "Na", "K", "Rb", "Cs") else 2
        lig_old = [i for i in range(na) if i != m_i]
        rw_l = Chem.RWMol()
        for old in lig_old:
            at = Chem.Atom(xyz_mol.GetAtomWithIdx(old).GetSymbol())
            at.SetNoImplicit(True)
            rw_l.AddAtom(at)
        conf_l = Chem.Conformer(len(lig_old))
        for idx, old in enumerate(lig_old):
            conf_l.SetAtomPosition(idx, (float(coords[old, 0]), float(coords[old, 1]), float(coords[old, 2])))
        ml = rw_l.GetMol()
        ml.AddConformer(conf_l, assignId=True)
        try:
            rdDetermineBonds.DetermineBondOrders(
                ml, charge=q_tot - m_q, allowChargedFragments=True, embedChiral=False, maxIterations=250
            )
            Chem.SanitizeMol(ml)
        except Exception:
            return None
        rw_full = Chem.RWMol()
        for i in range(na):
            at = Chem.Atom(xyz_mol.GetAtomWithIdx(i).GetSymbol())
            at.SetNoImplicit(True)
            rw_full.AddAtom(at)
        rw_full.GetAtomWithIdx(m_i).SetFormalCharge(m_q)
        for loc_i, old in enumerate(lig_old):
            la = ml.GetAtomWithIdx(loc_i)
            rw_full.GetAtomWithIdx(old).SetFormalCharge(int(la.GetFormalCharge()))
            rw_full.GetAtomWithIdx(old).SetIsAromatic(la.GetIsAromatic())
        for b in ml.GetBonds():
            rw_full.AddBond(lig_old[b.GetBeginAtomIdx()], lig_old[b.GetEndAtomIdx()], b.GetBondType())
        # Connect non-H donor atoms within coordination shell (<= 2.65 A, up to 4 donors) via DATIVE bonds
        donors = []
        for old in lig_old:
            z_o = int(xyz_mol.GetAtomWithIdx(old).GetAtomicNum())
            if z_o == 1:
                continue
            d = float(np.linalg.norm(coords[old] - coords[m_i]))
            if d <= 2.65:
                donors.append((d, old))
        donors.sort()
        if not donors:
            return None
        for _, old in donors[:4]:
            rw_full.AddBond(old, m_i, Chem.BondType.DATIVE)
        out_m = rw_full.GetMol()
        out_m.AddConformer(Chem.Conformer(xyz_mol.GetConformer()), assignId=True)
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(out_m, flags)
        out_m.UpdatePropertyCache(strict=False)
        if len(Chem.GetMolFrags(out_m)) == 1:
            return out_m
    return None


def _parse_oe62_chunk(args: Tuple[List[Dict[str, Any]], int, int]) -> List[Tuple[Dict[str, np.ndarray], int]]:
    rows, max_atoms, seed = args
    RDLogger.DisableLog("rdApp.*")
    out: List[Tuple[Dict[str, np.ndarray], int]] = []
    for row in rows:
        mol: Optional[Chem.Mol] = None
        try:
            mol = adapt_oe62_record(row, FLOWMOL_ATOM_TYPES, max_atoms=max_atoms)
        except Exception:
            try:
                mol = _recover_oe62_fallback(row, max_atoms=max_atoms)
            except Exception:
                mol = None
        if mol is None:
            continue
        try:
            rec = mol_to_sparse_record(mol, atom_to_idx=ATOM_TO_IDX, charge_to_idx=CHARGE_TO_IDX, max_atoms=max_atoms)
            if rec is None:
                continue
            key_str = str(row.get("refcode_csd") or row.get("canonical_smiles") or "")
            if not key_str:
                key_str = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=False)
            sp = _assign_split(f"oe62:{key_str}", seed=seed)
            out.append((rec, sp))
        except Exception:
            continue
    return out


def process_oe62(
    work_dir: Path,
    *,
    max_atoms: int,
    workers: int,
    seed: int,
    max_mols: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    json_path = _download_file(URLS["oe62"], work_dir / "df_62k.json", min_bytes=100_000_000, n_conns=8)
    t0 = time.perf_counter()
    import pandas as pd

    frame = pd.read_json(json_path, orient="split")
    rows = frame.to_dict("records")
    if max_mols is not None:
        rows = rows[:max_mols]
    chunk_size = 500
    chunks = [(rows[i : i + chunk_size], max_atoms, seed) for i in range(0, len(rows), chunk_size)]
    recs: List[Dict[str, np.ndarray]] = []
    splits: List[int] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_parse_oe62_chunk, chunks):
            for r, sp in res:
                recs.append(r)
                splits.append(sp)
    print(f"[oe62] accepted {len(recs):,} / {len(rows):,} molecules in {time.perf_counter() - t0:.1f}s", flush=True)
    return recs, splits


def process_pcqm4mv2(
    work_dir: Path,
    *,
    max_atoms: int,
    workers: int,
    seed: int,
    max_mols: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    archive = _download_file(URLS["pcqm4mv2"], work_dir / "pcqm4m-v2-train.sdf.tar.gz", min_bytes=500_000_000, n_conns=8)
    t0 = time.perf_counter()
    recs: List[Dict[str, np.ndarray]] = []
    splits: List[int] = []

    with tarfile.open(archive, "r:gz") as tf:
        sdf_member = None
        for m in tf:
            if m.isfile() and m.name.endswith(".sdf"):
                sdf_member = m
                break
        if sdf_member is None:
            raise RuntimeError("No .sdf member found inside pcqm4m-v2-train.sdf.tar.gz")
        raw_stream = tf.extractfile(sdf_member)
        assert raw_stream is not None
        text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8", errors="ignore")

        pending = []
        max_inflight = max(workers * 3, 16)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in _iter_sdf_chunks(text_stream, chunk_size=2500, max_mols=max_mols):
                pending.append(ex.submit(_parse_sdf_chunk, (chunk, max_atoms, seed, 2)))
                if len(pending) >= max_inflight:
                    for r, sp in pending.pop(0).result():
                        recs.append(r)
                        splits.append(sp)
                    if len(recs) % 300_000 < 2500:
                        print(f"[pcqm4mv2] progress: {len(recs):,} accepted ({time.perf_counter() - t0:.1f}s)", flush=True)
            for fut in pending:
                for r, sp in fut.result():
                    recs.append(r)
                    splits.append(sp)

    print(f"[pcqm4mv2] accepted {len(recs):,} molecules in {time.perf_counter() - t0:.1f}s", flush=True)
    return recs, splits


def _parse_pitt_geom_chunk(args: Tuple[List[bytes], int, int]) -> List[Dict[str, np.ndarray]]:
    """Worker: parse serialized RDKit mol bytes from FlowMol3 Pitt geom_raw split."""
    mol_bins, max_atoms, _ = args
    RDLogger.DisableLog("rdApp.*")
    out: List[Dict[str, np.ndarray]] = []
    for mb in mol_bins:
        try:
            mol = Chem.Mol(mb)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                continue
            if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
                continue
            Chem.SanitizeMol(mol)
            rec = mol_to_sparse_record(mol, atom_to_idx=ATOM_TO_IDX, max_atoms=max_atoms)
            if rec is not None:
                out.append(rec)
        except Exception:
            continue
    return out


def process_geom_drugs(
    work_dir: Path,
    *,
    max_atoms: int,
    max_confs_per_mol: int,
    workers: int,
    max_mols: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Load official FlowMol3 / MiDi pre-split GEOM-Drugs (`train/val/test_data.pickle`)."""
    split_specs = [
        ("geom_drugs_train", work_dir / "geom_train_data.pickle", 0),
        ("geom_drugs_val", work_dir / "geom_val_data.pickle", 1),
        ("geom_drugs_test", work_dir / "geom_test_data.pickle", 2),
    ]
    recs: List[Dict[str, np.ndarray]] = []
    splits: List[int] = []
    t0 = time.perf_counter()

    for url_key, local_path, split_id in split_specs:
        pkl_path = _download_file(URLS[url_key], local_path, min_bytes=100_000_000, n_conns=12)
        t_sp = time.perf_counter()
        print(f"[geom_drugs] unpickling {pkl_path.name} (split={split_id})...", flush=True)
        with pkl_path.open("rb") as f:
            raw_data = pickle.load(f)

        # raw_data is list of (smiles, [rd_mol, ...])
        mol_bins: List[bytes] = []
        n_mols_seen = 0
        for entry in raw_data:
            confs = entry[1] if isinstance(entry, (tuple, list)) and len(entry) >= 2 else []
            for item in confs[:max_confs_per_mol]:
                m = item.get("rd_mol") if isinstance(item, dict) else item
                if m is not None:
                    mol_bins.append(m.ToBinary())
            n_mols_seen += 1
            if max_mols is not None and n_mols_seen >= max_mols:
                break
        del raw_data
        gc.collect()

        chunk_size = 2000
        chunks = [(mol_bins[i : i + chunk_size], max_atoms, split_id) for i in range(0, len(mol_bins), chunk_size)]
        del mol_bins
        gc.collect()

        before = len(recs)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_parse_pitt_geom_chunk, chunks):
                recs.extend(res)
                splits.extend([split_id] * len(res))
        print(
            f"[geom_drugs] split={split_id}: mols={n_mols_seen:,} accepted_confs={len(recs) - before:,} in {time.perf_counter() - t_sp:.1f}s",
            flush=True,
        )

    print(f"[geom_drugs] TOTAL accepted {len(recs):,} conformers in {time.perf_counter() - t0:.1f}s", flush=True)
    return recs, splits


_RE_CSD = re.compile(r"CSD_code\s*=\s*(\S+)")
_RE_Q = re.compile(r"\bq\s*=\s*([+-]?\d+)")
_RE_S = re.compile(r"\bS\s*=\s*(\d+)")
_RE_MND = re.compile(r"\bMND\s*=\s*(\d+)")


def _iter_tmqm_xyz_gz(path: Path, max_atoms: int) -> Iterable[Tuple[str, str, List[str], np.ndarray]]:
    """Stream `(csd_code, header, symbols, pos_fp32)` from a `tmQM_X*.xyz.gz` archive."""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            ls = line.strip()
            if not ls:
                continue
            try:
                na = int(ls)
            except ValueError:
                continue
            hdr = f.readline().strip()
            if na < 2 or na > max_atoms:
                for _ in range(na):
                    if not f.readline():
                        break
                continue
            syms: List[str] = []
            pos = np.empty((na, 3), dtype=np.float32)
            ok = True
            for k in range(na):
                aline = f.readline()
                if not aline:
                    ok = False
                    break
                parts = aline.split()
                if len(parts) < 4:
                    ok = False
                elif ok:
                    syms.append(parts[0])
                    pos[k, 0] = float(parts[1])
                    pos[k, 1] = float(parts[2])
                    pos[k, 2] = float(parts[3])
            if not ok:
                continue
            m_code = _RE_CSD.search(hdr)
            if m_code is None:
                continue
            m_spin = _RE_S.search(hdr)
            if m_spin is not None and int(m_spin.group(1)) != 0:
                continue
            yield m_code.group(1), hdr, syms, pos


def _load_tmqm_bo_map(path: Path, wanted_codes: set[str]) -> Dict[str, List[str]]:
    """Load Wiberg bond-order blocks from `tmQM_X*.BO.gz` only for `wanted_codes`."""
    out: Dict[str, List[str]] = {}
    cur_code: Optional[str] = None
    cur_rows: List[str] = []
    keep = False
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            if ls.startswith("CSD_code"):
                if keep and cur_code is not None:
                    out[cur_code] = cur_rows
                m = _RE_CSD.search(ls)
                cur_code = m.group(1) if m is not None else None
                keep = bool(cur_code in wanted_codes)
                cur_rows = []
            elif keep:
                cur_rows.append(ls)
        if keep and cur_code is not None:
            out[cur_code] = cur_rows
    return out


def _build_tmqm_complex(hdr: str, syms: List[str], pos: np.ndarray, bo_rows: List[str]) -> Optional[Chem.Mol]:
    """Reconstruct a closed-shell organometallic RDKit Mol from DFT XYZ + Wiberg BOs + MND coordination."""
    pt = Chem.GetPeriodicTable()
    na = len(syms)
    z_list = [int(pt.GetAtomicNumber(s)) for s in syms]
    if any(z <= 0 or z > 118 for z in z_list):
        return None
    metal_set = [i for i, z in enumerate(z_list) if z not in NON_METAL_ATOMIC_NUMS]
    if len(metal_set) != 1:
        return None
    m_idx = metal_set[0]

    m_q = _RE_Q.search(hdr)
    m_mnd = _RE_MND.search(hdr)
    q_tot = int(m_q.group(1)) if m_q is not None else 0
    mnd = int(m_mnd.group(1)) if m_mnd is not None else 6

    bo_pairs: Dict[Tuple[int, int], float] = {}
    m_nbs: List[Tuple[float, int]] = []
    for row in bo_rows:
        toks = row.split()
        if len(toks) < 3:
            continue
        u = int(toks[0]) - 1
        if u < 0 or u >= na:
            continue
        k = 3
        while k + 2 < len(toks):
            v = int(toks[k + 1]) - 1
            w = float(toks[k + 2])
            k += 3
            if v < 0 or v >= na or u == v:
                continue
            if u == m_idx:
                m_nbs.append((w, v))
            elif v != m_idx:
                a, b = (u, v) if u < v else (v, u)
                if w > bo_pairs.get((a, b), 0.0):
                    bo_pairs[(a, b)] = w

    lig_old = [i for i in range(na) if i != m_idx]
    if not lig_old:
        return None
    old2lig = {old: idx for idx, old in enumerate(lig_old)}
    rw_lig = Chem.RWMol()
    for old in lig_old:
        at = Chem.Atom(syms[old])
        at.SetNoImplicit(True)
        rw_lig.AddAtom(at)
    conf_l = Chem.Conformer(len(lig_old))
    for idx, old in enumerate(lig_old):
        conf_l.SetAtomPosition(idx, (float(pos[old, 0]), float(pos[old, 1]), float(pos[old, 2])))

    lig_edges: Dict[Tuple[int, int], float] = {}
    for (u, v), w in bo_pairs.items():
        if w >= 0.48:
            lig_edges[(old2lig[u], old2lig[v])] = w

    # Enforce single covalent attachment for H atoms (pick highest Wiberg BO partner)
    h_best: Dict[int, Tuple[float, int]] = {}
    for (lu, lv), w in list(lig_edges.items()):
        for hc, other in ((lu, lv), (lv, lu)):
            if z_list[lig_old[hc]] == 1 and w > h_best.get(hc, (0.0, -1))[0]:
                h_best[hc] = (w, other)
    for (lu, lv) in list(lig_edges.keys()):
        if z_list[lig_old[lu]] == 1 and h_best.get(lu, (0.0, -1))[1] != lv:
            del lig_edges[(lu, lv)]
        elif z_list[lig_old[lv]] == 1 and h_best.get(lv, (0.0, -1))[1] != lu:
            del lig_edges[(lu, lv)]

    for lu, lv in lig_edges:
        rw_lig.AddBond(lu, lv, Chem.BondType.SINGLE)
    lig_base = rw_lig.GetMol()
    lig_base.AddConformer(conf_l, assignId=True)

    frags = Chem.GetMolFrags(lig_base, asMols=True, sanitizeFrags=False)
    frag_atom_lists = Chem.GetMolFrags(lig_base, asMols=False, sanitizeFrags=False)

    solved_frags: List[Tuple[Tuple[int, ...], Chem.Mol]] = []
    for f_mol, f_atoms in zip(frags, frag_atom_lists):
        total_elec = sum(z_list[lig_old[a]] for a in f_atoms)
        cand_qs = [q for q in (0, -1, 1, -2, 2, -3) if (total_elec - q) % 2 == 0]
        best_f: Optional[Chem.Mol] = None
        best_err = 1e9
        for cq in cand_qs:
            try:
                mc = Chem.Mol(f_mol)
                rdDetermineBonds.DetermineBondOrders(
                    mc, charge=cq, allowChargedFragments=True, embedChiral=False, maxIterations=250
                )
                Chem.SanitizeMol(mc)
                if any(a.GetNumRadicalElectrons() > 0 for a in mc.GetAtoms()):
                    continue
                err = 0.18 * abs(cq)
                for b in mc.GetBonds():
                    ga = f_atoms[b.GetBeginAtomIdx()]
                    gb = f_atoms[b.GetEndAtomIdx()]
                    key = (ga, gb) if ga < gb else (gb, ga)
                    w_dft = lig_edges.get(key, 1.0)
                    b_ord = 1.5 if b.GetIsAromatic() else float(b.GetBondTypeAsDouble())
                    err += abs(b_ord - w_dft)
                if err < best_err:
                    best_err = err
                    best_f = mc
            except Exception:
                continue
        if best_f is None:
            return None
        solved_frags.append((f_atoms, best_f))

    rw_full = Chem.RWMol()
    for i in range(na):
        at = Chem.Atom(syms[i])
        at.SetNoImplicit(True)
        rw_full.AddAtom(at)

    lig_q_sum = 0
    for f_atoms, best_f in solved_frags:
        for loc_i, l_idx in enumerate(f_atoms):
            orig_i = lig_old[l_idx]
            fa = best_f.GetAtomWithIdx(loc_i)
            fq = int(fa.GetFormalCharge())
            lig_q_sum += fq
            rw_full.GetAtomWithIdx(orig_i).SetFormalCharge(fq)
            rw_full.GetAtomWithIdx(orig_i).SetIsAromatic(fa.GetIsAromatic())
        for b in best_f.GetBonds():
            u_orig = lig_old[f_atoms[b.GetBeginAtomIdx()]]
            v_orig = lig_old[f_atoms[b.GetEndAtomIdx()]]
            rw_full.AddBond(u_orig, v_orig, b.GetBondType())

    q_metal = max(-4, min(6, q_tot - lig_q_sum))
    rw_full.GetAtomWithIdx(m_idx).SetFormalCharge(q_metal)

    m_nbs.sort(reverse=True)
    coord_targets = {v for w, v in m_nbs[: min(max(mnd, 1), 14)] if w >= 0.12}
    for f_atoms, _ in solved_frags:
        orig_set = {lig_old[x] for x in f_atoms}
        if not (orig_set & coord_targets):
            cand = [(w, v) for w, v in m_nbs if v in orig_set]
            if cand:
                coord_targets.add(max(cand)[1])
            else:
                dists = [(float(np.linalg.norm(pos[v] - pos[m_idx])), v) for v in orig_set]
                coord_targets.add(min(dists)[1])
    if len(coord_targets) > 16:
        return None

    for v in sorted(coord_targets):
        if rw_full.GetBondBetweenAtoms(v, m_idx) is None:
            rw_full.AddBond(v, m_idx, Chem.BondType.DATIVE)

    mol = rw_full.GetMol()
    conf = Chem.Conformer(na)
    for i in range(na):
        conf.SetAtomPosition(i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    mol.AddConformer(conf, assignId=True)
    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    Chem.SanitizeMol(mol, flags)
    mol.UpdatePropertyCache(strict=False)
    return mol


def _parse_tmqm_chunk(
    args: Tuple[List[Tuple[str, str, List[str], np.ndarray, List[str]]], int, int]
) -> List[Tuple[Dict[str, np.ndarray], int]]:
    items, max_atoms, seed = args
    RDLogger.DisableLog("rdApp.*")
    out: List[Tuple[Dict[str, np.ndarray], int]] = []
    for csd_code, hdr, syms, pos, bo_rows in items:
        try:
            mol = _build_tmqm_complex(hdr, syms, pos, bo_rows)
            if mol is None:
                continue
            rec = mol_to_sparse_record(mol, atom_to_idx=ATOM_TO_IDX, charge_to_idx=CHARGE_TO_IDX, max_atoms=max_atoms)
            if rec is None:
                continue
            # Verify lossless round-trip through tensors_to_rdkit_mol
            n_a = int(rec["pos"].shape[0])
            e_up = np.zeros(n_a * (n_a - 1) // 2, dtype=np.int64)
            if len(rec["bond_u"]) > 0:
                bu = rec["bond_u"].astype(np.int64)
                bv = rec["bond_v"].astype(np.int64)
                e_up[bu * n_a - (bu * (bu + 1)) // 2 + (bv - bu - 1)] = rec["bond_type"].astype(np.int64)
            _, diag = tensors_to_rdkit_mol(rec["pos"], rec["atom_type"], rec["charge"], e_up, sanitize=True)
            if not (diag["valid"] and diag["connected"]):
                continue
            sp = _assign_split(f"tmqm:{csd_code}", seed=seed)
            out.append((rec, sp))
        except Exception:
            continue
    return out


def process_tmqm(
    work_dir: Path,
    *,
    max_atoms: int,
    workers: int,
    seed: int,
    max_mols: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Load UiO `tmQM` (`tmQM_X1..3.{xyz,BO}.gz`, 86,665 3d/4d/5d+La DFT complexes)."""
    part_specs = [
        ("tmqm_x1_xyz", "tmqm_x1_bo", "tmQM_X1.xyz.gz", "tmQM_X1.BO.gz"),
        ("tmqm_x2_xyz", "tmqm_x2_bo", "tmQM_X2.xyz.gz", "tmQM_X2.BO.gz"),
        ("tmqm_x3_xyz", "tmqm_x3_bo", "tmQM_X3.xyz.gz", "tmQM_X3.BO.gz"),
    ]
    recs: List[Dict[str, np.ndarray]] = []
    splits: List[int] = []
    t0 = time.perf_counter()
    seen_total = 0

    for xyz_key, bo_key, xyz_fn, bo_fn in part_specs:
        xyz_p = _download_file(URLS[xyz_key], work_dir / xyz_fn, min_bytes=1_000_000, n_conns=4)
        bo_p = _download_file(URLS[bo_key], work_dir / bo_fn, min_bytes=1_000_000, n_conns=4)
        xyz_entries: List[Tuple[str, str, List[str], np.ndarray]] = []
        for entry in _iter_tmqm_xyz_gz(xyz_p, max_atoms=max_atoms):
            xyz_entries.append(entry)
            seen_total += 1
            if max_mols is not None and seen_total >= max_mols:
                break
        wanted = {e[0] for e in xyz_entries}
        bo_map = _load_tmqm_bo_map(bo_p, wanted)
        paired = [(c, h, s, p, bo_map[c]) for (c, h, s, p) in xyz_entries if c in bo_map]
        chunk_size = 300
        chunks = [(paired[i : i + chunk_size], max_atoms, seed) for i in range(0, len(paired), chunk_size)]
        before = len(recs)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_parse_tmqm_chunk, chunks):
                for r, sp in res:
                    recs.append(r)
                    splits.append(sp)
        print(
            f"[tmqm] {xyz_fn}: <=max_atoms={len(paired):,} accepted={len(recs) - before:,} ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        if max_mols is not None and seen_total >= max_mols:
            break

    print(f"[tmqm] TOTAL accepted {len(recs):,} organometallic complexes in {time.perf_counter() - t0:.1f}s", flush=True)
    return recs, splits


def _merge_flat_shards(
    out_dir: Path,
    shard_dirs: Sequence[Path],
    source_names: Sequence[str],
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Concatenate per-source Flat-CSR shards in pure NumPy without holding millions of Python dicts in RAM."""
    from loom.flowmol.constants import (
        FLOWMOL_BOND_TYPES,
        FLOWMOL_FORMAL_CHARGES,
        N_ATOM_TYPES,
        N_BOND_TYPES,
        N_CHARGES,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    atom_ptr_parts = [np.zeros(1, dtype=np.int64)]
    bond_ptr_parts = [np.zeros(1, dtype=np.int64)]
    na_parts, sp_parts, sid_parts = [], [], []
    pos_parts, at_parts, chg_parts, bu_parts, bv_parts, bt_parts = [], [], [], [], [], []

    a_off = 0
    b_off = 0
    for sid, sdir in enumerate(shard_dirs):
        ap = np.load(sdir / "atom_ptr.npy")
        bp = np.load(sdir / "bond_ptr.npy")
        na = np.load(sdir / "n_atoms.npy")
        sp = np.load(sdir / "split.npy")
        atom_ptr_parts.append(ap[1:] + a_off)
        bond_ptr_parts.append(bp[1:] + b_off)
        a_off += int(ap[-1])
        b_off += int(bp[-1])
        na_parts.append(na)
        sp_parts.append(sp)
        sid_parts.append(np.full(len(na), sid, dtype=np.uint8))
        pos_parts.append(np.load(sdir / "pos.npy"))
        at_parts.append(np.load(sdir / "atom_type.npy"))
        chg_parts.append(np.load(sdir / "charge.npy"))
        bu_parts.append(np.load(sdir / "bond_u.npy"))
        bv_parts.append(np.load(sdir / "bond_v.npy"))
        bt_parts.append(np.load(sdir / "bond_type.npy"))

    atom_ptr = np.concatenate(atom_ptr_parts, axis=0)
    bond_ptr = np.concatenate(bond_ptr_parts, axis=0)
    n_atoms_arr = np.concatenate(na_parts, axis=0)
    split_arr = np.concatenate(sp_parts, axis=0)
    src_arr = np.concatenate(sid_parts, axis=0)
    pos_all = np.concatenate(pos_parts, axis=0)
    atom_all = np.concatenate(at_parts, axis=0)
    chg_all = np.concatenate(chg_parts, axis=0)
    bu_all = np.concatenate(bu_parts, axis=0) if b_off > 0 else np.zeros(0, dtype=np.uint8)
    bv_all = np.concatenate(bv_parts, axis=0) if b_off > 0 else np.zeros(0, dtype=np.uint8)
    bt_all = np.concatenate(bt_parts, axis=0) if b_off > 0 else np.zeros(0, dtype=np.uint8)

    np.save(out_dir / "atom_ptr.npy", atom_ptr)
    np.save(out_dir / "bond_ptr.npy", bond_ptr)
    np.save(out_dir / "n_atoms.npy", n_atoms_arr)
    np.save(out_dir / "split.npy", split_arr)
    np.save(out_dir / "source_id.npy", src_arr)
    np.save(out_dir / "pos.npy", pos_all)
    np.save(out_dir / "atom_type.npy", atom_all)
    np.save(out_dir / "charge.npy", chg_all)
    np.save(out_dir / "bond_u.npy", bu_all)
    np.save(out_dir / "bond_v.npy", bv_all)
    np.save(out_dir / "bond_type.npy", bt_all)

    n_mols = int(len(n_atoms_arr))
    train_na = n_atoms_arr[split_arr == 0]
    hist_vals, hist_counts = np.unique(train_na if len(train_na) > 0 else n_atoms_arr, return_counts=True)
    elem_counts_np = np.bincount(atom_all.astype(np.int64), minlength=N_ATOM_TYPES)
    chg_counts_np = np.bincount(chg_all.astype(np.int64), minlength=N_CHARGES)
    bond_counts_np = np.bincount(bt_all.astype(np.int64), minlength=N_BOND_TYPES)

    mol_source_atom_id = np.repeat(src_arr, n_atoms_arr.astype(np.int64))
    per_source_n_atoms_hist: Dict[str, Dict[int, int]] = {}
    per_source_active_elements: Dict[str, List[str]] = {}
    split_source_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for sid, sname in enumerate(source_names):
        s_mask = src_arr == sid
        for sp_id, sp_name in ((0, "train"), (1, "val"), (2, "test")):
            split_source_counts[sp_name][sname] = int((s_mask & (split_arr == sp_id)).sum())
        s_train_na = n_atoms_arr[s_mask & (split_arr == 0)]
        s_na_ref = s_train_na if len(s_train_na) > 0 else n_atoms_arr[s_mask]
        if len(s_na_ref) > 0:
            sv, sc = np.unique(s_na_ref, return_counts=True)
            per_source_n_atoms_hist[sname] = {int(k): int(v) for k, v in zip(sv.tolist(), sc.tolist())}
        else:
            per_source_n_atoms_hist[sname] = {}
        s_atoms = atom_all[mol_source_atom_id == sid]
        s_ec = np.bincount(s_atoms.astype(np.int64), minlength=N_ATOM_TYPES)
        per_source_active_elements[sname] = [FLOWMOL_ATOM_TYPES[i] for i in range(N_ATOM_TYPES) if int(s_ec[i]) > 0]

    meta: Dict[str, Any] = {
        "atom_types": list(FLOWMOL_ATOM_TYPES),
        "formal_charges": list(FLOWMOL_FORMAL_CHARGES),
        "bond_types": list(FLOWMOL_BOND_TYPES),
        "n_atom_types": N_ATOM_TYPES,
        "n_charges": N_CHARGES,
        "n_bond_types": N_BOND_TYPES,
        "n_mols": n_mols,
        "n_atoms_total": int(atom_ptr[-1]),
        "n_bonds_total": int(bond_ptr[-1]),
        "source_names": list(source_names),
        "source_counts": {sname: int((src_arr == sid).sum()) for sid, sname in enumerate(source_names)},
        "split_source_counts": split_source_counts,
        "split_counts": {
            "train": int((split_arr == 0).sum()),
            "val": int((split_arr == 1).sum()),
            "test": int((split_arr == 2).sum()),
        },
        "n_atoms_min": int(n_atoms_arr.min()),
        "n_atoms_median": float(np.median(n_atoms_arr)),
        "n_atoms_p95": float(np.percentile(n_atoms_arr, 95)),
        "n_atoms_max": int(n_atoms_arr.max()),
        "n_atoms_hist": {int(k): int(v) for k, v in zip(hist_vals.tolist(), hist_counts.tolist())},
        "per_source_n_atoms_hist": per_source_n_atoms_hist,
        "per_source_active_elements": per_source_active_elements,
        "n_active_elements": int((elem_counts_np > 0).sum()),
        "element_counts": {FLOWMOL_ATOM_TYPES[i]: int(elem_counts_np[i]) for i in range(N_ATOM_TYPES)},
        "charge_counts": {str(FLOWMOL_FORMAL_CHARGES[i]): int(chg_counts_np[i]) for i in range(N_CHARGES)},
        "covalent_bond_counts": {FLOWMOL_BOND_TYPES[i]: int(bond_counts_np[i]) for i in range(1, N_BOND_TYPES)},
        **(extra_meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def _extract_reused_v1_shard(v1_dir: Path, source_name: str, out_shard_dir: Path) -> bool:
    """Extract and remap a source slice (e.g. `geom_drugs`) from an existing Flat-CSR corpus to full-118 vocab."""
    from loom.flowmol.constants import LEGACY_6_FORMAL_CHARGES

    meta_p = v1_dir / "meta.json"
    if not meta_p.exists():
        return False
    v1_meta = json.loads(meta_p.read_text())
    src_names = v1_meta.get("source_names", [])
    if source_name not in src_names:
        return False
    sid = src_names.index(source_name)

    src_arr = np.load(v1_dir / "source_id.npy")
    mol_ids = np.nonzero(src_arr == sid)[0]
    if len(mol_ids) == 0:
        return False
    # Fast contiguous slice check (sources are stored contiguously by build_flowmol_corpus)
    is_contig = bool(mol_ids[-1] - mol_ids[0] + 1 == len(mol_ids))
    if not is_contig:
        return False

    m0, m1 = int(mol_ids[0]), int(mol_ids[-1]) + 1
    ap = np.load(v1_dir / "atom_ptr.npy")[m0 : m1 + 1].copy()
    bp = np.load(v1_dir / "bond_ptr.npy")[m0 : m1 + 1].copy()
    a0, a1 = int(ap[0]), int(ap[-1])
    b0, b1 = int(bp[0]), int(bp[-1])
    ap -= a0
    bp -= b0

    na = np.load(v1_dir / "n_atoms.npy")[m0:m1].copy()
    sp = np.load(v1_dir / "split.npy")[m0:m1].copy()
    pos = np.load(v1_dir / "pos.npy", mmap_mode="r")[a0:a1].copy()
    raw_at = np.load(v1_dir / "atom_type.npy", mmap_mode="r")[a0:a1]
    raw_chg = np.load(v1_dir / "charge.npy", mmap_mode="r")[a0:a1]
    bu = np.load(v1_dir / "bond_u.npy", mmap_mode="r")[b0:b1].copy()
    bv = np.load(v1_dir / "bond_v.npy", mmap_mode="r")[b0:b1].copy()
    bt = np.load(v1_dir / "bond_type.npy", mmap_mode="r")[b0:b1].copy()

    old_atoms = tuple(v1_meta.get("atom_types", FLOWMOL_ATOM_TYPES))
    if old_atoms != FLOWMOL_ATOM_TYPES:
        alut = np.array([ATOM_TO_IDX[s] for s in old_atoms], dtype=np.uint8)
        at = alut[raw_at]
    else:
        at = raw_at.copy()

    old_chgs = tuple(int(q) for q in v1_meta.get("formal_charges", LEGACY_6_FORMAL_CHARGES if int(v1_meta.get("n_charges", 11)) == 6 else (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)))
    al_q = np.array([CHARGE_TO_IDX[q] for q in old_chgs], dtype=np.uint8)
    chg = al_q[raw_chg]

    out_shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_shard_dir / "atom_ptr.npy", ap)
    np.save(out_shard_dir / "bond_ptr.npy", bp)
    np.save(out_shard_dir / "n_atoms.npy", na)
    np.save(out_shard_dir / "split.npy", sp)
    np.save(out_shard_dir / "source_id.npy", np.zeros(len(na), dtype=np.uint8))
    np.save(out_shard_dir / "pos.npy", pos)
    np.save(out_shard_dir / "atom_type.npy", at)
    np.save(out_shard_dir / "charge.npy", chg)
    np.save(out_shard_dir / "bond_u.npy", bu)
    np.save(out_shard_dir / "bond_v.npy", bv)
    np.save(out_shard_dir / "bond_type.npy", bt)
    (out_shard_dir / "meta.json").write_text(json.dumps({"reused_from": str(v1_dir), "n_mols": int(len(na))}) + "\n")
    print(f"[build_corpus] reused & 118-remapped {len(na):,} '{source_name}' conformers from {v1_dir}", flush=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Build unified multi-source Flat-CSR 3D corpus for FlowMol.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("/dev/shm/flowmol_corpus_work"))
    p.add_argument("--sources", type=str, default="qm9,oe62,pcqm4mv2,geom_drugs,tmqm")
    p.add_argument("--reuse-corpus-dir", type=str, default="", help="Optional existing Flat-CSR corpus to reuse slow sources from")
    p.add_argument("--reuse-sources", type=str, default="geom_drugs", help="Comma-separated sources to reuse if present in --reuse-corpus-dir")
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--geom-max-confs", type=int, default=5)
    p.add_argument("--max-mols-per-source", type=int, default=0, help="0 = unlimited (full build)")
    p.add_argument("--workers", type=int, default=min(44, max(4, (os.cpu_count() or 8) - 2)))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assert args.max_atoms <= 255, "uint8 bond indices require max_atoms <= 255"
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    reuse_dir = Path(args.reuse_corpus_dir) if args.reuse_corpus_dir.strip() else None
    reuse_set = {s.strip().lower() for s in args.reuse_sources.split(",") if s.strip()} if (reuse_dir and reuse_dir.exists()) else set()
    if reuse_dir is not None and (reuse_dir / "meta.json").exists():
        existing_srcs = set(json.loads((reuse_dir / "meta.json").read_text()).get("source_names", []))
        reuse_set &= existing_srcs
    else:
        reuse_set = set()

    max_mols = args.max_mols_per_source if args.max_mols_per_source > 0 else None
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_corpus] sources={sources} reuse={sorted(reuse_set)} max_atoms={args.max_atoms} workers={args.workers} work={args.work_dir}", flush=True)

    # Pre-download all non-reused source files concurrently across threads so slow servers overlap!
    t_dl = time.perf_counter()
    dl_jobs: List[Tuple[Sequence[str], Path, int, int]] = []
    if "qm9" in sources and "qm9" not in reuse_set:
        dl_jobs.append((URLS["qm9"], args.work_dir / "qm9.zip", 10_000_000, 4))
    if "oe62" in sources and "oe62" not in reuse_set:
        dl_jobs.append((URLS["oe62"], args.work_dir / "df_62k.json", 100_000_000, 8))
    if "pcqm4mv2" in sources and "pcqm4mv2" not in reuse_set:
        dl_jobs.append((URLS["pcqm4mv2"], args.work_dir / "pcqm4m-v2-train.sdf.tar.gz", 500_000_000, 8))
    if "geom_drugs" in sources and "geom_drugs" not in reuse_set:
        dl_jobs.append((URLS["geom_drugs_train"], args.work_dir / "geom_train_data.pickle", 100_000_000, 12))
        dl_jobs.append((URLS["geom_drugs_val"], args.work_dir / "geom_val_data.pickle", 100_000_000, 8))
        dl_jobs.append((URLS["geom_drugs_test"], args.work_dir / "geom_test_data.pickle", 100_000_000, 8))
    if "tmqm" in sources and "tmqm" not in reuse_set:
        for k, fn in (
            ("tmqm_x1_xyz", "tmQM_X1.xyz.gz"),
            ("tmqm_x2_xyz", "tmQM_X2.xyz.gz"),
            ("tmqm_x3_xyz", "tmQM_X3.xyz.gz"),
            ("tmqm_x1_bo", "tmQM_X1.BO.gz"),
            ("tmqm_x2_bo", "tmQM_X2.BO.gz"),
            ("tmqm_x3_bo", "tmQM_X3.BO.gz"),
        ):
            dl_jobs.append((URLS[k], args.work_dir / fn, 1_000_000, 4))

    if dl_jobs:
        with ThreadPoolExecutor(max_workers=len(dl_jobs)) as tex:
            futs = [tex.submit(_download_file, u, d, min_bytes=mb, n_conns=nc) for u, d, mb, nc in dl_jobs]
            for f in futs:
                f.result()
        print(f"[build_corpus] all downloads finished in {time.perf_counter() - t_dl:.1f}s", flush=True)

    shard_dirs: List[Path] = []
    used_sources: List[str] = []

    t_start = time.perf_counter()
    for sname in sources:
        sdir = args.work_dir / f"shard_{sname}_max{args.max_atoms}"
        if (sdir / "meta.json").exists():
            print(f"[build_corpus] reusing cached parsed shard {sdir.name}", flush=True)
            shard_dirs.append(sdir)
            used_sources.append(sname)
            continue
        if sname in reuse_set and reuse_dir is not None and _extract_reused_v1_shard(reuse_dir, sname, sdir):
            shard_dirs.append(sdir)
            used_sources.append(sname)
            continue

        if sname == "qm9":
            recs, sp = process_qm9(args.work_dir, max_atoms=args.max_atoms, workers=args.workers, seed=args.seed, max_mols=max_mols)
        elif sname == "oe62":
            recs, sp = process_oe62(args.work_dir, max_atoms=args.max_atoms, workers=args.workers, seed=args.seed, max_mols=max_mols)
        elif sname == "pcqm4mv2":
            recs, sp = process_pcqm4mv2(args.work_dir, max_atoms=args.max_atoms, workers=args.workers, seed=args.seed, max_mols=max_mols)
        elif sname == "geom_drugs":
            recs, sp = process_geom_drugs(
                args.work_dir,
                max_atoms=args.max_atoms,
                max_confs_per_mol=args.geom_max_confs,
                workers=args.workers,
                max_mols=max_mols,
            )
        elif sname == "tmqm":
            recs, sp = process_tmqm(
                args.work_dir,
                max_atoms=args.max_atoms,
                workers=args.workers,
                seed=args.seed,
                max_mols=max_mols,
            )
        else:
            raise ValueError(f"Unknown source: {sname}")

        write_flat_corpus(
            sdir,
            recs,
            splits=sp,
            source_ids=[0] * len(recs),
            source_names=[sname],
        )
        del recs, sp
        gc.collect()
        shard_dirs.append(sdir)
        used_sources.append(sname)

    meta = _merge_flat_shards(
        args.output_dir,
        shard_dirs,
        used_sources,
        extra_meta={
            "max_atoms_filter": args.max_atoms,
            "geom_max_confs": args.geom_max_confs,
            "build_elapsed_sec": round(time.perf_counter() - t_start, 2),
        },
    )
    summary_meta = {
        k: (
            {ek: ev for ek, ev in v.items() if int(ev) > 0}
            if k == "element_counts" and isinstance(v, dict)
            else v
        )
        for k, v in meta.items()
        if k not in ("n_atoms_hist", "per_source_n_atoms_hist")
    }
    print(f"[build_corpus] DONE -> {args.output_dir}", flush=True)
    print(json.dumps(summary_meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
