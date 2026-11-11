"""Overfit a tiny synthetic chemistry set and sample: integration, not a benchmark."""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from orbit3d.constraints import TargetIntervals
from orbit3d.generator.octree_tokenizer import (
    rdkit_mol_to_mol3d_instance,
    encode_mol_to_octree_tokens,
    decode_tokens_to_mol3d,
    mol3d_to_validated_rdkit_mol,
    mol_to_clean_smiles,
)
from orbit3d.generator.uni3dar_model import ConditionalUni3DAR

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(42)
started = time.perf_counter()
mols = []
for smi in ["C", "CC", "CO", "CN", "C=O", "C#N"]:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(m, randomSeed=42) == 0
    mols.append(rdkit_mol_to_mol3d_instance(m, np.array([m.GetNumAtoms()], dtype=np.float32)))
y = np.array([m.props for m in mols])
mean = y.mean(0)
scale = y.std(0)
seqs = [encode_mol_to_octree_tokens(m, max_atoms=16) for m in mols]
tokens = torch.zeros(len(mols), max(map(len, seqs)), dtype=torch.long)
conds = []
for i, (m, seq) in enumerate(zip(mols, seqs)):
    tokens[i, : len(seq)] = torch.tensor(seq)
    conds.append(
        TargetIntervals.parse((float(m.props[0]) - 0.1, float(m.props[0]) + 0.1), ["atoms"]).condition(
            mean, scale
        )
    )
cond = torch.tensor(np.stack(conds))
labels = torch.tensor((y - mean) / scale)
model = ConditionalUni3DAR(
    d_model=64, n_heads=4, n_layers=2, cond_dim=4, output_dim=1, max_atoms=16, max_seq_len=66
)
opt = torch.optim.AdamW(model.parameters(), lr=0.003)
for step in range(600):
    model.train()
    logits, _ = model(tokens[:, :-1], cond)
    ce = F.cross_entropy(
        model.constrained_logits(logits).flatten(0, 1), tokens[:, 1:].flatten(), ignore_index=0
    )
    _, pred = model(tokens, torch.zeros_like(cond))
    loss = ce + 0.1 * F.smooth_l1_loss(pred, labels)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()
    if step % 100 == 0:
        print(step, float(ce.detach()), flush=True)
    if ce < 0.02:
        break
model.eval()
weights = out / "synthetic_generator.pt"
torch.save(model.state_dict(), weights)
records = []
with torch.no_grad():
    for i in range(24):
        torch.manual_seed(2026 + i)
        idx = i % len(mols)
        seq, _, _ = model.sample_autoregressive(cond[idx : idx + 1], temperature=0.65, top_p=0.9)
        row = {"sample_id": i, "requested_atoms": int(y[idx, 0]), "tokens": seq[0].tolist()}
        try:
            inst = decode_tokens_to_mol3d(row["tokens"])
            m = mol3d_to_validated_rdkit_mol(inst)
            row.update(
                valid=True,
                smiles=mol_to_clean_smiles(m),
                atoms=m.GetNumAtoms(),
                hit=m.GetNumAtoms() == int(y[idx, 0]),
            )
        except ValueError as exc:
            row.update(valid=False, hit=False, reason=str(exc))
        records.append(row)
report = {
    "scope": "synthetic overfit integration only; no held-out or quantum-property claim",
    "steps": step + 1,
    "teacher_forced_ce": float(ce.detach()),
    "attempted": len(records),
    "valid": sum(r["valid"] for r in records),
    "joint_valid_atom_count_hit": sum(r["valid"] and r["hit"] for r in records) / len(records),
    "elapsed_seconds": time.perf_counter() - started,
    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    "records": records,
}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k != "records"})
