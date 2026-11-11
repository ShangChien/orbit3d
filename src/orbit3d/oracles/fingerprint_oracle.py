"""Bootstrap fingerprint baseline with physical-unit targets and exact black-box inference."""

from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from .base import PropertyOracleBase


class FingerprintOracle(PropertyOracleBase):
    has_3d_unimol = False

    def __init__(self, target_cols=("homo", "lumo"), n_bits=2048):
        super().__init__()
        self.target_cols = list(target_cols)
        self.output_dim = len(self.target_cols)
        self.n_bits = n_bits
        self.models = []
        self.training_elements = None
        self.fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=True)

    def features(self, mols):
        rows = []
        for mol in mols:
            if mol is None:
                raise ValueError("Invalid molecule for fingerprint oracle")
            mol = Chem.RemoveHs(Chem.Mol(mol))
            Chem.SanitizeMol(mol)
            bits = self.fp.GetFingerprintAsNumPy(mol).astype(np.float32)
            desc = np.array(
                [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    mol.GetNumAtoms(),
                    Chem.GetFormalCharge(mol),
                ],
                dtype=np.float32,
            )
            rows.append(np.concatenate([bits, desc]))
        if not rows:
            return np.empty((0, self.n_bits + 8), dtype=np.float32)
        result = np.stack(rows)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite features")
        return result

    def fit(self, mols, targets, save_dir, seed=42, n_models=3, n_trees=128, n_jobs=4):
        x = self.features(mols)
        y = np.asarray(targets, dtype=np.float64).reshape(len(mols), self.output_dim)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite targets")
        self.training_elements = sorted({atom.GetSymbol() for mol in mols
                                         for atom in Chem.AddHs(mol).GetAtoms()})
        self.training_mean = y.mean(0)
        self.training_std = y.std(0).clip(1e-6)
        self.models = []
        for index in range(n_models):
            rng = np.random.default_rng(seed + index)
            indices = rng.integers(0, len(mols), size=len(mols))
            model = ExtraTreesRegressor(
                n_estimators=n_trees,
                max_depth=28,
                min_samples_leaf=2,
                max_features=0.5,
                n_jobs=n_jobs,
                random_state=seed + index,
            )
            model.fit(x[indices], y[indices])
            self.models.append(model)
            print(f"Bootstrap model {index + 1}/{n_models} fitted", flush=True)
        self.save(save_dir)

    def predict_features(self, x):
        if not self.models:
            raise ValueError("Oracle is not trained")
        # Fix tree accumulation order across reloads and thread scheduling.
        for model in self.models:
            model.n_jobs = 1
        members = np.stack([m.predict(x).reshape(len(x), self.output_dim) for m in self.models])
        return members.mean(0), members.std(0)

    def predict_3d_mols(self, mols):
        mean, std = self.predict_features(self.features(mols))
        return (mean[:, 0], std[:, 0]) if self.output_dim == 1 else (mean, std)

    def forward_differentiable_3d(self, *args, **kwargs):
        raise NotImplementedError("Fingerprint oracle is a black-box graph predictor")

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "ensemble.joblib"
        joblib.dump(self.models, path, compress=3)
        metadata = {
            "oracle_type": "fingerprint_bootstrap_v1",
            "target_cols": self.target_cols,
            "units": ["eV"] * self.output_dim,
            "n_bits": self.n_bits,
            "training_elements": self.training_elements,
            "training_mean": self.training_mean.tolist(),
            "training_std": self.training_std.tolist(),
            "uncertainty": "bootstrap disagreement; not calibrated probability",
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load_from_dir(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        path = directory / "ensemble.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise ValueError("Oracle checksum mismatch")
        oracle = cls(metadata["target_cols"], metadata["n_bits"])
        # Load only artifacts produced by this trusted training workflow.
        oracle.models = joblib.load(path)
        oracle.training_elements = metadata.get("training_elements")
        oracle.training_mean = np.array(metadata["training_mean"])
        oracle.training_std = np.array(metadata["training_std"])
        return oracle
