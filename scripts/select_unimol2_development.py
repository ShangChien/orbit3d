"""Choose the frozen Uni-Mol2 evaluator using development labels only."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from orbit3d.oracles.unimol2_oracle import InternalUniMol2Oracle
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes
from profile_dual_oracle import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+', default=['unimol2_310m', 'unimol2_11b'])
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2', '3']
    torch.set_num_threads(2)
    freeze_run(args.output, __file__)
    sdf = args.root / 'data/valid.sdf'
    mols = list(Chem.SDMolSupplier(str(sdf), removeHs=False))
    labels = np.array([[float(m.GetProp(k)) for k in ['homo', 'lumo']] for m in mols])
    results = {}
    files = {'development': sdf, 'manifest': args.root/'data/manifest.json'}
    for name in args.candidates:
        directory = args.root/'training'/name/'oracle'
        assert (directory.parent/'completed.json').exists(), name
        files[name] = directory
        meta = json.loads((directory/'oracle_meta.json').read_text())
        model = InternalUniMol2Oracle.load_from_dir(directory)
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction, _ = model.predict_3d_mols(mols, batch_size=16)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        with torch.autocast('cuda', dtype=torch.bfloat16):
            low_precision, _ = model.predict_3d_mols(mols, batch_size=16)
        delta = low_precision-prediction
        result = {
            'model_size': meta['model_size'],
            'checkpoint_sha256': hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),
            'metrics': metrics(labels, prediction),
            'normalized_mae': float(np.mean(abs(prediction-labels)/np.array(meta['scale']))),
            'fp32_seconds': seconds,
            'bf16_fp32_mean_absolute_difference_eV': abs(delta).mean(0).tolist(),
            'bf16_fp32_max_absolute_difference_eV': abs(delta).max(0).tolist(),
            'bf16_metrics': metrics(labels, low_precision),
            'best_epoch': meta['best_epoch'],
            'best_kind': meta['best_kind'],
        }
        results[name] = result
        np.savez_compressed(args.output/f'{name}_development.npz', labels=labels,
                            fp32=prediction, bf16=low_precision,
                            source_ids=[m.GetProp('SourceID') for m in mols])
        print(name, json.dumps(result), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    selected = min(results, key=lambda name: results[name]['normalized_mae'])
    result = {
        'selected_unimol': selected,
        'selection_rule': 'Minimum FP32 normalized HOMO/LUMO MAE on the 512 fixed development molecules across predefined Uni-Mol2 sizes; no test or generated-candidate scores used.',
        'inference_precision': 'FP32',
        'candidates': results,
        'hashes': record_input_hashes(args.output, files),
    }
    (args.output/'selection.json').write_text(json.dumps(result, indent=2)+'\n')
    print('SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
