"""Check the adapted input tensors against pinned upstream feature/padding code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from orbit3d.oracles.unimol2_oracle import source_features, collate_unimol2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.upstream / 'unimol2/unimol2/data/unimol2_dataset.py'
    names = {'get_graph_features', 'convert_to_single_emb', 'floyd_warshall',
             'pad_1d', 'pad_1d_feat', 'pad_2d', 'pad_2d_feat', 'pad_attn_bias'}
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []  # Run the same Floyd-Warshall body without numba.
            nodes.append(node)
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    pad_names = {'src_token': 'pad_1d', 'src_pos': 'pad_1d_feat',
                 'atom_feat': 'pad_1d_feat', 'atom_mask': 'pad_1d',
                 'edge_feat': 'pad_2d_feat', 'shortest_path': 'pad_2d',
                 'degree': 'pad_1d', 'pair_type': 'pad_2d_feat',
                 'attn_bias': 'pad_attn_bias'}
    features = []
    # Synthetic coordinates deliberately avoid invoking any conformer generator.
    for smiles in ['CO', 'CCO', 'c1ccccc1', 'CC(=O)NC', 'B(O)O']:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetPositions(np.random.default_rng(42).normal(size=(mol.GetNumAtoms(), 3)))
        mol.AddConformer(conf)
        actual = source_features(mol)
        n = len(actual['src_token'])
        # With drop_feat=True every graph-dependent tensor is overwritten by
        # upstream. Empty edges therefore exercise that official contract.
        expected = namespace['get_graph_features'](
            np.zeros((0, 3), dtype=np.int32), np.zeros((2, 0), dtype=np.int32),
            np.zeros((n, 9), dtype=np.int32), True)
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        features.append(actual)
    actual, _ = collate_unimol2([(f, np.zeros(2)) for f in features])
    padded = (max(len(f['src_token']) for f in features) + 4) // 4 * 4 - 1
    for key, function in pad_names.items():
        expected = namespace[function]([f[key] for f in features], padded)
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)

    vendored = Path('src/orbit3d/compat/unimol2')
    source_checks = {}
    def without_imports(file):
        tree = ast.parse(file.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        return ast.dump(tree, include_attributes=False)
    for name in ['unimol2.py', 'layers.py', 'transformer_encoder_with_pair.py']:
        upstream = args.upstream / 'unimol2/unimol2/models' / name
        assert without_imports(upstream) == without_imports(vendored/name), name
        source_checks[name] = hashlib.sha256(upstream.read_bytes()).hexdigest()
    result = {
        'upstream_commit': '90f52c41299a1a582da0f9765e9f87aa21faa16a',
        'graph_features_and_padding': 'exact equality against upstream functions',
        'model_source': 'AST identical apart from import redirection',
        'models_original_sha256': source_checks,
        'input_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'molecules': len(features),
        'scope': 'Input and model-source verification; does not claim bitwise parity with Uni-Core fused CUDA kernels.',
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
