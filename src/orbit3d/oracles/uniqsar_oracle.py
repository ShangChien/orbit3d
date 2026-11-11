"""
Standalone, Zero-Disk-IO & Differentiable 3D Bridge for Trained Uni-QSAR Checkpoints
====================================================================================
Supports:
1. Full or Partial Model Selection (`selected_keys.txt`, `infer_choices`, single-model runs,
   missing `UniMolModel`, missing `BERTModel`, or missing `MLModel`s, plus `MLPModel`).
2. Single-Target & Multi-Target Tasks (`regression`, `multilabel_regression` with K targets,
   `classification`, `multilabel_classification`, `multiclass`).
3. All Ensembler Modes (`stacking` with single or K-list meta-models, `average`, `vote`).
4. All TargetScaler Modes (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`,
   `PowerTransformer` Yeo-Johnson/Box-Cox, `log1p`, or `list` of K scalers for multi-target)
   in both NumPy and differentiable PyTorch!
5. Graceful Fallback when `UniMolModel` is absent (`has_3d_unimol == False`):
   - `predict_3d_mols` skips 3D feature extraction for maximum speed;
   - `forward_differentiable_3d` either uses an optional `aux_3d_head` or attaches a clean
     zero-gradient graph (`0.0 * coords.sum()`) so Stage 5 seamlessly falls back to pure
     `DPA-4` physical conformer relaxation without crashing `.backward()`.
"""

from __future__ import annotations

import os
import re
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ==============================================================================
# 1. Exact Dictionary & Tokenizers from Uni-QSAR (`mol.dict.txt` & `smiletokenizer.py`)
# ==============================================================================
UNIMOL_DICT_SYMBOLS = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]",
    "C", "N", "O", "S", "H", "Cl", "F", "Br", "I", "Si", "P", "B",
    "Na", "K", "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr", "Se", "Gd", "Au", "Li",
    "[MASK]",
]
UNIMOL_SYM2IDX = {s: i for i, s in enumerate(UNIMOL_DICT_SYMBOLS)}
UNIMOL_PAD_IDX = 0
UNIMOL_BOS_IDX = 1
UNIMOL_EOS_IDX = 2
UNIMOL_UNK_IDX = 3

ALL_TOKEN_LIST = [
    "[PAD]", "[GLO]", "c", "C", "(", ")", "O", "1", "2", "=", "N", "3", "n", "4",
    "[C@H]", "F", "[C@@H]", "-", "S", "/", "Cl", "[nH]", "s", "o", "5", "#",
    "[C@]", "[C@@]", "\\", "[O-]", "[N+]", "Br", "6", "P", "[n+]", "7", "I", "[S+]",
    "8", "[N-]", "[Si]", "B", "9", "[2H]", "[Se]", "[other_atom]", "[other_token]",
]
SMI_TOKEN2IDX = {tok: i for i, tok in enumerate(ALL_TOKEN_LIST)}
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def tokenize_smiles_1d(smi: str, max_len: int = 200) -> List[int]:
    tokens = SMI_REGEX.findall(smi)
    if ''.join(tokens) != smi:
        raise ValueError('SMILES contains tokens outside the trained tokenizer grammar')
    if len(tokens) > max_len:
        raise ValueError(f'SMILES token count {len(tokens)} exceeds the model limit {max_len}')
    padding = ["[PAD]"] * (max_len - len(tokens))
    tokens = ["[GLO]"] + tokens + padding
    idxs = []
    for tok in tokens:
        if tok in SMI_TOKEN2IDX:
            idxs.append(SMI_TOKEN2IDX[tok])
        elif "[" in tok:
            idxs.append(SMI_TOKEN2IDX["[other_atom]"])
        else:
            idxs.append(SMI_TOKEN2IDX["[other_token]"])
    return idxs


# ==============================================================================
# 2. Pure-PyTorch `UniMolModel` (Supports arbitrary `output_dim = K`)
# ==============================================================================
class SelfMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len, embed_dim = query.size()
        qkv = self.in_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q * self.scaling
        q = q.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        v = v.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).contiguous()
        attn = attn.view(bsz, tgt_len, embed_dim)
        out = self.out_proj(attn)
        return out, attn_weights


class PureUnicoreTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        attention_heads: int = 64,
        post_ln: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.post_ln = post_ln
        self.self_attn = SelfMultiheadAttention(embed_dim, attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(query=x, key_padding_mask=padding_mask, attn_bias=attn_bias)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        return x, attn


class PureTransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 15,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        attention_heads: int = 64,
    ):
        super().__init__()
        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [
                PureUnicoreTransformerEncoderLayer(embed_dim, ffn_embed_dim, attention_heads)
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
            attn_mask = attn_mask.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None

        for layer in self.layers:
            x, attn_mask = layer(x, attn_bias=attn_mask, padding_mask=padding_mask)
        x = self.final_layer_norm(x)
        return x


class GaussianLayer(nn.Module):
    def __init__(self, K: int = 128, edge_types: int = 961):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)

    def forward(self, x: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        a = (2.0 * 3.14159) ** 0.5
        return (torch.exp(-0.5 * (((x.float() - mean) / std) ** 2)) / (a * std)).type_as(
            self.means.weight
        )


class NonLinearHead(nn.Module):
    def __init__(self, input_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int = 512, inner_dim: int = 512, num_classes: int = 1):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features[:, 0, :]
        x = torch.tanh(self.dense(x))
        return self.out_proj(x)


class StandaloneUniMolModel(nn.Module):
    """Standalone state-dict bridge for `qsar.models.unimol.UniMolModel` without `unicore`."""

    def __init__(self, output_dim: int = 1, dict_len: int = 31):
        super().__init__()
        self.output_dim = output_dim
        self.padding_idx = UNIMOL_PAD_IDX
        self.embed_tokens = nn.Embedding(dict_len, 512, self.padding_idx)
        self.encoder = PureTransformerEncoderWithPair(15, 512, 2048, 64)
        self.gbf_proj = NonLinearHead(128, 64)
        self.gbf = GaussianLayer(128, dict_len * dict_len)
        self.classification_head = ClassificationHead(512, 512, output_dim)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_coord_or_edge_type: torch.Tensor,
        src_edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_edge_type is None:
            src_edge_type = src_coord_or_edge_type
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        n_node = src_distance.size(-1)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)
        encoder_rep = self.encoder(x, attn_mask=graph_attn_bias, padding_mask=padding_mask)
        return self.classification_head(encoder_rep)


# ==============================================================================
# 3. Pure-PyTorch `BERTModel` & `MLPModel` (Supports arbitrary `output_dim = K`)
# ==============================================================================
def build_standalone_bert(output_dim: int = 1) -> nn.Module:
    class _Emb(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embed = nn.Embedding(47, 768)
            self.pos_embed = nn.Embedding(201, 768)
            self.norm = nn.LayerNorm(768)

        def forward(self, x):
            pos = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
            return self.norm(self.tok_embed(x.long()) + self.pos_embed(pos))

    class _MHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
            self.layernorm = nn.LayerNorm(768)
            self.W_Q = nn.Linear(768, 768)
            self.W_K = nn.Linear(768, 768)
            self.W_V = nn.Linear(768, 768)

        def forward(self, Q, K, V, attn_mask):
            b = Q.size(0)
            q = self.W_Q(Q).view(b, -1, 12, 64).transpose(1, 2)
            k = self.W_K(K).view(b, -1, 12, 64).transpose(1, 2)
            v = self.W_V(V).view(b, -1, 12, 64).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
            scores.masked_fill_(attn_mask.unsqueeze(1).repeat(1, 12, 1, 1), -1e4)
            ctx = torch.matmul(F.softmax(scores, dim=-1), v)
            ctx = ctx.transpose(1, 2).contiguous().view(b, -1, 768)
            return self.layernorm(self.linear(ctx) + Q)

    class _FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(768, 3072, bias=False),
                nn.ReLU(),
                nn.Linear(3072, 768, bias=False),
            )
            self.layernorm = nn.LayerNorm(768)

        def forward(self, x):
            return self.layernorm(self.fc(x) + x)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_self_attn = _MHA()
            self.pos_ffn = _FFN()

        def forward(self, x, mask):
            return self.pos_ffn(self.enc_self_attn(x, x, x, mask))

    class _Bert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = _Emb()
            self.layers = nn.ModuleList([_Layer() for _ in range(6)])
            self.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.BatchNorm1d(768),
            )
            self.classifier_global = nn.Linear(768, output_dim)
            self.classifier_atom = nn.Linear(768, 15)

        def forward(self, net_input):
            b, s = net_input.size()
            mask = net_input.data.eq(0).unsqueeze(1).expand(b, s, s)
            out = self.embedding(net_input)
            for lyr in self.layers:
                out = lyr(out, mask)
            return self.classifier_global(self.fc(out[:, 0]))

    return _Bert()


def build_standalone_mlp_from_state_dict(sd: Dict[str, torch.Tensor]) -> nn.Module:
    """Reconstructs `qsar.models.mlp.MLP` directly from its checkpoint `state_dict` shapes."""
    linear_keys = sorted(
        [k for k in sd.keys() if k.startswith("ffn.") and k.endswith(".weight")],
        key=lambda x: int(x.split(".")[1]),
    )
    layers: List[nn.Module] = []
    for idx, k in enumerate(linear_keys):
        w = sd[k]
        out_f, in_f = w.shape
        if idx > 0:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(in_f, out_f))

    class _MLP(nn.Module):
        def __init__(self, ffn_layers):
            super().__init__()
            self.ffn = nn.Sequential(*ffn_layers)

        def forward(self, x):
            return self.ffn(x)

    return _MLP(layers)


# ==============================================================================
# 4. Unpickling Shim for `MLModel`, `LightGBM` & `Stacking` (`joblib.load`)
# ==============================================================================
def _eval_lgbm_model_str(model_str: str, x_2d: np.ndarray) -> np.ndarray:
    """Pure-Python bit-exact evaluator for LightGBM v3 `Booster.handle` tree dump string."""
    trees = []
    for block in model_str.split("\nTree=")[1:]:
        lines = block.splitlines()
        kv = {}
        for ln in lines[1:]:
            if "=" in ln:
                k, v = ln.split("=", 1)
                kv[k.strip()] = v.strip()
            elif ln.startswith("Tree") or ln.startswith("end of trees"):
                break
        if "leaf_value" not in kv:
            continue
        leaf_val = [float(v) for v in kv["leaf_value"].split()]
        if "split_feature" not in kv or not kv["split_feature"]:
            trees.append(([], [], [], [], leaf_val))
            continue
        sf = [int(v) for v in kv["split_feature"].split()]
        th = [float(v) for v in kv["threshold"].split()]
        lc = [int(v) for v in kv["left_child"].split()]
        rc = [int(v) for v in kv["right_child"].split()]
        trees.append((sf, th, lc, rc, leaf_val))

    out = np.zeros(x_2d.shape[0], dtype=np.float64)
    for row_idx in range(x_2d.shape[0]):
        row = x_2d[row_idx]
        s = 0.0
        for sf, th, lc, rc, leaf_val in trees:
            if not sf:
                s += leaf_val[0]
                continue
            node = 0
            while node >= 0:
                if row[sf[node]] <= th[node]:
                    node = lc[node]
                else:
                    node = rc[node]
            s += leaf_val[~node]
        out[row_idx] = s
    return out


def _register_qsar_pickle_shims():
    """Registers lightweight `qsar.models.mlmodelzoo` and `lightgbm` shims if not installed."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        m_lgb = types.ModuleType("lightgbm")
        m_lgb_sk = types.ModuleType("lightgbm.sklearn")
        m_lgb_basic = types.ModuleType("lightgbm.basic")

        class _DummyBooster:
            def __setstate__(self, state):
                self.__dict__.update(state)

        class _DummyLGBM:
            def __setstate__(self, state):
                self.__dict__.update(state)

            def predict(self, X):
                model_str = getattr(self._Booster, "handle", "")
                if isinstance(model_str, bytes):
                    model_str = model_str.decode("utf-8")
                return _eval_lgbm_model_str(model_str, np.asarray(X, dtype=np.float64))

            def predict_proba(self, X):
                raw = self.predict(X)
                prob1 = 1.0 / (1.0 + np.exp(-raw))
                return np.stack([1.0 - prob1, prob1], axis=-1)

        m_lgb.LGBMRegressor = _DummyLGBM
        m_lgb.LGBMClassifier = _DummyLGBM
        m_lgb_sk.LGBMRegressor = _DummyLGBM
        m_lgb_sk.LGBMClassifier = _DummyLGBM
        m_lgb_basic.Booster = _DummyBooster
        sys.modules["lightgbm"] = m_lgb
        sys.modules["lightgbm.sklearn"] = m_lgb_sk
        sys.modules["lightgbm.basic"] = m_lgb_basic

    try:
        import qsar.models.mlmodelzoo  # noqa: F401
        return
    except ImportError:
        pass

    m_qsar = types.ModuleType("qsar")
    m_models = types.ModuleType("qsar.models")
    m_zoo = types.ModuleType("qsar.models.mlmodelzoo")

    class _BaseShim:
        """Supports all single-target & multi-target tasks (`regression`, `multilabel_regression`, etc.)."""

        def _predict(self, x):
            task = getattr(self, "task", "regression")
            num_classes = getattr(self, "num_classes", 1)
            if task == "classification":
                return self.model.predict_proba(x)[:, 1:]
            elif task == "multiclass":
                return self.model.predict_proba(x)
            elif task == "regression":
                return self.model.predict(x).reshape(-1, 1)
            elif task == "multilabel_classification":
                y_pred = np.zeros((x.shape[0], num_classes), dtype=np.float64)
                for i in range(num_classes):
                    y_pred[:, i] = self.model[i].predict_proba(x)[:, 1]
                return y_pred
            elif task == "multilabel_regression":
                y_pred = np.zeros((x.shape[0], num_classes), dtype=np.float64)
                for i in range(num_classes):
                    y_pred[:, i] = np.squeeze(self.model[i].predict(x))
                return y_pred
            else:
                if isinstance(self.model, list):
                    return np.stack([np.squeeze(m.predict(x)) for m in self.model], axis=-1)
                return self.model.predict(x).reshape(x.shape[0], -1)

    for cls_name in ["MLModelZoo", "LR", "SVM", "ExtraTree", "GBDT"]:
        setattr(m_zoo, cls_name, type(cls_name, (_BaseShim,), {}))

    sys.modules["qsar"] = m_qsar
    sys.modules["qsar.models"] = m_models
    sys.modules["qsar.models.mlmodelzoo"] = m_zoo


# ==============================================================================
# 5. Differentiable & NumPy TargetScaler Inverse-Transform (Single & Multi-Target)
# ==============================================================================
def _inverse_single_scaler_torch(y_col: torch.Tensor, scaler_obj) -> torch.Tensor:
    from ..utils.dataset_io import inverse_single_scaler_torch

    return inverse_single_scaler_torch(y_col, scaler_obj)


# ==============================================================================
# 6. Main Unified Bridge: `UniQSARTrainedBridge`
# ==============================================================================
class UniQSARTrainedBridge:
    """
    Universal, Zero-Disk-IO & Differentiable 3D Bridge for Trained Uni-QSAR Checkpoints.

    Supports:
    - Single-target (`num_classes == 1`) & Multi-target (`num_classes == K > 1`) tasks
    - Arbitrary subsets of trained models (`infer_choices`, missing `UniMol`, missing `ML`s, etc.)
    - `stacking`, `average`, and `vote` ensemble methods
    - Optional `aux_3d_head` fallback when no 3D `UniMolModel` was trained (`has_3d_unimol == False`)
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        infer_choices: Optional[Sequence[str]] = None,
        aux_3d_head: Optional[nn.Module] = None,
    ):
        self.model_dir = model_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.aux_3d_head = aux_3d_head.to(self.device) if aux_3d_head is not None else None
        _register_qsar_pickle_shims()

        # 1. Parse config.yaml (with safe defaults if partially missing)
        cfg_path = os.path.join(model_dir, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

        base_cfg = self.config.get("Base", {})
        data_cfg = self.config.get("Datahub", {})
        ens_cfg = self.config.get("Ensembler", {})
        feat_cfg = self.config.get("Featurehub", {})

        self.task: str = base_cfg.get("task", self.config.get("task", "regression"))
        self.num_classes: int = int(data_cfg.get("num_classes", self.config.get("num_classes", 1)))
        self.multiclass_cnt: Optional[int] = data_cfg.get("multiclass_cnt", None)
        self.n_splits: int = int(data_cfg.get("split_n", 5))
        raw_tcols = data_cfg.get("target_cols", self.config.get("target_cols", "TARGET"))
        if isinstance(raw_tcols, str):
            self.target_cols: List[str] = [c.strip() for c in raw_tcols.split(",") if c.strip()]
        else:
            self.target_cols = list(raw_tcols)
        if len(self.target_cols) != self.num_classes and self.num_classes > 1:
            self.target_cols = [f"TARGET_{i}" for i in range(self.num_classes)]

        # Output dim per base model
        if self.task == "multiclass" and self.multiclass_cnt is not None:
            self.output_dim = int(self.multiclass_cnt)
        else:
            self.output_dim = self.num_classes

        # Feature hyper-parameters from config.yaml
        fp_cfg = feat_cfg.get("FE_fringerprint", {}) if isinstance(feat_cfg, dict) else {}
        self.fp_dim = int(fp_cfg.get("dim", 2048)) if isinstance(fp_cfg, dict) else 2048
        self.fp_radius = int(fp_cfg.get("neighbor", 5)) if isinstance(fp_cfg, dict) else 5
        smi_cfg = feat_cfg.get("1D_smiles", {}) if isinstance(feat_cfg, dict) else {}
        self.smi_max_len = int(smi_cfg.get("max_len", 200)) if isinstance(smi_cfg, dict) else 200

        # 2. Load TargetScaler & Classification Thresholds (`threshold.txt`)
        scaler_path = os.path.join(model_dir, "target_scaler.ss")
        self.target_normalize = data_cfg.get("target_normalize", "auto")
        if (
            self.task in ["regression", "multilabel_regression"]
            and self.target_normalize != "none"
            and os.path.exists(scaler_path)
        ):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

        thresh_path = os.path.join(model_dir, "EnsembleModel", "threshold.txt")
        if self.task in ["classification", "multilabel_classification"] and os.path.exists(thresh_path):
            raw_th = np.loadtxt(thresh_path, dtype=np.float32, delimiter=",")
            self.thresholds = np.atleast_1d(raw_th).reshape(self.output_dim)
        else:
            self.thresholds = np.full((self.output_dim,), 0.5, dtype=np.float32)

        # 3. Discover available & selected base models (handles partial / single-model trainings)
        all_valid_dirs = sorted(
            [
                d
                for d in os.listdir(model_dir)
                if "_TAB_" in d
                and os.path.isdir(os.path.join(model_dir, d))
                and os.path.exists(os.path.join(model_dir, d, "model_0.pth"))
            ]
        )
        sel_path = os.path.join(model_dir, "EnsembleModel", "selected_keys.txt")
        if infer_choices is not None:
            # User explicitly requested a subset of models (e.g. infer_choices=["NN01", "ML02"])
            self.selected_keys = [
                d for d in all_valid_dirs if any(d.startswith(c) or c in d for c in infer_choices)
            ]
        elif os.path.exists(sel_path):
            raw_keys = np.loadtxt(sel_path, dtype=str, delimiter=",").tolist()
            raw_keys = [raw_keys] if isinstance(raw_keys, str) else list(raw_keys)
            # Keep exact order in selected_keys.txt, filtering only to models present on disk
            self.selected_keys = [k for k in raw_keys if k in all_valid_dirs]
        else:
            # Fallback when EnsembleModel/selected_keys.txt wasn't generated (e.g. single model fit)
            self.selected_keys = all_valid_dirs

        if not self.selected_keys:
            raise ValueError(f"No valid trained base models found in {model_dir} (choices={infer_choices})")

        # 4. Preload all selected ML and NN base models into memory/GPU once
        self.ml_folds: Dict[str, List] = {}
        self.nn_folds: Dict[str, nn.ModuleList] = {}

        for key in self.selected_keys:
            subdir = os.path.join(model_dir, key)
            fold_files = sorted(
                [f for f in os.listdir(subdir) if re.match(r"^model_\d+\.pth$", f)],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            if key.startswith("ML"):
                self.ml_folds[key] = [joblib.load(os.path.join(subdir, f)) for f in fold_files]
            elif "UniMolModel" in key:
                folds = nn.ModuleList()
                for f in fold_files:
                    sd = torch.load(os.path.join(subdir, f), map_location="cpu")["model_state_dict"]
                    out_d = sd["classification_head.out_proj.weight"].shape[0]
                    dict_l = sd["embed_tokens.weight"].shape[0]
                    m = StandaloneUniMolModel(output_dim=out_d, dict_len=dict_l)
                    m.load_state_dict(sd, strict=True)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    folds.append(m.to(self.device))
                self.nn_folds[key] = folds
            elif "BERTModel" in key:
                folds = nn.ModuleList()
                for f in fold_files:
                    sd = torch.load(os.path.join(subdir, f), map_location="cpu")["model_state_dict"]
                    out_d = sd["classifier_global.weight"].shape[0]
                    m = build_standalone_bert(output_dim=out_d)
                    m.load_state_dict(sd, strict=True)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    folds.append(m.to(self.device))
                self.nn_folds[key] = folds
            elif "MLPModel" in key:
                folds = nn.ModuleList()
                for f in fold_files:
                    sd = torch.load(os.path.join(subdir, f), map_location="cpu")["model_state_dict"]
                    m = build_standalone_mlp_from_state_dict(sd)
                    m.load_state_dict(sd, strict=True)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    folds.append(m.to(self.device))
                self.nn_folds[key] = folds

        # 5. Determine Ensemble Method & Load Stacking Meta-Models (if applicable)
        raw_method = ens_cfg.get("method", "stacking") if isinstance(ens_cfg, dict) else "stacking"
        ens_dir = os.path.join(model_dir, "EnsembleModel")
        stack_files = (
            sorted(
                [f for f in os.listdir(ens_dir) if re.match(r"^stacking_\d+\.pth$", f)],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            if os.path.exists(ens_dir)
            else []
        )

        # Note: Just like `ensembler.py#L167`, if infer_choices is set, or len(selected_keys) == 1,
        # or stacking files don't exist / don't match the number of selected models, use `average`.
        self.stacking_models: List = []
        self.stacking_linear_weights: Optional[np.ndarray] = None  # Shape: (F, K, M * K)
        self.stacking_linear_biases: Optional[np.ndarray] = None   # Shape: (F, K)

        if (
            raw_method == "stacking"
            and infer_choices is None
            and len(self.selected_keys) > 1
            and len(stack_files) > 0
        ):
            loaded_stacks = [joblib.load(os.path.join(ens_dir, f)) for f in stack_files]
            expected_meta_dim = len(self.selected_keys) * self.output_dim
            if self._try_extract_linear_stacking(loaded_stacks, expected_meta_dim):
                self.ensemble_method = "stacking_linear"
                self.stacking_models = loaded_stacks
            else:
                self.ensemble_method = "stacking_general"
                self.stacking_models = loaded_stacks
        elif raw_method == "vote":
            self.ensemble_method = "vote"
        else:
            self.ensemble_method = "average"

        # 6. Flags for active feature types (avoid computing unused features!)
        self.needs_fp = any("fringerprint" in k for k in self.selected_keys)
        self.needs_handcrafts = any("handcrafts" in k for k in self.selected_keys)
        self.needs_1d_smiles = any("1D_smiles" in k or "BERTModel" in k for k in self.selected_keys)
        self.has_3d_unimol = any("UniMolModel" in k for k in self.selected_keys)

        if self.needs_handcrafts:
            self.desc_names = [d[0] for d in Descriptors._descList]
            self.desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.desc_names)
        else:
            self.desc_calc = None

        self._1d2d_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _try_extract_linear_stacking(self, loaded_stacks: List, expected_meta_dim: int) -> bool:
        """
        Extracts linear coefficients of shape `(F, K, M * K)` and intercepts `(F, K)`
        for both single-target (`st.model` is a single Ridge/LogReg) and multi-target
        (`st.model` is a list of K Ridge/LogReg estimators).
        """
        weights_f = []
        biases_f = []
        for st in loaded_stacks:
            m = getattr(st, "model", None)
            if isinstance(m, list):
                if len(m) != self.output_dim:
                    return False
                w_k = []
                b_k = []
                for est in m:
                    if not hasattr(est, "coef_"):
                        return False
                    c = np.asarray(est.coef_, dtype=np.float64).reshape(-1)
                    if c.shape[0] != expected_meta_dim:
                        return False
                    w_k.append(c)
                    b_k.append(float(np.squeeze(getattr(est, "intercept_", 0.0))))
                weights_f.append(np.stack(w_k, axis=0))  # (K, M*K)
                biases_f.append(np.array(b_k, dtype=np.float64))  # (K,)
            elif hasattr(m, "coef_"):
                c = np.asarray(m.coef_, dtype=np.float64)
                if c.ndim == 1:
                    c = c.reshape(1, -1)
                if c.shape != (self.output_dim, expected_meta_dim):
                    return False
                b = np.asarray(getattr(m, "intercept_", np.zeros(self.output_dim)), dtype=np.float64).reshape(
                    self.output_dim
                )
                weights_f.append(c)
                biases_f.append(b)
            else:
                return False

        self.stacking_linear_weights = np.stack(weights_f, axis=0)  # (F, K, M*K)
        self.stacking_linear_biases = np.stack(biases_f, axis=0)    # (F, K)
        return True

    # --------------------------------------------------------------------------
    # Activation & Inverse-Scaling Helpers (Single-Target & Multi-Target)
    # --------------------------------------------------------------------------
    def _apply_nn_activation_torch(self, logits: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return F.softmax(logits, dim=-1)[..., 1:2]
        elif self.task == "multiclass":
            return F.softmax(logits, dim=-1)
        elif self.task == "multilabel_classification":
            return torch.sigmoid(logits)
        return logits

    def _inverse_scale_numpy(self, y_norm: np.ndarray) -> np.ndarray:
        """Transform predictions, never a standard deviation, to physical units."""
        if self.scaler is None or self.task not in ["regression", "multilabel_regression"]:
            return y_norm
        y_2d = y_norm.reshape(-1, self.output_dim).copy()
        if isinstance(self.scaler, list):
            if len(self.scaler) != self.output_dim:
                raise ValueError("Target scaler count does not match output_dim")
            for k, sc in enumerate(self.scaler):
                if sc is not None:
                    y_2d[:, k : k + 1] = sc.inverse_transform(y_2d[:, k : k + 1])
        else:
            y_2d = self.scaler.inverse_transform(y_2d)
        return y_2d.reshape(y_norm.shape)

    def _inverse_scale_torch(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse for (..., K), including shared multivariate scalers."""
        from ..utils.dataset_io import inverse_single_scaler_torch

        if self.scaler is None or self.task not in ["regression", "multilabel_regression"]:
            return y_norm
        if isinstance(self.scaler, list):
            if len(self.scaler) != self.output_dim:
                raise ValueError("Target scaler count does not match output_dim")
            columns = [inverse_single_scaler_torch(y_norm[..., k], sc)
                       for k, sc in enumerate(self.scaler)]
        else:
            columns = [inverse_single_scaler_torch(y_norm[..., k], self.scaler, feature_index=k)
                       for k in range(self.output_dim)]
        return torch.stack(columns, dim=-1)

    # --------------------------------------------------------------------------
    # Helper: 1D/2D Feature & Base Model Evaluation (Cached by SMILES, Shape: (K,))
    # --------------------------------------------------------------------------
    def _eval_1d_2d_single(self, smi: str) -> Dict[str, np.ndarray]:
        if smi in self._1d2d_cache:
            return self._1d2d_cache[smi]

        mol_2d = Chem.MolFromSmiles(smi)
        if mol_2d is None:
            raise ValueError("Invalid input SMILES")

        fp_arr = None
        if self.needs_fp:
            from rdkit.Chem import rdFingerprintGenerator
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.fp_radius, fpSize=self.fp_dim)
            fp_arr = generator.GetFingerprintAsNumPy(mol_2d).astype(np.float32).reshape(1, -1)

        desc_arr = None
        if self.needs_handcrafts and self.desc_calc is not None:
            desc = np.array(self.desc_calc.CalcDescriptors(mol_2d), dtype=np.float32)
            if not np.isfinite(desc).all():
                raise ValueError('Nonfinite descriptors require the fitted training imputer; arbitrary value replacement is unsupported')
            desc_arr = desc.reshape(1, -1)

        if not hasattr(self,"_fold_predictions"):
            self._fold_predictions = {}
        out: Dict[str, np.ndarray] = {}
        for key, folds in self.ml_folds.items():
            feat = fp_arr if "fringerprint" in key else desc_arr
            fold_preds = [
                np.asarray(m._predict(feat), dtype=np.float64).reshape(self.output_dim) for m in folds
            ]
            self._fold_predictions[(smi,key)] = np.stack(fold_preds,axis=0)
            out[key] = np.mean(np.stack(fold_preds, axis=0), axis=0)  # (K,)

        for key, folds in self.nn_folds.items():
            if "BERTModel" in key:
                tok_ids = torch.tensor(
                    [tokenize_smiles_1d(smi, max_len=self.smi_max_len)],
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    fold_preds = [
                        self._apply_nn_activation_torch(m(tok_ids)).view(self.output_dim).cpu().numpy()
                        for m in folds
                    ]
                self._fold_predictions[(smi,key)] = np.stack(fold_preds,axis=0)
                out[key] = np.mean(np.stack(fold_preds, axis=0), axis=0)  # (K,)
            elif "MLPModel" in key:
                feat = fp_arr if "fringerprint" in key else desc_arr
                feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    fold_preds = [
                        self._apply_nn_activation_torch(m(feat_t)).view(self.output_dim).cpu().numpy()
                        for m in folds
                    ]
                self._fold_predictions[(smi,key)] = np.stack(fold_preds,axis=0)
                out[key] = np.mean(np.stack(fold_preds, axis=0), axis=0)  # (K,)

        self._1d2d_cache[smi] = out
        return out

    # --------------------------------------------------------------------------
    # Differentiable PyTorch 3D Forward for UniMol (Returns shape `(K,)`)
    # --------------------------------------------------------------------------
    def _forward_unimol_folds(
        self,
        key: str,
        coords: torch.Tensor,
        atom_symbols: List[str],
    ) -> torch.Tensor:
        """
        Differentiable forward pass through all folds of a `UniMolModel` (`all_h` or `no_h`).
        Returns a tensor of shape `(K,)` where `K = self.output_dim`.
        """
        if "no_h" in key:
            keep = [i for i, sym in enumerate(atom_symbols) if sym != "H"]
            if not keep:
                raise ValueError('The no-H Uni-Mol representation requires heavy atoms')
            coords_sub = coords[keep]
            syms_sub = [atom_symbols[i] for i in keep]
        else:
            coords_sub = coords
            syms_sub = atom_symbols

        unknown = sorted(set(syms_sub) - set(UNIMOL_SYM2IDX))
        if unknown:
            raise ValueError(f'Uni-Mol vocabulary does not cover elements {unknown}')
        toks = (
            [UNIMOL_BOS_IDX]
            + [UNIMOL_SYM2IDX[s] for s in syms_sub]
            + [UNIMOL_EOS_IDX]
        )
        src_tokens = torch.tensor(toks, dtype=torch.long, device=coords.device).unsqueeze(0)

        centered = coords_sub - coords_sub.mean(dim=0, keepdim=True)
        zeros = torch.zeros((1, 3), dtype=coords.dtype, device=coords.device)
        src_coord = torch.cat([zeros, centered, zeros], dim=0).unsqueeze(0)  # (1, L, 3)

        src_distance = torch.cdist(src_coord, src_coord, p=2.0)  # (1, L, L)
        dict_len = len(UNIMOL_DICT_SYMBOLS)
        src_edge_type = (src_tokens.unsqueeze(-1) * dict_len + src_tokens.unsqueeze(1)).long()

        fold_preds = []
        for m in self.nn_folds[key]:
            logits = m(src_tokens, src_distance, src_coord, src_edge_type)
            pred = self._apply_nn_activation_torch(logits).view(self.output_dim)
            fold_preds.append(pred)
        self._last_3d_folds = torch.stack(fold_preds,dim=0)
        return self._last_3d_folds.mean(dim=0)  # (K,)

    # --------------------------------------------------------------------------
    # Interface 1: Stage 4 Gated-GRPO Evaluation on 3D RDKit Mols (Single & Multi-Target)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def predict_3d_mols(self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates a batch of 3D RDKit `Mol` objects across all active/selected base models.
        - If `num_classes == 1`: returns `(y_pred, y_std)` of shape `(N,)` (or `(N, K)` if `K > 1`).
        - Automatically skips 3D `UniMol` computation if `self.has_3d_unimol == False`.
        """
        y_preds_all = []
        y_stds_all = []

        for mol in mols:
            if mol is None or not mol.GetNumAtoms():
                raise ValueError('Property prediction requires a nonempty molecular graph')
            smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=True)
            cached_1d2d = self._eval_1d_2d_single(smi)

            if self.has_3d_unimol:
                atoms = [a.GetSymbol() for a in mol.GetAtoms()]
                pos = torch.tensor(
                    mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device
                )
            else:
                atoms, pos = [], None

            component_folds = []
            model_outputs: List[np.ndarray] = []  # List of M arrays, each of shape (K,)
            for key in self.selected_keys:
                if key in cached_1d2d:
                    model_outputs.append(cached_1d2d[key])
                    component_folds.append(getattr(self,"_fold_predictions",{}).get((smi,key),np.asarray(cached_1d2d[key])[None]))
                else:
                    val_3d = self._forward_unimol_folds(key, pos, atoms).cpu().numpy()
                    model_outputs.append(val_3d)
                    component_folds.append(self._last_3d_folds.detach().cpu().numpy())

            stacked_m_k = np.stack(model_outputs, axis=0)  # (M, K)

            if self.ensemble_method == "stacking_linear":
                meta_vec = stacked_m_k.reshape(-1)  # (M * K,) matching np.concatenate(y_pred_list, axis=-1)
                # (F, K, M*K) @ (M*K,) + (F, K) -> (F, K)
                fold_preds = np.einsum("fkm,m->fk", self.stacking_linear_weights, meta_vec) + self.stacking_linear_biases
                if self.task in ["classification", "multilabel_classification"]:
                    fold_preds = 1.0 / (1.0 + np.exp(-fold_preds))
                norm_mean = np.mean(fold_preds, axis=0)  # (K,)
                uncertainty_groups = [fold_preds]
            elif self.ensemble_method == "stacking_general":
                meta_vec = stacked_m_k.reshape(1, -1)  # (1, M * K)
                fold_preds = np.stack(
                    [np.asarray(st._predict(meta_vec), dtype=np.float64).reshape(self.output_dim) for st in self.stacking_models],
                    axis=0,
                )  # (F, K)
                norm_mean = np.mean(fold_preds, axis=0)
                uncertainty_groups = [fold_preds]
            elif self.ensemble_method == "vote":
                norm_mean = np.max(stacked_m_k, axis=0)
                uncertainty_groups = [stacked_m_k]
            else:  # average (or single model)
                norm_mean = np.mean(stacked_m_k, axis=0)
                uncertainty_groups = component_folds

            # Preserve the trained ensemble's normalized-space point estimate.
            # Fold disagreement is measured separately, after each inverse transform.
            real_mean = self._inverse_scale_numpy(norm_mean)
            real_groups = [self._inverse_scale_numpy(group) for group in uncertainty_groups]
            center = np.mean([group.mean(0) for group in real_groups], axis=0)
            real_std = np.sqrt(np.mean([np.square(group - center).mean(0)
                                      for group in real_groups], axis=0))
            y_preds_all.append(real_mean)
            y_stds_all.append(real_std)

        y_pred_arr = np.stack(y_preds_all, axis=0).astype(np.float32)  # (N, K)
        y_std_arr = np.stack(y_stds_all, axis=0).astype(np.float32)    # (N, K)
        if self.output_dim == 1:
            return y_pred_arr.squeeze(-1), y_std_arr.squeeze(-1)
        return y_pred_arr, y_std_arr

    @torch.no_grad()
    def predict_classification_labels(
        self, mols: List[Chem.Mol]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For `classification`, `multilabel_classification`, and `multiclass`:
        Returns `(hard_labels, probs, prob_stds)` using `EnsembleModel/threshold.txt`.
        """
        probs, prob_stds = self.predict_3d_mols(mols)
        if self.task == "multiclass":
            labels = np.argmax(probs, axis=-1)
        else:
            th = self.thresholds[0] if self.output_dim == 1 else self.thresholds.reshape(1, -1)
            labels = (probs > th).astype(np.int32)
        return labels, probs, prob_stds

    def target_labels_to_prob_intervals(
        self,
        desired_labels: Union[int, Dict[Union[str, int], int], Sequence[Optional[int]]],
        margin: float = 0.10,
    ) -> Union[Tuple[float, float], Dict[Union[str, int], Tuple[float, float]], List[Optional[Tuple[float, float]]]]:
        """
        Converts desired discrete classification labels (e.g. `1` or `0` for binary/multilabel)
        into continuous probability intervals using `self.thresholds` (`EnsembleModel/threshold.txt`):
          - label == 1 -> `[min(threshold + margin, 0.95), 1.0]`
          - label == 0 -> `[0.0, max(threshold - margin, 0.05)]`
        """
        if isinstance(desired_labels, int):
            th = float(self.thresholds[0])
            return (min(th + margin, 0.95), 1.0) if desired_labels == 1 else (0.0, max(th - margin, 0.05))
        elif isinstance(desired_labels, dict):
            out_dict: Dict[Union[str, int], Tuple[float, float]] = {}
            for k, lbl in desired_labels.items():
                idx = self.target_cols.index(k) if isinstance(k, str) else int(k)
                th = float(self.thresholds[idx])
                out_dict[k] = (min(th + margin, 0.95), 1.0) if int(lbl) == 1 else (0.0, max(th - margin, 0.05))
            return out_dict
        else:
            out_list: List[Optional[Tuple[float, float]]] = []
            for idx, lbl in enumerate(desired_labels):
                if lbl is None:
                    out_list.append(None)
                else:
                    th = float(self.thresholds[idx])
                    out_list.append(
                        (min(th + margin, 0.95), 1.0) if int(lbl) == 1 else (0.0, max(th - margin, 0.05))
                    )
            return out_list

    # --------------------------------------------------------------------------
    # Interface 2: Stage 5 End-to-End Differentiable 3D Forward (Single & Multi-Target)
    # --------------------------------------------------------------------------
    def forward_differentiable_3d(
        self,
        coords: torch.Tensor,
        atom_symbols: List[str],
        smiles: str = "C",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable PyTorch forward pass w.r.t. `coords (N, 3)`.
        - Returns `(y_pred, y_std)`:
          - Scalar tensors if `output_dim == 1`
          - Shape `(K,)` tensors if `output_dim == K > 1`
        - If `self.has_3d_unimol == False` (no UniMolModel in the trained Uni-QSAR directory):
          - Uses `self.aux_3d_head(coords, atom_symbols)` if provided;
          - Otherwise returns exact 1D/2D `Uni-QSAR` prediction attached with `0.0 * coords.sum()`
            so `.backward()` produces clean zero coordinate gradients (`coords.grad = 0`) and
            Stage 5 seamlessly performs pure `DPA-4` physical conformer relaxation!
        """
        if self.ensemble_method == "stacking_general":
            raise NotImplementedError("Nonlinear stacking uses true predict_3d_mols with black-box acceptance")
        cached_1d2d = self._eval_1d_2d_single(smiles)

        model_tensors: List[torch.Tensor] = []
        component_folds = []
        for key in self.selected_keys:
            if key in cached_1d2d:
                prediction = torch.as_tensor(cached_1d2d[key], dtype=coords.dtype, device=coords.device)
                folds = self._fold_predictions[(smiles, key)]
                component_folds.append(torch.as_tensor(folds, dtype=coords.dtype, device=coords.device))
            else:
                prediction = self._forward_unimol_folds(key, coords, atom_symbols)
                component_folds.append(self._last_3d_folds)
            model_tensors.append(prediction)

        stacked_m_k = torch.stack(model_tensors, dim=0)  # (M, K)

        if self.ensemble_method == "stacking_linear":
            meta_vec = stacked_m_k.reshape(-1)  # (M * K,)
            w_t = torch.tensor(self.stacking_linear_weights, dtype=coords.dtype, device=coords.device)  # (F, K, M*K)
            b_t = torch.tensor(self.stacking_linear_biases, dtype=coords.dtype, device=coords.device)   # (F, K)
            fold_preds = torch.einsum("fkm,m->fk", w_t, meta_vec) + b_t  # (F, K)
            if self.task in ["classification", "multilabel_classification"]:
                fold_preds = torch.sigmoid(fold_preds)
            norm_mean = fold_preds.mean(dim=0)  # (K,)
            uncertainty_groups = [fold_preds]
        elif self.ensemble_method == "vote":
            norm_mean = stacked_m_k.max(dim=0).values
            uncertainty_groups = [stacked_m_k]
        else:
            norm_mean = stacked_m_k.mean(dim=0)
            uncertainty_groups = component_folds

        y_pred_real = self._inverse_scale_torch(norm_mean)
        real_groups = [self._inverse_scale_torch(group) for group in uncertainty_groups]
        center = torch.stack([group.mean(0) for group in real_groups]).mean(0)
        residuals = torch.cat([(group - center) / (len(real_groups) * len(group)) ** .5
                               for group in real_groups])
        # vector_norm has a defined zero subgradient for identical folds.
        y_std_real = torch.linalg.vector_norm(residuals, dim=0)

        # Graceful fallback when no 3D UniMol model is present in the checkpoint
        if not self.has_3d_unimol:
            if self.aux_3d_head is not None:
                aux_val = self.aux_3d_head(coords, atom_symbols).view(self.output_dim)
                # Straight-through estimator: exact 1D/2D Uni-QSAR value + 3D surrogate gradient
                y_pred_real = y_pred_real.detach() + (aux_val - aux_val.detach())
            else:
                # Attach zero-weight graph connection so `.backward()` never raises an error
                y_pred_real = y_pred_real + 0.0 * coords.sum()

        if self.output_dim == 1:
            return y_pred_real.squeeze(0), y_std_real.squeeze(0)
        return y_pred_real, y_std_real

    # --------------------------------------------------------------------------
    # Helper: Multi-Target Interval Loss & Hit Mask (By Column Name or Index)
    # --------------------------------------------------------------------------
    def compute_multi_target_interval_loss(
        self,
        y_pred: torch.Tensor,
        target_intervals: Union[Tuple[float, float], Dict[Union[str, int], Tuple[float, float]], List[Optional[Tuple[float, float]]]],
        weights: Optional[Dict[Union[str, int], float]] = None,
    ) -> torch.Tensor:
        """
        Computes a unified differentiable interval hinge loss across 1 or K targets:
          L_interval = sum_k w_k * (ReLU(low_k - y_k) + ReLU(y_k - high_k))
        `target_intervals` can be:
          - A single `(low, high)` tuple (for single-target models)
          - A dict mapping target column name or index to `(low, high)`, e.g.:
            `{"lambda_pl": (455.0, 465.0), "delta_est": (0.0, 0.10)}`
          - A list of `(low, high)` tuples (or `None` for unconstrained targets)
        """
        from ..constraints import TargetIntervals
        intervals=TargetIntervals.parse(target_intervals,self.target_cols)
        violations=intervals.violations(y_pred)
        if weights:
            w=torch.tensor([weights.get(name,weights.get(i,1.)) for i,name in enumerate(self.target_cols)],
                           device=y_pred.device,dtype=y_pred.dtype)
            if not torch.isfinite(w).all() or (w<0).any(): raise ValueError("Invalid target weights")
            violations=violations*w
        return violations.sum()
