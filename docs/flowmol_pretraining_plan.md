# 全周期表等变三维分子流匹配基座（`FlowMol-118`）独立预训练方案与架构规范

> **项目定位与解耦原则**：本方案面向独立发表的全周期表（118 元素）连续-离散混合流匹配三维分子基座模型（`src/loom/flowmol/`，权重格式 `flowmol_foundation_v1`）。该预训练体系在代码、语料、评测与论文叙事上**与 `Uni-Loom` 论文工作完全解耦**（不修改 `paper/` 下任何 `Uni-Loom` 稿件内容）：
> 1. **独立基座能力**：无需依赖任何 `loom.weaving` 组件即可通过 `FlowMolModel.from_pretrained(...)` 独立完成全周期表有机分子、主族杂原子功能分子与过渡金属/镧系金属有机配合物的无条件三维生成、分域基准采样与微调；
> 2. **下游代码与权重即插即用**：同时预留零初始化条件调制钩子、可开关 LoRA（`ToggleableLoRALinear` + `disable_lora()`）与解析步进对数似然 `sde_ctmc_step_logprob()`，使 `LoomFlowMolGenerator.from_pretrained_flowmol(...)` 可零改造直接加载同一份预训练权重。

---

## 1. 全原子化学表示与物理等变主干设计

### 1.1 全周期表统一离散状态空间
- **元素词表（`N_ATOM_TYPES = 118`）**：严格按原子序数覆盖 `H (Z=1)` 至 `Og (Z=118)`（槽位索引 `idx = Z - 1`），专用吸收态掩码索引 `ATOM_MASK_IDX = 118`（彻底废除旧版复用 `'Se'` 作掩码槽位的歧义设计）。
- **形式电荷词表（`N_CHARGES = 11`）**：覆盖 `(-4, -3, -2, -1, 0, +1, +2, +3, +4, +5, +6)`，专用掩码索引 `CHARGE_MASK_IDX = 11`，完整兼容有机两性离子、高价主族中心及 `3d/4d/5d` 过渡金属与镧系元素氧化态。
- **共价与配位键词表（`N_BOND_TYPES = 6`）**：覆盖 `("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "COORDINATE")`，专用掩码索引 `BOND_MASK_IDX = 6`；在三维分子图重建（`tensors_to_rdkit_mol`）中自动识别金属受体与路易斯酸碱对，将无向 `COORDINATE` 边定向还原为 RDKit `DATIVE`（`Ligand -> Metal`）配位键，并对脱除金属中心后的配体子图执行严格化合价/芳香性校验。

### 1.2 34 维元素周期律物理不变量与双线性输出残差头
针对 118 维纯独热编码中稀有/未见元素输入列死区与输出负类梯度压制问题，构建冻结的 `(119, 34)` 周期律物理特征矩阵（`[MASK]` 行严格全零）：
- 归一化原子序数 $Z/118$（1 维）、周期 `1..7` 独热（7 维）、IUPAC 族 `0..18` 独热（19 维，`0` 表征镧系/锕系 `f` 区）、`s/p/d/f` 分区独热（4 维）、归一化共价半径 $r_{\text{cov}}/2.5\text{ \AA}$、范德华半径 $r_{\text{vdW}}/3.0\text{ \AA}$ 及外层价电子数 $n_{\text{val}}/8$（3 维）。
- **输入与自条件融合**：节点嵌入同时读取离散状态特征与 `34` 维物理描述符；几何自条件（Self-Conditioning）通道传入软预测分布对周期表特征矩阵的期望向量 $\hat{\mathbf{p}}_a \mathbf{\Phi}_{\text{PT}}$。
- **因式分解双线性输出残差**：元素分类头采用 $\mathbf{z}_a = \text{MLP}(\mathbf{h}) + (\mathbf{W}_{\text{PT}}\mathbf{h})\mathbf{\Phi}_{\text{PT}}^\top$，使同族/同区/相近共价半径元素共享物理分类流形，同时支持推理期按需施加 `allowed_elements` 与 `allowed_charges` 掩码。

### 1.3 纯 PyTorch 无 DGL 严格 $SE(3)$ 等变 GVP 主干与几何先验
- **零 DGL 依赖**：基于扁平边索引与 `index_add_` 实现全连接图 `GVPConv`、`NodePositionUpdate` 与 `EdgeUpdate`，径向基上限扩展至 `rbf_dmax = 20.0 Å` 以覆盖百原子有机分子与大体积配合物直径。
- **尺度自适应先验与手性保持对齐**：采用体积匹配的零质心高斯先验标准差 $\sigma(N) = 1.15\,(N/25)^{1/3}$，配合匈牙利排列与严格保持手性的 $SO(3)$ Kabsch 对齐（强制 $\det(\mathbf{R}) = +1$，禁止镜面反射翻转立体化学构型）。
- **按分子均衡损失（Per-Molecule Loss Normalization）**：坐标 MSE 与掩码交叉熵先在单个分子内部按节点数/掩码槽位数归一，再跨分子求均值，消除动态边预算 batch 中百原子大分子因 $O(N^2)$ 边数优势对小分子梯度的压制。
- **三档对数均匀模型扩展谱系（Model Scaling Law Suite，`~3.1×` 等比跨度覆盖约一个数量级）**：
  | 预设档位 | 参数量 | 标量/边通道 (`s / e`) | 向量/叉乘通道 (`v / cp`) | 更新层数 (`L`) | 定位 |
  |---|---|---|---|---|---|
  | **`small`** | **`4.23M`** (`4,230,844`) | `224 / 128` | `16 / 4` | `6` | 对齐原版 `FlowMol / FlowMol3` (~4.2M) 基线规模锚点 |
  | **`standard`** | **`12.74M`** (`12,739,132`) | `320 / 192` | `24 / 6` | `9` | 中等规模主力对照档（`3.01× small`） |
  | **`flagship`** | **`40.19M`** (`40,190,400`) | `512 / 256` | `32 / 8` | `12` | 全周期表旗舰基座（`3.15× standard`，`9.50× small`） |
  - 三档模型在完全相同的五源全周期表语料（文渊 `flowmol-corpus-118:1`）、相同源重采样权重 `{"tmqm": 4.0, "oe62": 3.0, "qm9": 1.5}` 与统一 `180M`（`1.8 亿`）分子-时间步预算下训练，支持同时绘制严格 **Iso-Data（等样本量 `180M`）** 与 **Iso-Compute（等算力 `GPU-h`）** 两条全周期表三维流匹配扩展律（Scaling Law）曲线；本地产物分别归档于 `pretrained/flowmol/{flagship,standard,small}/`。

---

## 2. 五源全原子三维语料构建（`flowmol-corpus-v2`）

统一整合五套公开高质量全原子（含显式氢）三维分子与金属配合物数据集（过滤上限 `max_atoms = 120`）：

| 数据源 | 理论级别与规模 | 化学空间定位 |
|---|---|---|
| **QM9** | `B3LYP/6-31G(2df,p)` DFT（~13.1 万分子） | 小分子高精度平衡键长/键角锚点 |
| **OE62** | `PBE+vdW` 弛豫三维几何（~6.1 万分子） | 覆盖 `H, Li, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, Te, I` 16 元素光电功能分子与锂配合物 |
| **PCQM4Mv2** | PubChemQC `B3LYP/6-31G*` DFT（~312 万分子） | 广谱主族化学拓扑与高精度量子化学基态构象骨架 |
| **GEOM-Drugs** | `CREST / GFN2-xTB` 官方划分（~146 万构象） | 药物尺度柔性多构象与大尺寸杂环覆盖 |
| **tmQM** | `TPSSh/def2-SVP` DFT + Wiberg 键级（~8.6 万配合物） | 完整覆盖全部 **30 种 `3d/4d/5d` 过渡金属 + `La`**，依托 DFT Wiberg 键级与闭壳层配体整数规划恢复精确共价键级、形式电荷与 `MND` `DATIVE` 配位键 |

- **紧凑 Flat-CSR 存储与分域元数据**：输出无碎片二进制数组及 `meta.json`（同时记录全局与各数据源独立的原子数经验直方图 `per_source_n_atoms_hist` 和活跃元素集 `per_source_active_elements`），训练期默认启用重采样权重 `{"tmqm": 4.0, "oe62": 3.0, "qm9": 1.5}` 强化过渡金属与重主族杂原子梯度信号。

---

## 3. 16×A100 多机 RDMA 训练、数值精度门禁与无过冲少步求解

1. **Ampere 混合精度与局部 `FP32` 强制白名单**：
   - 线性层启用 `BF16 AMP` + `FP32` 主权重/EMA 权重 + `TF32` 兜底；
   - 成对距离、`RBF`、单位方向向量、GVP 向量范数与叉乘、Zero-COM 质心扣除、$SO(3)$ Kabsch 对齐、插值调度系数及损失/对数似然归约全程包裹 `torch.autocast(..., enabled=False)` 强制 `FP32`。
   - 全量开训前在双机 16×A100 RDMA 环境执行 `2,000` 步确定性验证集 `纯 TF32` vs `BF16+FP32白名单` 准入审计（要求 `|Δval_loss_x| < 0.05`, `|Δval_loss_e| < 0.05`, `nan_skips == 0`）。
2. **无过冲 $\alpha$-空间二阶 Heun 求解与退火 CTMC**：
   - 采用精确有限步凸组合权重 $w_x = \text{clamp}\!\left(\frac{\alpha_s - \alpha_t}{1-\alpha_t}, 0, 1\right)$ 替代切线欧拉外推 $\Delta t\frac{\dot{\alpha}_t}{1-\alpha_t}$，彻底消除 20–25 步非均匀网格在 $t\to 1$ 处的过冲震荡；
   - 离散 CTMC 重掩码率引入 $(1-\alpha_s)$ 退火因子 $\eta\,\Delta t\,(1-\alpha_s)$，避免少步采样末端大步长重掩码破坏已收敛共价骨架。

---

## 4. 独立调用与下游接入接口

```python
# 1. 独立基座加载与全周期表 / 分域采样（零 loom.weaving 依赖）
from loom.flowmol import FlowMolModel

model, meta = FlowMolModel.from_pretrained("pretrained/flowmol/best_ema.pt", device="cuda:0")
tmqm_sizes = FlowMolModel.sample_empirical_sizes(meta, n_mols=64, domain="tmqm", seed=42)
samples = model.sample(tmqm_sizes, solver="heun", n_steps=25, grid_type="quadratic_end")

# 2. 下游框架即插即用加载同一份权重（可选挂载零初始化条件头与 LoRA）
from loom.weaving.flowmol_slot import LoomFlowMolGenerator

slot = LoomFlowMolGenerator.from_pretrained_flowmol(
    "pretrained/flowmol/best_ema.pt",
    target_cols=["homo"],
    target_units=["eV"],
    mean=[-5.8],
    scale=[0.8],
    lora_rank=16,
)
```
