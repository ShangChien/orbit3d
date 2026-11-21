# Uni-Loom 后训练演进与双生成底座（`Uni-3DAR` vs `Hybrid Flow-GRPO`）工作计划

> **文档定位**：本文档记录 `Uni-Loom` 面向下一阶段的后训练算法演进、通用流匹配底座补齐方案、24GB 消费级显卡适配规范，以及修订后的 **`Uni-Loom + Uni-3DAR`** 与 **`Uni-Loom + Hybrid Flow-GRPO`** 全面架构对比。本文档仅作为仓库内部研发路线图，不纳入当前论文正文。
>
> **状态基线（2026-11-17）**：`src/loom/weaving/slots.py`、`uni3dar_slot.py`、`flowmol_slot.py`、`critic_bus.py` 与 `grpo.py` 已完成基础生成器热插拔（`build_generator_slot`）及在线优势加权流匹配（`alignment_mode="awr_sft"`）落地并通过全量测试（`328 passed`）。以下耗时估算基于本仓库在 NVIDIA A800 上的实测吞吐基准折算。

---

## 1. 修订版核心对比：`Uni-Loom + Uni-3DAR` vs `Uni-Loom + Hybrid Flow-GRPO`

在引入 **“空间势能 Logit 偏置 + 锚点序列前置”** 与 **“八叉树粗轮廓截断强化学习（Coarse-Silhouette GRPO）”** 两项机制后，八叉树自回归底座突破了传统自回归模型在局部三维几何控制上的局限，并进一步放大了层次化空间表示的计算红利。假设双方均配备成熟的多源通用预训练底座并在同一套 `CompositeCriticBus`（`Uni-Mol2` + `DPA-4` 受力 + `SA_Score <= 3.5`）下对齐，修订后的全景对比如下：

| 评估维度 | **`Uni-Loom + Uni-3DAR`**<br>（八叉树层次化自回归 + 空间势能偏置 + 粗轮廓分层 GRPO） | **`Uni-Loom + Hybrid Flow-GRPO`**<br>（预训练等变图流 `FlowMol` + 少步 SDE-CTMC GRPO） | **综合判定与原因剖析** |
|---|---|---|---|
| **1. 原始局部三维几何与对称性**<br>（未弛豫 `F_max`、共轭平面、配位角） | 体素网格离散化 + 叶子层局部精细坐标分箱；非严格解析旋转等变（依赖预训练增强与推理期多视角旋转采样 + 末端弛豫兜底） | **严格解析 $SE(3)$ 旋转平移等变**；连续 $\mathbb{R}^{3N}$ 坐标与显式键级矩阵 $\mathbf{E}$ 全程连续协同演化，无需旋转增强，原始未弛豫键角/二面角更平滑 | **`Hybrid Flow-GRPO` 占优**<br>（物理对称性与亚埃级连续几何内生性更强） |
| **2. 原子数自适应与跨尺度骨架外推**<br>（极端靶标分位点探索） | **原子数 $N$ 随八叉树分裂自然涌现**；策略可通过增减稠环或支链体素自主调节分子尺寸以匹配极端轨道能级或分子量约束 | 采样启动前必须在 $t=0$ **预先固定节点数 $N$**（或依赖虚拟空节点槽位）；若条件原子数先验 $p(N\mid y)$ 偏窄会限制尾部外推 | **`Uni-3DAR` 明显占优**<br>（无需预设节点数，生成自由度更高） |
| **3. 全轨迹在线 RL 训练效率与稳定性** | **单次因果掩码前向即可并行算出整条序列 100% 精确对数似然与解析 KL**；无随机微分噪声超参，单张 A800/4090 约 `1–2h` 稳定收敛 | 不同时间步依赖上一步三维构象，无法合并为单次图前向；连续 SDE 噪声 $\sigma_t$ 与离散 CTMC 跳变耦合对超参较敏感，单卡慢约 **`2–3×`** | **`Uni-3DAR` 明显占优**<br>（Teacher-Forcing 并行似然天然契合 GRPO） |
| **4. 粗分辨率截断极速 RL 能力**<br>（不解码至最终原子层的加速上限） | **独有金字塔红利**：可在粗层边界停表（`_stop_at_leaf_boundary=True`），跳过 `~80%+` 叶子解码步与末端重资产推理，以“空间包络精确势能 + `OctreeProcessCritic` + 9:1 慢步校准”实现 **`8–15×` 训练提速** | **无法实现单步计算缩减**：即便只积分到 $t=0.3$，每步仍须运行完整的 $N$ 节点、$N(N-1)$ 边等变图网络，不存在低维粗轮廓快速通道 | **`Uni-3DAR` 独享优势**<br>（层次化 Token 金字塔 vs 全尺寸等变图） |
| **5. 推理吞吐与中途早停剪枝**<br>（OE62 40–80 原子规模，Best-of-$M$） | 裸采样约 `28.8 mol/s`；粗层八叉树仅消耗 **`<20%` 算力即可由 `OctreeProcessCritic` 剪除劣质分支** | 配合 20–25 步少步二阶求解，裸采样可达 **`~35–45 mol/s`（追平或略快于裸 `Uni-3DAR`）**；但流轨迹需积分至 $t\approx 0.45$ 骨架才初显，`Mid-Flow Critic` 只能省约 `50%` 算力 | **裸生成打平；<br>大 $M$ 剪枝筛选 `Uni-3DAR` 快 `~1.5×`** |
| **6. 大分子复杂度扩展律**<br>（$N \ge 100$ 重原子/大环/复合物） | 稀疏八叉树序列长度近似按 **$O(N \log L)$ 线性温和增长**，配合 FlashAttention + KV-Cache 易于扩展 | 为防长程成环断裂与空间碰撞需维护 **$O(N^2)$ 全连接边图**（120 原子即 `14,280` 条有向边），上百原子后显存与延迟二次方增长 | **`Uni-3DAR` 占优**<br>（空间稀疏分割优于全连接对边图） |
| **7. 局部三维空间条件控制**<br>（口袋限域 SBDD、特定位点定长生长、3D Linker） | • **免微调档（现有 `num_tree=1` 权重）**：依托 `tree_logit_mask` 外部空间禁入掩码 + 单树强制体素占据与 `known_atoms` 计数守恒；但粗层看不到口袋化学语义，且同层 Morton 空间序存在局部因果盲区与网格朝向敏感。<br>• **配对微调档（扩展 `num_tree=2` SFT）**：以 `tree 0`（口袋/母核 KV-Cache）→ `tree 1`（配体/侧链）消除因果盲区，兼具高层防碰撞早停剪枝及定长/变长架桥一体优势。 | • **免微调档（Replacement + 引导）**：不能直接从 $t=0$ 硬锁离散态，须应对**全图 Zero-COM 质心耦合漂移**、**吸收态 CTMC 解掩比例失配**及简单替换的接缝键角畸变（需同步掩码调度 + 重构梯度/往返重采样）；且数百原子蛋白口袋无法并入 $O(N^2)$ 小分子全连接图。<br>• **配对微调档（掩码条件/口袋交叉注意力 SFT）**：严格 $SE(3)$ 等变、无网格朝向偏差，但须预先指定总原子数 $N$。 | **双方均须区分“免训近似”与“配对微调”**：<br>• 免微调双方皆有物理/因果折中；<br>• 经配对 SFT 后：变长架桥/大口袋早停剪枝 **`Uni-3DAR` 占优**，零朝向敏感的严格等变母核微调 **`Flow` 占优** |

### 核心战略判断
1. **`Uni-3DAR` 插槽是 `Uni-Loom` 综合性价比最高的主力引擎**：在全局性质导向生成、高通量树剪枝筛选、大尺寸分子扩展，以及结合空间势能偏置与粗轮廓截断 RL 后的变长局部三维设计（如未知原子数 Linker 与口袋空腔填充）上，具备训练快、显存省、原子数自适应的系统级优势。
2. **通用预训练 `FlowMol + Hybrid Flow-GRPO` 插槽是高价值的正交第二引擎**：在要求严格解析 $SE(3)$ 等变性、固定节点数的精细连续构象生成、显式键级协同演化，以及证明 `Uni-Loom` 跨生成范式普适性（Autoregressive + Continuous-Discrete Flow）的平台定位上不可替代。

---

## 2. 路线一工作计划：`Uni-3DAR` 局部三维空间控制与粗轮廓分层 RL

依托现有代码中的 `geometric_guard`（`src/loom/weaving/uni3dar_slot.py`）、`_stop_at_leaf_boundary`、`topology_only_policy`（`src/loom/weaving/grpo.py`）及 `OctreeProcessCritic`，按两个增量模块推进：

### 2.1 局部三维空间条件控制模块（Pocket & Site-Specific Substructure Growth）

> **源码现状基线核查**：内置 `Uni-3DAR` 源码中，`known_atoms` 根计数覆盖与逐层剩余原子守恒（`uni3dar_sampler.py`）、`tree_logit_mask` 与 `atom_constraint` 配额掩码均已现成可用；但 `uni3dar.py` 中仅 `crystal` 模式启用了 `num_tree=2`，**现有小分子预训练权重严格为单树（`num_tree=1, tree_emb=None`）**，且 `grid_dataset.py` 未实现 `--data-type complex`（`ProteinGridDataset` 仅提取蛋白主链 `CA`）。因此不能在现有单树权重上零样本直接运行双树序列，工程落地严格按 **“Phase A 免微调单树掩码档”** 与 **“Phase B 双树配对条件 SFT 生产档”** 两级推进：

1. **Phase A — 免微调单树控制档（基于现有 `num_tree=1` 小分子权重，推理与 RL 即插即用）**：
   - **外部口袋几何限域**：不将蛋白原子送入 Transformer 序列，而是作为外部三维空间场：在粗层 `tree_logit_mask` 中将落入口袋重原子范德华排斥球或空腔包络外的子八分体直接置 `-inf` 提前剪枝，在叶子层叠加排斥与药效团吸引 Logit 偏置（注：粗层隐藏态本身不感知口袋静电语义，需靠末端对接/势能奖励经在线 RL 内化）。
   - **单树母核强制占据与定长生长**：设总原子数 `known_atoms[:, 0] = N_frag + N_new`；根据已知母核坐标预先算出各级体素的必占子八分体掩码与最小原子数下界 $c(V) \ge c_{\text{frag}}(V)$（非生长禁入区令 $c(V) = c_{\text{frag}}(V)$），并在 `level=0` 对母核所在体素 Teacher-Force 填入已知元素与精细坐标，其余自由体素由策略采样并累加似然（注：受同层 Morton 空间序限制，序号靠前的新生分支对序号靠后的母核精细叶子存在同层因果盲区，且须配 `4–8` 路随机旋转采样缓解网格朝向偏差）。
2. **Phase B — 双树配对条件 SFT 生产档（扩充 `num_tree=2`，彻底消除因果盲区）**：
   - **架构扩充与成对微调**：在 `LoomUni3DARGenerator` 中扩充 `num_tree=2`（初始化 `tree_emb` 并从 `0` 号头拷贝初始化第 `1` 套 `tree_heads / atom_heads / count_head`），构造 `(蛋白口袋/固定母核 + is_anchor_site 位点标记, 配体/新增侧链)` 成对三维数据做专门的条件 SFT，使模型习得“`tree 0` 收尾后跳回 `level 7` 生成 `tree 1` 并跨树 attend 完整上下文”的序列分布。
   - **定长/变长一体与体素早停剪枝**：`tree 0` 预先写入 `kv_cache`（推理与 GRPO 全程冻结不计似然）；在 `tree 1` 根节点通过设置 `known_atoms[:, 1] = N_new`（定长生长，可配 `atom_constraint`）或保持 `-1`（未知长度 3D Linker / 空腔自适应填充）实现统一切换，同时保留 `tree_logit_mask` 高层零穿模剪枝与指定位点成键壳层（$1.1\text{–}1.9\text{ \AA}$）正偏置。

### 2.2 粗轮廓截断快慢双频强化学习（Coarse-Silhouette Dual-Cadence GRPO）
1. **快步循环（占比 `90%`，Coarse-Only Step）**：
   - 开启 `_stop_at_leaf_boundary=True` 与 `topology_only_policy=True`，仅解码前 `~15%–20%` 的粗层八叉树分裂序列；
   - 奖励由两部分合成：① **精确空间几何奖励**（口袋体积排斥/贴合 3D IoU、锚点间体素连通性、目标体积/回转半径）；② **代理电子/合成效用奖励**（由 `OctreeProcessCritic` 读取 `coarse_h_concat` 与 `coarse_geom_feats` 毫秒级给出）。
2. **慢步校准循环（占比 `10%`，Full-Rollout Calibration Step）**：
   - 每隔 `N=10` 步执行一次完整叶子层解码，调用 `CompositeCriticBus`（`Uni-Mol2` + `DPA-4` + `SA_Score`）获取真实端到端奖励：一方面同步微调叶子层 `atom_heads`，另一方面用新采集的真实 `(粗轮廓, 终态效用)` 样本在线增量刷新 `OctreeProcessCritic`，防止策略对静态 Critic 刷分（Reward Hacking）。

---

## 3. 路线二工作计划：16×A100 全周期表通用 `FlowMol` 预训练底座与 `Hybrid Flow-GRPO`

> **工程基座选型与独立发版解耦原则**：全周期表预训练底座（`src/loom/flowmol/`，独立技术方案见 `docs/flowmol_pretraining_plan.md`）作为独立研究工作构建与发版，**不耦合进 `Uni-Loom` 论文正文**；对 `Uni-Loom` 仅通过标准权重格式（`flowmol_foundation_v1`）与 `LoomFlowMolGenerator.from_pretrained_flowmol(...)` 提供即插即用的代码与权重支持。

### 3.1 数值精度规范（面向 16×A100 / A800 Ampere RDMA 集群）
- **标准配置**：启用 **`BF16` 自动混合精度（AMP）+ `FP32` 主权重/EMA/优化器状态 + `TF32` 矩阵乘兜底**（Ampere 不支持硬件 FP8；不采用需动态缩放的 FP16）。
- **强制局部 `FP32` 白名单（`torch.autocast(..., enabled=False)`）**：
  1. 三维成对坐标差、欧氏距离 `||x_i - x_j||`、`rbf_dmax=20.0 Å` 径向基展开与方向单位向量归一化；
  2. GVP 等变向量通道的 $L_2$ 范数与叉乘特征（`n_cp_feats`）；
  3. 质心扣除（COM removal）、体积自适应先验 $\sigma(N)=1.15(N/25)^{1/3}$ 与手性保持 $SO(3)$ Kabsch 对齐（$\det(\mathbf{R})=+1$）；
  4. 插值调度系数 $\alpha_t, \dot{\alpha}_t$、按分子均衡的 CTMC 掩码交叉熵归约，以及后训练中的优势指数权重或 SDE-CTMC 步进 log-prob 累加。
- **准入准则**：全量开训前须完成 `2,000` 步确定性验证集 `纯 TF32` vs `BF16+局部FP32` 对照，验证 `val_loss_x`、`val_loss_e`、梯度范数与 `nan_skips == 0`。

### 3.2 16×A100 全原子预训练分阶段配置
1. **Stage 1 — 全周期表 118 元素词表与可插拔钩子定型（已完成）**：
   - 一次性对齐 **全周期表 `118` 元素（`Z=1..118` + `[MASK]=118`）**、**`11` 档形式电荷（`-4..+6` + `[MASK]=11`）** 与 **`6` 类化学键（含 `COORDINATE/DATIVE` 配位键 + `[MASK]=6`）**，辅以 `34` 维周期律物理描述符与因式分解双线性输出残差头，下游可通过 `allowed_elements` / `allowed_charges` 按需裁剪；
   - 主干纯学无条件分布 $p_\theta(\mathbf{X},\mathbf{A},\mathbf{C},\mathbf{E})$ + 几何自条件，预留零初始化条件调制接口、阶段 5 片段锚定掩码投影 `fragment_cond_proj` 与可开关 GVP LoRA 接口（`ToggleableLoRALinear` + `disable_lora()`）。
2. **Stage 2 — 五源有机+金属配合物三维语料与动态边数分桶（构建 `flowmol-corpus-v2`）**：
   - 整合五套公开三维全原子数据集（**`QM9 + OE62 + PCQM4Mv2 + GEOM-Drugs + tmQM`**，覆盖有机药物分子、16 元素主族功能分子及全部 30 种 `3d/4d/5d` 过渡金属 + `La` 配合物，`max_atoms=120`）；
   - 实现跨 16 卡同步的 **Dynamic Edge-Budget Bucketing** 与精确步内断点续训（`skip_batches`）。
3. **Stage 3 — 双机 16×A100 RDMA 预训练与无过冲少步求解**：
   - **旗舰底座（`flagship`，`~40.1M` 参数）** 与 **标准底座（`standard`，`~12.8M` 参数）**；
   - 采用无过冲精确有限步 $\alpha$-空间二阶 Heun 求解器与 $(1-\alpha_s)$ 退火 CTMC，支持 `25` 步高质量快速生成及全开放 118 元素 / 分域采样评估。
4. **Stage 4 — 下游低成本迁移与双模后训练支持**：
   - 预训练底座就绪后，适配任意新性质预测模型仅需：**轻量 LoRA 条件 SFT（8×A800 约 `10–20 分钟` / 单卡 `~1–1.5h`）+ `Uni-Loom` 在线奖励对齐**；
   - 在 `LoomFlowMolGenerator` 中并存提供两种在线对齐档位：
     - **快速默认档（`awr_sft`，已实现）**：单卡约 `0.6–0.8 GPU-h`（8 卡 `<10 分钟`），零噪声超参；
     - **完整强化档（`Hybrid Flow-GRPO`，待扩展）**：实现连续坐标 ODE-to-SDE 高斯核与离散原子/电荷/键级 CTMC 跳变核的联合单步对数似然 `sde_ctmc_step_logprob()`，采用训练步数压缩（$T_{\text{train}}=12\text{–}16$）进行正负双向策略梯度更新（8×A800 单次后训练约 **`30–45 分钟`**），并支持在 $t\approx 0.45$ 挂载轻量 `Mid-Flow Critic` 做中途筛选剪枝。
5. **Stage 5 — 局部条件生成与轨迹 Inpainting 规范（需处理三大分布约束，待扩展）**：
   - **不可直接硬套朴素替换法的三项物理/分布约束**：① **全图 Zero-COM 冲突**（`flowmol_slot.py` 要求全部 $N = N_{\text{frag}} + N_{\text{new}}$ 节点总质心归零且先验做 Kabsch 对齐，而采样前未知新增侧链带来的总质心偏移，硬钉死母核再每步扣全图质心会引发耦合平移跳变）；② **吸收态 CTMC 掩码率失配**（预训练在 $t=0$ 为 100% 全 `[MASK]`，若早期直接将母核及非生长位点 `NO_BOND` 边暴露为已解掩终态会导致严重 OOD）；③ **蛋白口袋不可入图**（数百原子的口袋不能拼入小分子 $O(N^2)$ 全连接共价图）。
   - **免微调近似档 vs 掩码条件微调生产档**：
     - **免微调档**：仅适用于小分子母核定长生长——锚定离散原子/内部键及非位点 `NO_BOND` 边须按 $t$ 时刻正向掩码率同步随机保留 `[MASK]`（仅在解掩跳变时约束跳向已知状态），连续坐标配合重构梯度引导（$\nabla_{x_t}\|\hat{x}_1^{\mathcal{F}}(x_t) - x_1^{\mathcal{F}}\|^2$）或同时间步往返重采样（RePaint corrector）消除接缝畸变与质心漂移。
     - **生产微调档（推荐）**：在预训练底座上增设 `is_fixed_fragment / is_growth_site` 节点与边条件掩码（坐标系改以锚定片段质心或口袋中心为参考系，解除全图 Zero-COM 耦合；口袋场景外挂轻量等变口袋编码器做交叉注意力）进行局部条件 SFT；在 `Hybrid Flow-GRPO` 中，`sde_ctmc_step_logprob()` 仅累加自由节点与自由边的转移似然。

---

## 4. 单卡 24GB RTX 4090 上的 `Hybrid Flow-GRPO` 后训练与推理适配规范

为确保通用 `FlowMol` 底座开源后，仅有单张 **24GB RTX 4090** 的下游用户也能无障碍运行 `Hybrid Flow-GRPO` 后训练（预计单次 `1.5–2.5 小时`）与高吞吐推理，强制执行以下五项显存控制设计（训练峰值显存控制在 **`11–15 GB`**，推理 `bs=128` 峰值 **`6–9 GB`**）：

1. **评价资产 `BF16` 化与分阶段 CPU 锁页卸载（Stage-Wise Offloading）**：
   - `Uni-Mol2-1.1B` 全程以 `BF16` 推理（显存由 `~4.4 GB` 降至 `~2.2 GB`），`DPA-4` 弛豫限制 `chunk_size=8`；
   - `CompositeCriticBus` 打分完毕后即将预测器与势函数异步卸载至 CPU Pinned Memory（PCIe 4.0 耗时 `~0.12s`），使策略梯度反传阶段 GPU 显存 100% 留给生成底座。
2. **`disable_lora()` 共享主干替代 `deepcopy(ref_model)`**：
   - 后训练冻结预训练 GVP 主干，仅更新 `<8%` 的 LoRA 与条件头参数；计算 KL 参考对数似然时通过临时关闭 LoRA 旁路复用同一份主干权重，消除双倍模型与优化器显存占用。
3. **轨迹图快照驻留 CPU Pinned Memory + 单时间步逐拍梯度累加**：
   - Rollout 阶段（`torch.no_grad()`）将每步图状态 `(x_t, a_t, c_t, e_t, old_logprob)` 存入 CPU 锁页内存（32 个 80 原子分子 × 16 步仅占 `<40 MB`）；
   - 更新阶段每次仅取 **单时间步、`micro_batch=8~16`** 上卡前向并立即对 $\mathcal{L}_t / K$ 调用 `.backward()` 释放计算图，使激活显存关于时间步数严格保持 **$O(1)$**。
4. **分层随机时间步子采样（Stochastic Timestep Subsampling）**：
   - 每条 $T_{\text{train}}=16$ 步轨迹在单个 PPO epoch 中分层抽取 **`K=3~4` 个关键时间步**（侧重拓扑成型段 $t\in[0.3,0.7]$ 与成键精修段 $t\in[0.7,0.95]$）更新梯度，将单卡 4090 反传耗时压缩 `3–4×`。
5. **全连接边特征梯度检查点（GVP Activation Checkpointing）**：
   - 对 60–80 原子全连接图的 `n_molecule_updates` 各更新层启用 `torch.utils.checkpoint`，以约 `25%` 算力开销换取约 `65%` 边特征激活显存削减；推理期合并 LoRA 权重并以 `noise_level=0` 执行 20–25 步确定性 ODE 采样。

---

## 5. 实证收官基线（2026-11-20）：100% 同平台对称 `LoomService`（`Uni-Loom [Uni-3DAR]` vs `Uni-Loom [PMF + AWR]`）、`GEOM-Drugs` 百原子渐近超越与 `QM9` 六性质全面验证

为定量锚定上述双底座路线图的收官基线，已在单节点 NVIDIA A800-SXM4-80GB（对称开启 `TF32`、推理期严格零 MLIP 弛豫、零 `audit_dpa4` 泄漏）上完成 **`OE62` 100% 同平台对称对决（`M ∈ {1, 2, 4, 8, 12, 16}` + `N ∈ {20..60}` + `42`-prompt `PBE0/def2-TZVP`）**、**`GEOM-Drugs` 10,000 分子无条件生成与 `N = 30..100` 大分子复杂度扩展**，以及 **魔改 `Uni-3DAR-Mol` 在 `PropMolFlow` `50k/50k` 无重叠 `QM9` 六性质基准上的全量评测**（原始 JSON 归档于 `docs/experiments/`，核心配图见 `docs/figures/oe62_llm_style_accuracy_vs_cost.png`、`docs/figures/oe62_atom_scaling_speed_and_mae.png`、`docs/figures/oe62_uni3dar_m4_ablation_and_sdl.png` 与 `docs/figures/geom_drugs_pareto_and_scaling.png`）：

1. **`OE62` 100% 同平台对称对决：`Uni-Loom [Uni-3DAR]`（仅 `2.41 GPU-h`）全面领先 `52.12 GPU-h` `Uni-Loom [PMF + AWR]`（`oe62_uni3dar_v4_pro_fair_30atom{,_m12_m16,_dft_unbiased}.json` & `oe62_pmf_{3h,12h,52h}_awr_fair_30atom{,_m12_m16}.json`）**：
   - **严格平台内评估协议对称**：所有 `Uni-Loom [...]` 组统一通过同一套 `LoomService`、逐条锁死 `(HOMO_i, N_atoms_i)`（`N = 1,792`，均值 `30.38` 原子）、`112`-dev 阻尼仿射校准、`29-D` 化学计量不变 `ResidualCalibratedAsset(stoich_v2)` 与闭式高斯区间命中概率优选，并以 `grpo.py` 精确核函数合成 `Q_3D` 三维综合发现质量指数（`0–100`）；平台外基线统一标注为 `PropMolFlow + Raw Critic`。
   - **全谱系 `M ∈ {1, 2, 4, 8, 12, 16}` 扩展**：`Uni-Loom [Uni-3DAR]`（全口径训练仅 `2.4094 GPU-h`，比 `52.12 GPU-h` 少 **`21.6×`**）随推理预算提升实现 **`M=1: 16.9% [47.58 mol/s, Q_3D=10.0] → M=2: 41.6% [7.86 mol/s, Q_3D=33.8] → M=4: 51.3% [4.02 mol/s, Q_3D=41.9] → M=8: 59.4% (1,065/1,792) [1.94 mol/s, F_max=1.91 eV/Å, MAE=0.2948 eV, Q_3D=48.7, 50k=9.57h] → M=12: 62.3% (1,117/1,792) [MAE=0.2745 eV, Q_3D=50.4]`**；同平台 `Uni-Loom [PMF @ 52.12h + AWR]` 则从 `M=1: 21.2% (Q_3D=13.8)` 扩展至 `M=4: 45.6% (Q_3D=30.5)`、`M=8: 52.3% (F_max=2.90 eV/Å, MAE=0.2907 eV, Q_3D=34.9, 50k=58.55h)`、`M=12: 59.9% (Q_3D=38.6)` 与 `M=16: 62.1% (MAE=0.2496 eV, Q_3D=39.1)`。在主力 `high (M=8)` 档，`2.41 GPU-h` 八叉树底座领先 `52.12 GPU-h` 对齐流匹配 **`+7.1 pp` 盲测命中率** 与 **`+40%` `Q_3D` 质量指数（`48.7` vs `34.9`）**，`50k` 分子总算力成本低 **`6.1×`（`9.57` vs `58.55 GPU-h`）**；同时 `+0.12 GPU-h` 在线 `AWR` 亦将 `3.12h / 12.12h / 52.12h` 流匹配的可合成率从 `88.5%` 提升至 `91.3%–98.0%`。
   - **无偏 `42`-prompt 在线 `GPU4PySCF PBE0/def2-TZVP/DF` 量子终验（`oe62_head2head_dft_anchor_52gpuh_awr_unbiased.json`）**：`Uni-Loom [Uni-3DAR, M=8]` 实现 **`42/42 (100.0%)` SCF 收敛、`100% SA ≤ 3.5`、`F_max = 1.75 eV/Å`、实测 `PBE0` 可合成命中率 `64.3% raw / 64.3% centered (27/42)`、`PBE0 HOMO MAE = 0.2289 / 0.2298 eV`**（`DPA-4 vs PBE0 MAE = 0.1086 eV`），`M=16` 进一步升至 **`66.7% raw / 69.0% centered (28–29/42)`（`MAE = 0.2294 eV`）**；全面超越对称同平台 `Uni-Loom [PMF @ 52.12h + AWR, M=8]`（**`52.4% / 57.1%`，`MAE = 0.2429 eV`，`F_max = 3.38 eV/Å`**）及平台外 `PropMolFlow @ 52h + Raw Critic`（`42.9% / 54.8%`，`MAE = 0.3609 eV`）。
2. **固定中等预算 `M=4` 内部 `2×2` 架构消融与 `2D` 异步部分缺失反馈校准（`oe62_uni3dar_v4_pro_m4_ablation.json` & `oe62_sdl_multi_objective_epistemic_rsi.json`）**：
   - 在外层 `Uni-Mol2` 评估调用严格锁死为 `M=4` 的 `1,792`-prompt 对照中，开启两阶段推测式八叉树剪枝（`32→8→4` vs 纯外层 `4→4→4`）将盲审可合成命中率从 **`40.0%` 拉升 `+13.1 pp` 至 `53.1%`**（`BlindMAE: 0.441 → 0.322 eV`，`|ΔN|: 3.13 → 1.29`）；叠加 `< 0.5 ms` `SE(3)` `PostTreeGraphRefiner` 在零推理期 MLIP 调用下将零步最大力中位数 `F_max` 从 **`3.96 eV/Å` 压降 `49.0%` 至 `2.02 eV/Å`**（`P90: 17.61 → 5.42 eV/Å`），端到端吞吐达 `4.24 mol/s`。
   - 在 `2D (HOMO, LUMO)` 多目标自驱动实验室闭环中，支持异步 `NaN` 单通道读数的 `ResidualCalibratedAsset.ingest_feedback`（`96` 个 OOD 分子，每通道仅 `64` 条有效读数）将独立留出双轨道盲区 DFT 误差同步从 **`0.775 / 1.292 eV` 压降至 `0.713 / 1.106 eV`**（联动 `1D` 实机 `GPU4PySCF` 闭环 `9.65×` 盲审命中增长）。
3. **受控跨原子数渐近扩展对比（`OE62` `N ∈ {20, 30, 40, 50, 60}`，`280` prompts/桶，对称 `stoich_v2`；`oe62_atom_size_buckets_scaling.json`）**：
   - `Uni-Loom [Uni-3DAR, M=8]` 在全部五个原子分桶上 `Q_3D` 全面领先（`65.7, 56.6, 35.0, 9.6, 2.7` vs `Uni-Loom [PMF @ 52.12h + AWR, M=8]` 的 `44.3, 36.3, 18.5, 8.6, 2.0`），盲审可合成命中率在 `N=20`（`65.7%` vs `64.3%`）与大分子段显著拉开：**`N=40: 50.7% vs 35.4% (+15.3 pp)`、`N=50: 27.5% vs 18.9% (+8.6 pp)`、`N=60: 14.3% vs 5.4% (2.65×)`**；在 `N=60` 处，`Uni-3DAR` 单发 `M=1`（`62s/1k`）比 `Vanilla PropMolFlow M=1`（`185s/1k`）快 **`2.9×`**，`M=4`（`901s/1k`，`7.9%` 命中）比 $O(N^2)$ 稠密图 `PMF M=8`（`1,538s/1k`）快 **`1.7×`** 且命中率更高。
4. **`GEOM-Drugs` 药物尺度 10,000 分子无条件生成与 `N = 30..100` 百原子复杂度交叉超越（`geom_drugs_uni3dar_mol_10k_eval.json` & `geom_drugs_large_molecule_scaling.json`）**：
   - 仅用 **`0.4914 GPU-h`** 后训练（`0.148h` SFT + `0.059h` `PostTreeGraphRefiner` [`val_bond_dist_mae = 0.0126 Å`, `val_bond_order_acc = 99.27%`] + `0.285h` `TwoStageOctreeProcessCritic`），无显式键进化的八叉树模型 `Uni-3DAR-Mol [SpecTree M=4 (32→8→4)]`（`4.00 mol/s`，`100%` 唯一性与新颖性）将 `10,000` 分子 **`Valid & Connected` 从原始 `drug.pt` 的 `12.70%` 提升 `6.78×` 至 `86.15%`**（超越 `500` 步联合扩散 `MiDi-Adaptive` 的 `77.8%`），**显式全原子价态稳定率达 `99.83%`**（追平 `SemlaFlow`/`FlowMol` 的 `99.7%–99.8%`），**显式整分子稳定率从 `6.42%` 跃升 `12.5×` 至 `80.15%`**（为 `GeoLDM` `5.2%` 的 `15.4×`），同时键长 JSD 降至 `0.0951`（芳香 `C=C` `0.0207`、芳香 `C=N` `0.0152`）、键角 JSD 降 `36.6%`（`0.0516 → 0.0327`）、环尺寸 JSD 降 `47.8%`（`0.0471 → 0.0246`）、高张力三/四元环占比降 `49.4%`（`12.35% → 6.25%`）。
   - 在 `N ∈ {30, 45, 60, 80, 100}` 渐近复杂度扩展测试中（`250` 分子/桶），稠密全图 `100` 步 `FlowMol` 延迟从 `53.8s/1k (N=30)` 膨胀 `8.6×` 至 `463.1s/1k (N=100, 2.16 mol/s)` 且百原子有效连通率跌至 `14.0%`；稀疏 $O(N \log N)$ `Uni-3DAR-Mol` 在 **`N = 100` 处 `Base M=1`（`63.4s/1k = 15.79 mol/s`）提速 `7.3×` 且有效连通率更高（`22.8%` vs `14.0%`）**，`SpecTree M=2`（`280.8s/1k = 3.56 mol/s`）同时实现 **`1.65×` 更快推理速度与近 `2×` 更高百原子有效连通率（`27.6%` vs `14.0%`）**。
5. **魔改 `Uni-3DAR-Mol` 在 `PropMolFlow` `50k/50k` 无重叠 `QM9` 六性质基准上的统一验证（`qm9_uni3dar_mol_benchmark.json`）**：
   - 在严格零 `gvp_rQM_v0` 重叠协议下，仅训练 **`1` 个统一全 $SO(3)$ `K=6` `Uni-3DAR-Mol` 模型（总算力仅 `0.6636 GPU-h`，比 `PropMolFlow` 6 个专用模型的 `155.97 GPU-h` 少 `235×`）**，在完全不使用 PCA 坐标系规范化且条件器不注入 `n_atoms` 的前提下，凭借 `ScaleFreeBudgetEmbedding` 将 `M=16` 原子数误差压缩至 **`|ΔN| = 0.165–0.210` 原子**，全档位 **`0.00%` 奇异形式电荷**、`M ≥ 4` 达 **`100.0%` 化学合法率**：`M=1` 吞吐达 **`121.22 mol/s`**（比单发 `PropMolFlow` `28.73 mol/s` 快 **`4.2×`**）；`M=4`（`15.37 mol/s`）在 `gap/homo/lumo/mu` 四项全面超越 `EDM` 与 `GeoLDM`；`M=16`（`74→22→16`，`4.13 mol/s`）在强度量性质上全面反超单发 6 模型 `PropMolFlow`——**偶极矩 `μ = 0.4171 D`（低 `32.7%` vs `0.6202 D`，且较初代 `Uni-Loom S1` `0.4740 D` 再降 `12.0%`）、`HOMO = 206.61 meV`（低 `18.7%` vs `253.99 meV`）、`gap = 390.09 meV`（优于 `390.98 meV`）、`LUMO = 329.21 meV`（较 `Uni-Loom S1/S2` `341.31 / 380.21 meV` 显著改善）**，并与携带显式 `(t, N)` 条件的 `PropMolFlow` 在广延量性质（`α = 4.58 Bohr³`, `C_v = 2.26 cal/mol·K`）上形成清晰的强度量/广延量机理对照。
6. **100% Zero-Loom 独立首发版 `Uni-3DAR-Mol` 基线收官（`qm9_uni3dar_mol_standalone.json` & `oe62_uni3dar_mol_standalone.json`，单分子输出 `M_out ≡ 1`、零外部代理预测器、零 RL、零推理 MLIP）**：
   - 为支撑“先独立发表 `Uni-3DAR-Mol`（`paper/uni3dar_mol/`）、再发表 `Uni-Loom` 平台（`paper/` & `paper/ncs/`）”的双篇递进布局，`OctreeElementProcessCritic` 升级 `64`-RBF 共价连续滤波边门控（`γ = 42.0 Å⁻²`）与 `12` 通道电荷中性化欧氏偶极/四极矩等变范数读出（`h_gate_scale = 0.0` + `G=8` 正负配对目标解耦），仅凭数据集真值标签即在 `QM9` 盲配错目标验证集上取得 **`r = 0.9867–0.9989`**。
   - **`QM9` 独立单分子输出（`0.4303 GPU-h` vs `PropMolFlow` `155.97 GPU-h`，少 `362×`）**：`1→1→1` 达 **`127.61 mol/s`**，`74→22→1`（`M_out = 1`，`4.47 mol/s`，`100.0%` 合法率，`|ΔN| = 0.22–0.38`）在零外部代理模型下取得 **`ε_HOMO = 221.84 meV`（较 6 模型 `PropMolFlow` `253.99 meV` 低 `12.7%`）与 `μ = 0.4902 D`（较 `0.6202 D` 低 `21.0%`）**，`20→6→1`（`16.51 mol/s`）全面超越 `EDM` 与 `GeoLDM` 四项电子/极性指标。
   - **`OE62` 纯真值 `HOMO` 独立单分子输出与 `PBE0/def2-TZVP` 验证（`0.5604 GPU-h` vs `PropMolFlow` `52.00 GPU-h`，少 `92.8×`）**：在 `N = 1,792` 固定 `30.4` 原子测试集上全档位 **`0.00%` 奇异形式电荷**，`16→4→1`（`26.45%`）超越 `12h PropMolFlow`（`25.39%`），`64→16→1`（**`33.65%`**，`1,439` 骨架）与 `96→24→1`（**`34.38%`**，`100.0%` 合法率，`slope = 0.99`）全面反超 `52h PropMolFlow`（`30.75%`）；在 `N ∈ {20, 30, 40}` 分桶命中率全面领先（`39.6%, 35.4%, 25.4%` vs `36.4%, 33.9%, 24.3%`）且 `N=60` 单发提速 **`5.3×`**；`42`-prompt 实机 `GPU4PySCF PBE0/def2-TZVP` 端到端量子命中率达 **`33.33%`（`14/42`，为 `52h PropMolFlow` `16.67%` 的 `2.0×`）**；内部 `6` 组消融证实 `SpecTree (32→8→1)` 较 `4→4→1` 提升 **`+7.25–9.77 pp`**，`< 0.5 ms` `PostTreeGraphRefiner` 将零步 `P90 F_max` 压降 **`38%–47%`（`19.16 → 10.12 eV/Å`）**。
