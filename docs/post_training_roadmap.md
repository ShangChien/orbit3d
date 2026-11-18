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

## 3. 路线二工作计划：8×A800 通用 `FlowMol` 预训练底座与 `Hybrid Flow-GRPO`

> **工程基座选型原则**：预训练底座直接基于干净的无条件 $SE(3)$ 等变 **`FlowMol / FlowMol3`** 主干构建（弃用原版 `PropMolFlow` 窄词表、单属性硬拼接外壳；也不以仅面向二维图像潜空间的官方 `Flow-GRPO` 仓库做分子底座），后训练统一收敛在 `Uni-Loom` 的 `LoomFlowMolGenerator` 与 `LoomGRPOTrainer` 内。

### 3.1 数值精度规范（面向 8×A800 Ampere 节点）
- **标准配置**：启用 **`BF16` 自动混合精度（AMP）+ `FP32` 主权重/优化器状态 + `TF32` 矩阵乘兜底**（A800 不支持硬件 FP8；不采用需动态缩放的 FP16）。
- **强制局部 `FP32` 白名单（`torch.autocast("cuda", enabled=False)`）**：
  1. 三维成对坐标差、欧氏距离 `||x_i - x_j||`、RBF 径向基展开与方向单位向量归一化；
  2. GVP 等变向量通道的 $L_2$ 范数与叉乘特征（`n_cp_feats`）；
  3. 质心扣除（COM removal）与先验旋转/平移最优传输对齐；
  4. 插值调度系数 $\alpha_t, \dot{\alpha}_t$、CTMC 掩码交叉熵归约，以及后训练中的优势指数权重或 SDE-CTMC 步进 log-prob 累加。
- **准入准则**：全量开训前须完成 `2,000` 步 `纯 TF32` vs `BF16+局部FP32` 对照，验证 `loss_x`、`loss_e`、梯度范数及生成最小原子间距分布一致。

### 3.2 8×A800 通用预训练分阶段排期
1. **Stage 1 — 词表与可插拔条件架构定型（约 2–3 天）**：
   - 一次性固定 `16–20` 种常见元素统一大词表 + 独立专用 `[MASK]` 索引（消除原版复用 `'Se'` 作为未解掩占位符的冲突），固定 `n_atom_charges=6, n_bond_types=5`，下游通过 `allowed_elements` 掩码动态裁剪；
   - 主干纯学无条件分布 $p_\theta(\mathbf{X},\mathbf{A},\mathbf{C},\mathbf{E})$ + 几何自条件（Self-Conditioning），预留零初始化 AdaLN-Zero / Cross-Attention 多属性条件编码器（对齐 `Uni-Loom` 的 `target_cols` + `active_mask` 协议）与 GVP 线性层 LoRA 接口。
2. **Stage 2 — 多源三维语料整合与动态边数分桶（8 卡并行预处理约 0.5–1 天）**：
   - 整合公开三维构象集（QM9 + GEOM 构象集 + OE62 + Uni-Mol 公开有机分子子集），统一显式氢、键级与形式电荷；
   - 实现 **Dynamic Edge-Budget Bucketing**（按原子数分桶、按单卡全连接总边数上限动态组 batch），消除 8 卡 DDP 中大分子拖慢小分子的同步空转。
3. **Stage 3 — 8×A800 预训练主循环与少步求解适配**：
   - **标准底座（~12M 参数）**：8 卡吞吐预估 `2,600–3,200 mol/s`，训练 `6,000 万–8,000 万` 分子-时间步样本，耗时约 **`6–9 小时`**；
   - **旗舰底座（~40M–50M 参数）**：扩宽 GVP 标量/边通道与更新层数，8 卡吞吐预估 `800–1,100 mol/s`，训练 `1.5 亿–2 亿` 样本，耗时约 **`2–3 天`**；
   - 末期增设非均匀时间网格与二阶 Heun 求解器适配，使推理步数从 `100` 步压缩至 **`20–25` 步**（单卡吞吐提升至 `~35–45 mol/s`）。
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

## 5. 实证收官基线（2026-11-19）：OE62 严格同尺寸公平对比、A800 八叉树加速、全谱系 `Loom-AWR`、跨原子数扩展与在线 `GPU4PySCF` 闭环

为定量锚定上述双底座路线图的起点，已在单节点 NVIDIA A800-SXM4-80GB（对称开启 `TF32`）上完成全套跨范式公平对照、多原子数扩展及第一性原理闭环实验（原始 JSON 归档于 `docs/experiments/`，汇总曲线见 `docs/figures/oe62_llm_style_accuracy_vs_cost.png` 与 `docs/figures/oe62_atom_scaling_speed_and_mae.png`）：

1. **`Uni-3DAR` 专属 A800 算子与流式 KV 剪枝优化 + 单调 `8→4` 漏斗（`oe62_unanchored_with_8to4.json`）**：
   - 在 `Uni-3DAR` 内部实现预分配缓冲 KV-Cache、`SDPA` FlashAttention、向量化 `RoPE 3D`、流式分块叶节点续写（`Process 16→4` 峰值显存由 `79.3 GiB` 压降至 **`25.4 GiB`**）及单调 `m_keep=4` 漏斗（淘汰过窄剪枝的 `16→2`）：无锚定 `Uni-Loom [Uni-3DAR]`（`3.0 GPU-h`）在 `M=1 → 8→1* → 8→2 → 8→4 → 16→4` 上实现严格单调递增的盲审可合成命中率（`13.7% [38.26 mol/s] → 22.6% [19.57 mol/s] → 33.4% [15.40 mol/s] → 38.9% [12.08 mol/s] → 43.9% [8.03 mol/s]`，`68.6% RewSynth`，`MAE = 0.390 eV`，无偏 42-prompt `PBE0 SynthHit = 50.0% raw`），`50k` 分子筛查总成本仅 **`4.73 GPU-h`**（较旧实现提速 `2.27×`）。
2. **严格固定每条 Prompt `30.4` 原子的跨范式公平对决（`N = 1,792`，每条 `(HOMO_i, N_atoms_i)` 跨全部 `M ∈ {1..8}` 与双底座严格锁死）**：
   - 消除此前 `PropMolFlow` 在 `M=8` 下跨 8 次随机尺寸抽样向下挑小分子（`37.5 → 30.0` 原子）的协议偏差后，PCA 极性规范化 + 事后重标定 AWR + 解耦坐标蒸馏的 **`Uni-Loom [Uni-3DAR, 30-Atom Matched]`（仅 `2.45 GPU-h`）彻底翻身**（`oe62_uni3dar_v3_fair_30atom.json`）：
     - 过程漏斗严格单调上升：`M=1`（`28.19 mol/s`, `10.0%`）`→ 8→1*`（`9.55 mol/s`, `21.4%`）`→ 8→2`（`7.13 mol/s`, `31.0%`）`→ 8→4`（`4.82 mol/s`, `37.7%`）**`→ 16→4`（`3.51 mol/s`, 均原子数 `30.34`, `|ΔN| = 1.57`, `99.9%` 合法, `93.2% SA<=3.5`, 中位 `Fmax = 2.88 eV/Å`, `1,399` 唯一 Murcko 骨架）**，实现 **`62.4% RewSynth (1,118/1,792)`** 与 **`42.5% BlindSynth (762/1,792)`**（`MAE = 0.481 eV`）——不仅大幅超越同算力 `3.0h PropMolFlow M=8`（`30.9% BlindSynth`, `1.93 mol/s`，快 `1.82×`）与 `3.12h PMF + Loom-AWR`（`36.1%`），且在无偏 42-prompt `PBE0/def2-TZVP` 终验中实现 **`39/39 (100%)` SCF 收敛、端到端 `PBE0 SynthHit = 42.9% raw`（`PBE0 MAE = 0.361 eV`，`DPA-4 vs PBE0 MAE = 0.091 eV`, `r = 0.985`）**，以 **`21.2×` 更低训练算力** 追平 `52h PropMolFlow` 原始 `PBE0` 命中率（`42.9%`）。
3. **从头流匹配扩展律与热插拔全谱系在线 `Loom-AWR`（`3h / 12h / 52h` → `+0.12 GPU-h = 7.4 min`）**：
   - 在严格同 Prompt `30.4` 原子基准下（`oe62_pmf_{3h,12h,52h}{,_awr}_fair_30atom.json`），单发 `Vanilla PropMolFlow M=1` 受制于最大似然对高 `SA` 先验的忠实复刻，在 `52h` 处仍停滞于 `15.2%–15.6% BlindSynth`（`SA<=3.5` 仅 `42.4%`）；挂载 `Loom-Critic (M=8)` 后盲审命中率对数扩展至 `30.9% (3h) → 48.4% (12h) → 52.9% (52h, MAE = 0.298 eV)`。
   - 进一步热插拔 `56` 步原生合法门控 + `20` 步 `DPA-4` 弛豫复核的在线 **`Loom-AWR`（仅 `+0.12 GPU-h`）** 在全部算力档位均带来 **`+5.2 ~ +7.3 pp`** 跃升：`3.12h` 达 **`36.1% BlindSynth`**；**`12.12h` 跃升至 `55.6% BlindSynth`（`81.8% RewSynth`, `MAE = 0.325 eV`），以 `4.3×` 更低训练算力（节省 `40 GPU-h`）正面超越 `52.0h` 从头流模型基座（`52.9%`）**；`52.12h` 刷新上限至 **`59.8% BlindSynth (1,072/1,792)`** 与 **`84.5% RewSynth (1,514/1,792)`**（`MAE = 0.283 eV`，无偏 42-prompt `PBE0 HOMO MAE = 0.270–0.288 eV`、`PBE0 SynthHit = 54.8% raw`）。
4. **受控跨原子数渐近扩展对比（`N ∈ {20, 30, 40, 50, 60}` 精确原子数，`280` prompts/桶，`oe62_atom_size_buckets_scaling.json`）**：
   - 从 `N=20` 增至 `N=60`，受 $O(N^2)$ 全连接图 100 步 ODE 制约的 `PropMolFlow M=8` 每千分子耗时从 `217.9s` 飙升 **`7.1×`** 至 `1,538.5s`（`0.65 mol/s`），而稀疏八叉树 `Uni-Loom [Uni-3DAR]` 每千分子耗时仅 `124.1s → 497.5s`（`16→4`）与 `86.3s → 401.6s`（`8→4`），相对加速比从 `1.8×` 单调放大至 **`3.1×`（`16→4`）与 `3.8×`（`8→4`）**；在大分子 `N=50` 桶上，`52h` 基座 `SA<=3.5` 跌至 `26.1%`（`6.4% BlindSynth`，低于 `2.45h Uni-3DAR 16→4` 的 `8.6%`），而 `Loom-AWR` 将 `N=50` 盲审可合成命中率拉升 **`3.1×` 至 `19.6%`**。
5. **实机 `GPU4PySCF PBE0/def2-TZVP` 在线外环 `Grounded RSI` 与均匀信任锚点分层（`v20_grounded_rsi_live_pbe0_eval.json`）**：
   - 将外环采集的 `64` 个全新生成盲区分子经 `25` 步 `DPA-4` 弛豫后直接送入在线 `GPU4PySCF PBE0/def2-TZVP/DF` 求解器，实现 **`64/64 (100.0%)` SCF 收敛**（单分子均耗时 `0.926s`，总量子耗时 `70.54s`），零基座重训将独立深能级 DFT 盲测集误差由 `0.990` 压降至 `0.823 eV` 并清零 `OUTER_BLINDSPOT (1 → 0)`；配合在全部已信任分位点均匀分层分配 `20%` 锚点回放预算，两轮闭环盲审清洁命中总数跃升 **`9.65×`（`23 [1.28%] → 151 [8.43%] → 222 [12.39%]`）**，深能级盲区命中率提升 **`8.17×`（`1.17% → 9.57%`）**，同时将浅能级尾端（`-5.85 eV`）均值漂移削减 **`80%`**（`-6.86 eV → -6.055 eV`，收束至 `±0.25 eV` 容差带内）。
