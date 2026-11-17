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
| **7. 局部三维空间条件控制**<br>（口袋限域 SBDD、3D Linker、母核修饰） | **通过“空间排斥/吸引势能偏置 + 锚点序列前置 + 层级强制分裂”原生支持**；且在**未知长度 3D Linker 架桥**中无需预猜连接原子数 $N_{\text{linker}}$（需配 4–8 路随机旋转增强消除网格边界效应） | **天然支持任意三维坐标/类型轨迹锁死（Inpainting）**，严格等变无需旋转增强；但在 3D Linker 任务中需枚举或额外预测连接原子数 $N_{\text{linker}}$ | **双方形成优势互补**：<br>• 变长桥接/大口袋限域：**`Uni-3DAR` 更灵活**<br>• 刚性锚定/严格等变微调：**`Flow` 更直接** |

### 核心战略判断
1. **`Uni-3DAR` 插槽是 `Uni-Loom` 综合性价比最高的主力引擎**：在全局性质导向生成、高通量树剪枝筛选、大尺寸分子扩展，以及结合空间势能偏置与粗轮廓截断 RL 后的变长局部三维设计（如未知原子数 Linker 与口袋空腔填充）上，具备训练快、显存省、原子数自适应的系统级优势。
2. **通用预训练 `FlowMol + Hybrid Flow-GRPO` 插槽是高价值的正交第二引擎**：在要求严格解析 $SE(3)$ 等变性、固定节点数的精细连续构象生成、显式键级协同演化，以及证明 `Uni-Loom` 跨生成范式普适性（Autoregressive + Continuous-Discrete Flow）的平台定位上不可替代。

---

## 2. 路线一工作计划：`Uni-3DAR` 局部三维空间控制与粗轮廓分层 RL

依托现有代码中的 `geometric_guard`（`src/loom/weaving/uni3dar_slot.py`）、`_stop_at_leaf_boundary`、`topology_only_policy`（`src/loom/weaving/grpo.py`）及 `OctreeProcessCritic`，按两个增量模块推进：

### 2.1 局部三维空间条件控制模块（Pocket & Substructure Conditioning）
1. **外部空间势能场偏置（免改权重，推理与 RL 即插即用）**：
   - **粗层分裂级（`tree_logits`）**：输入外部口袋重原子坐标或禁入体积，对完全落入范德华排斥球内的子八分体组合掩码直接施加 `-inf` 或软体积排斥偏置，在粗分辨率提前剪除碰撞分支。
   - **叶子原子级（`atom / xyz logits`）**：扩展 `_element_guard_logit_bias` 与 `_zbin_guard_logit_bias`，同时支持外部口袋排斥势与药效团（氢键给受体、疏水中心、锚点悬空共价键锥区）吸引势偏置，并保持 `trajectory_distributions()` 对偏置后策略概率的精确重放。
2. **内部固定子结构锚定（Anchor Prefix + Forced Octree Clamping）**：
   - 将已知锚点原子的三维坐标平移至八叉树包围盒中心，编码为前端条件前缀 Token（或启用 `uni3dar.py` 原生 `num_tree=2` 条件树通道），消除因果自回归对后置锚点的注意力盲区。
   - 在八叉树逐层展开时，强制激活包含锚点原子的粗层体素，并在叶子层通过 Teacher-Forcing 直接填入锚点元素与坐标分箱，仅对自由生长区做策略采样与梯度更新。
   - 推理侧默认支持 `4–8` 组刚体随机旋转批量并行采样 + `ValenceReconciler` 择优，抹平笛卡尔体素边界朝向效应。

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
