# ORBIT-3D

面向性质区间的分子生成与三维优化研究。当前可执行主线为：

**预训练 Uni-3DAR octree → 区间监督适配 → 候选排序 → DPA/HyFF 约束弛豫 → 独立评价。**

GRPO 是可选实验阶段，默认关闭。另有显式骨架图增长入口和 BRICS 片段基线。
BRICS 用训练分子的规则切分与随机重组，检验简单图提案能否改善候选供给；
它没有接入 octree 作为补位逻辑。[作用与局限](docs/architecture.md#brics-的作用与局限)
当前算法见[架构](docs/architecture.md)；拟调整方向和实施状态见[方案调整](docs/design_changes.md)，
实验协议和执行状态见[实验设计](docs/experiments.md)。单值条件等下一轮方案尚未落地。

![当前五阶段架构](paper/figures/fig0_orbit3d_system_architecture.svg)

## 文档导航

当前算法、拟调整方案、实验协议和结果分别维护：

| 文档 | 内容 |
| --- | --- |
| [算法架构](docs/architecture.md) | 当前实现的表示、条件生成、性质评价、排序与弛豫 |
| [方案调整](docs/design_changes.md) | 拟调整内容、理由、实施状态及采用条件 |
| [实验设计](docs/experiments.md) | 控制变量、预算、判别标准与实验执行状态 |
| [实验结果](docs/results.md) | 最终指标、负结果、适用范围与机器可读证据 |
| [必要踩坑说明](docs/pitfalls.md) | 分子身份、物理量定义、数值精度和评测口径 |
| [运行与模型接口](docs/runtime.md) | 安装、训练/推理、外部 Oracle、可迁移模型包 |
| [数据说明](data/README.md) | 数据来源、实际划分、坐标来源、元素与分子大小 |
| [论文](paper/ORBIT3D_arXiv_Paper.pdf) | 当前研究稿；[可编辑来源与构建](paper/README.md) |

关键结果：匹配的 512 分子测试集上，DPA 的 HOMO/LUMO/S1/T1 MAE 为
0.2895/0.2326/0.2314/0.1381 eV。三组 SFT 续训的 36 次评估中，延长训练
提高了 B/Si 学习，但独立联合命中从 23/1152 变为 21/1152。BRICS 在中心
目标窗口显著强于当前 octree 基线，但实际评估调用数不同。五个预选合格设计
均完成量化优化并保图，其中两个仍命中目标。上述性质误差针对 OCELOT CSV
标签；原始量子几何和逐分子 ω 缺失，不能据此声称已复现原始 DFT。

## Codemap

```text
mol_gen/
├── README.md                    # 项目入口、实验概况与本 codemap
├── pyproject.toml / uv.lock      # 包定义、可选后端与锁定依赖
├── Dockerfile                   # 可选部署配方；不等于已验证的平台 GPU 镜像
├── .github/workflows/           # CPU 正确性与安装包检查
├── configs/                     # octree 双/四性质与显式骨架工作流配置
├── src/orbit3d/
│   ├── cli.py                   # orbit3d --config 的统一入口
│   ├── pipeline/                # 后端分派、octree/骨架流程、共享评估器加载
│   ├── generator/
│   │   ├── molecular_state.py   # 分子图、元素/电子态、坐标与化学验证
│   │   ├── octree_model.py      # 条件 octree、自排序与完整轨迹概率
│   │   ├── octree_backend.py    # 数据编码、检查点契约、预训练/适配模型加载
│   │   ├── octree_alignment.py  # conditioner 的 GRPO 更新
│   │   ├── scaffold*.py         # 显式连接/环闭合、开放位点、局部 XYZ 支持集
│   │   └── 其他模块             # 空间窗口、独立随机流、条件与词表迁移
│   ├── baselines/brics.py       # 训练片段库与显式配对基线
│   ├── oracles/                 # DPA/HyFF 属性头、指纹、域判断、校准及可选外部模型
│   ├── physics/                 # DPA/HyFF 势、串行/批量受约束弛豫
│   ├── quantum/                 # PySCF/GPU4PySCF、ω 调参、激发态与几何检查
│   ├── utils/                   # 数据、身份比较、指标、运行快照与整批重放
│   ├── vendor/                  # 固定版本 Uni-3DAR 及其必要依赖
│   └── compat/hyff/             # 打包的 HyFF 源码、注册表与许可证
├── scripts/                     # 可重用实验与数据工具，按功能见下表
├── tests/                       # 当前功能的回归测试，防止清理/改造破坏已验证行为
├── data/                        # 数据 provenance、划分/目标协议、结构组成统计
├── docs/
│   ├── architecture.md          # 当前实现的算法架构
│   ├── design_changes.md        # 方案调整、实施状态和采用条件
│   ├── experiments.md           # 实验协议、执行状态及证据入口
│   ├── results.md               # 最终结果与解释
│   ├── pitfalls.md              # 必要方法与数值注意事项
│   ├── runtime.md               # 实际运行和模型接口
│   └── experiments/             # 最终 JSON、必要原始结果/哈希、量化 SDF/NPZ
├── paper/                       # 文稿、当前图、同步证据和可重建 arXiv 源码包
├── third_party/                 # 运行依赖的固定版本、来源和许可证
├── dataset/                     # 本地原始/处理数据，不进入 Git
├── pretrained/                  # 本地模型权重，不进入 Git
├── external/                    # 可选上游参考检出，不进入 Git
├── local_dev/                   # 仅保留私有服务器/模型位置与量化启动配置，不进入 Git
└── dist/、outputs/、checkpoints/ # 构建及运行时产物，不进入 Git
```

`scripts/` 保持平面布局，实验编排脚本通过同目录调用共享工具：

| 功能 | 主要入口 |
| --- | --- |
| 数据获取/准备 | `manage_data.py`、`inspect_mim_*.py`、`adapt_qm9s.py`、`prepare_ocelot.py`、`prepare_generator_sdf.py` |
| 性质训练/评价 | `train_ocelot_3d.py`、`train_ocelot_baseline.py`、`calibrate_ocelot.py`、`evaluate_trained_oracle.py` |
| 生成训练与迁移 | `train_octree_structural.py`、`migrate_octree_*.py`、`run_sft_duration_campaign.py` |
| 对照实验 | `evaluate_target_windows.py`、`evaluate_brics_baseline.py`、`run_grpo_factorial.py`、`ablate_*.py` |
| 独立评价/汇总 | `evaluate_generated*.py`、`summarize_*.py`、`audit_*.py` |
| 量化验证 | `select_qualified_quantum_cohort.py`、`validate_quantum.py`、`compare_quantum_oracles.py`、`optimize_excited_states.py` |
| 梯度/执行验证 | `compare_excitation_gradients.py`、`validate_*.py`、`profile_octree_sampling.py` |
| 打包与重放 | `check_wheel.py`、`pack_pretrained_weights.py`、`export_inference_bundle.py`、`validate_octree_reload.py` |

## 最小运行

本机 macOS 做 CPU 测试和文稿构建；真实 GPU 实验使用远程 Linux/CUDA 环境。

```bash
uv sync --frozen --extra octree
uv run --frozen --extra octree pytest tests -q
uv build --wheel
python scripts/check_wheel.py
```

DPA 后端使用 `uv sync --frozen --extra octree --extra deepmd`；HyFF 使用
`--extra hyff`，量化依赖单独配置。数据与权重需自行放入相应目录。
编辑配置中的实际路径后运行：

```bash
orbit3d --config configs/qm9s_ocelot_octree.json
# 四性质：configs/ocelot_optical_octree.json
# 固定骨架：configs/scaffold_graph_growth.json
```

推理设置 `mode: predict` 并提供适配检查点；`mode: train` 执行监督训练，
`mode: align` 单独执行 GRPO。完整参数、电子态范围和验证边界见
[运行说明](docs/runtime.md)。历史结果绑定各自的代码与模型哈希，不因当前源码
整理而改写；构建论文会从保留的证据重新生成表格。
