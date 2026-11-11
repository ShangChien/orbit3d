# QM9s / OCELOT 真实实验与主线修复

2026-11-11。本轮完成数据适配、性质训练、真实预训练八叉树接入，以及远程
生成/筛选/弛豫/重载实验。主线为 `uni3dar_octree`，dense-grid 留作对照。
以下数字来自实际输出，机器可读汇总见
[experiment JSON](experiments/qm9s_ocelot_20261111.json)。原始逐候选表、SDF、轨迹、
日志和配置保存在 `local_dev/review_evidence/real_experiments/`，权重保留在服务器。

## 数据与性质模型

QM9s 适配生成2048/256/256个 train/dev/test 结构，采用固定哈希选择，训练排除
原 valid 中的所有同结构记录。数据适配保持显式H原子顺序、键级和立体身份；
原始240通道光谱单独保存，不冒充HOMO/LUMO。新的八叉树适配直接保留源坐标，
不再依赖 dense-grid 的量化往返。选出的三个 QM9s 划分与 OCELOT 三个划分之间，
9组规范化结构交集均为0。

OCELOT 的目标为 HOMO/LUMO（eV），划分按骨架分组。原CSV未提供DFT坐标，
3D模型使用 ETKDG+MMFF 输入几何。两个模型的训练/测试规模不同，不能按下表
宣称3D模型优于指纹模型。

| 模型 | train / valid / test | 测试 HOMO MAE (eV) | 测试 LUMO MAE (eV) | 训练均值基线 MAE (H/L) |
| --- | --- | ---: | ---: | --- |
| Morgan + 描述符，3个 bootstrap ExtraTrees | 15283 / 1622 / 1734 | 0.37036 | 0.34945 | 0.50655 / 0.51316 |
| 冻结 DPA4 描述符，3个独立 bootstrap adapter | 512 / 64 / 64 | 0.29462 | 0.26216 | 0.49614 / 0.50772 |

两种模型重载后的预测最大差均为0。指纹模型 summary 中的9.53秒是重用已有
权重的评估耗时，不能作为训练时间。3D pilot 训练与评估合计约202秒（CPU）。
独立 bootstrap 是不同训练重采样和完整adapter权重，尚不等于校准后的置信区间。

## 直达根因的修改

1. **生成表示。** 放弃把小型 dense-grid 从头训练当作主路线。一个256样本实验
   虽然31/32个输出有效，但仅8种结构，且相对训练集的新颖数为0；更大的2048样本
   实验在epoch46出现非有限loss。现在直接复用真实 Uni-3DAR EMA 权重，冻结主干，
   训练条件器和结构理解头。必需源码已固定进安装包，无外部私有路径导入。
2. **性质坐标梯度。** DeepMD 力推理将边向量 detach，旧路径实际只微分了adapter
   距离分支。改用保留坐标计算图的 embedding 路径。4个真实OCELOT结构的普通/
   可微预测一致，自动微分与有限差分对应值为1.22580/1.22583、1.04655/1.04666、
   -1.25410/-1.25408、1.63041/1.62995。该检查验证计算图，不证明DFT梯度准确。
3. **GRPO 概率。** 完整纳入tree、element、X/Y/Z五类决策，current-policy KL保留
   梯度，rollout与训练保持相同温度/支持集。真实权重4个轨迹的缓存采样与全前缀
   重算平均每决策log-prob最大误差为6.36e-7（FP32）。SFT同样使用允许元素集合，
   不再夹紧错误标签后继续训练。
4. **确定性。** 只设seed不足以固定CUDA归约。原重载有一个残余力相差9.04e-5
   eV/Å；启用确定性算子后，64个候选和16个导出结构全部通过原1e-5容差检查。
5. **部署与计算。** 新检查点保存完整生成器、目标scaler与训练结构身份，推理
   不依赖原始预训练文件。共享DPA模型的接受测试一次计算性质与原始力；已在区间
   内时不再计算必为零的hinge梯度，逐目标接受规则保持不变。

## 最终主线端到端运行

在直接八叉树适配的数据上，自动生成2048条Oracle伪标签，完成100步条件器/
结构头SFT，然后以64个提案、32个外部排序评估、16个保留名额、30步弛豫运行。
本轮配置为 `configs/qm9s_ocelot_octree.json`，GRPO默认关闭。

| 指标 | 实测 |
| --- | ---: |
| 原始化学有效 / 全部提案 | 35 / 64 |
| 原始有效结构的唯一数 | 35 |
| 保留结构 / 不同结构数 | 16 / 16 |
| 全部可解析 QM9s 源结构中未出现的保留结构 | 14 / 16 |
| 优化Oracle双目标命中 | 15 / 16 |
| 化学+双目标+力阈值联合合格 | 14 / 64（21.875%） |
| 平均最终RMS力 | 0.17598 eV/Å |
| 独立指纹模型双目标命中 | 10 / 16 |
| 独立指纹模型+力阈值联合合格 | 9 / 64 |

另扫描全部QM9s源 train/valid：129,813条记录中有1159条SMILES不可解析，剩余
128,559种规范化结构作为参考。16个输出中14个不在该参考中，其中12个同时达到
优化Oracle与力阈值要求。该参考集合并非已核对的上游模型预训练清单。

训练后内存模型与独立加载的检查点，在64个候选和16个输出结构上全量一致
（预测/力及坐标容差1e-5）。重载配置移除了原始预训练文件与训练SDF路径；
新检查点自身提供模型、scaler和适配训练身份。检查点SHA-256为
`533557774d12922cd1cc2a854a8383efb033e9f00c0d75ac4f3616d080a70dd7`。

这次完整运行约270秒，包含伪标签、SFT和生成；独立推理约82秒。两者工作内容
不同，且GPU存在竞争，不能据此计算训练/推理加速比。下表是此前同数据同预算的
GRPO消融；最终主线的数据适配和SFT元素mask已更新，不能混作同一组对照。

## 同预算 SFT / GRPO 对照

这一组使用相同的早期2048条适配数据、seed=2026、64个原始提案、32个外部排序
评估、16个保留名额和最多30步弛豫。区间为HOMO [-8.2,-7.2] eV、LUMO [-0.6,0.8]
eV；联合合格要求化学有效、两个目标同时命中、原始RMS力≤0.5 eV/Å。
自排序使用屏蔽条件的结构头；外部Oracle再排序的预算显式计入。

| 路线 | 原始有效/64 | 原始有效唯一数 | 保留后性质命中/16 | 联合合格/64 | 平均最终RMS力(eV/Å) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 35 | 35 | 16 | 16 (25.0%) | 0.10052 |
| SFT + 20步 GRPO | 41 | 41 | 11 | 9 (14.06%) | 0.53123 |

两组各16个保留结构均不在适配训练集；这不是针对完整QM9预训练语料的新颖性。
GRPO提高了这一批的原始有效数，却降低最终联合合格数，因此默认配置关闭GRPO。
这是一组试验结果，不能由此断言GRPO在其他区间或seed上普遍无效。

独立OCELOT指纹模型对两组分别给出10/16、10/16个性质命中；加上力阈值后，
分别为10/64和8/64个联合合格。它未参与这一组优化，但仍是代理模型，不能替代
量化计算。SFT的HOMO平均模型间差异约0.54 eV，说明Oracle域外可靠性仍需优先解决。

## 工程验收与复现

本地锁定环境和远程科学环境各33个测试通过。安装包包含Uni-3DAR源码/许可及
HyFF manifest，并在源码树之外验证了导入。另一次关闭Stage 5的真实测试保留4个分子：迭代轨迹为空，最终原始力与初始值
完全一致且非零，联合合格数为0，没有把“未弛豫”记为“零力成功”。
运行环境存在其他GPU负载，所以
本轮耗时不能作为A800吞吐基准。远程NumPy版本与本地锁定环境不同，记录均已保存。

主入口及配置为 `orbit3d --config configs/qm9s_ocelot_octree.json`，先替换挂载路径。
`label_source: oracle` 会对QM9s结构生成明确标注来源的OCELOT Oracle伪标签。
完整复核使用 `scripts/validate_octree_reload.py`；独立模型复核使用
`scripts/evaluate_generated.py`。每次实验使用新的输出目录保存源码快照。

## 仍未建立的结论

当前证据支持流程正确运行和上述有限范围内的实测结果。尚未完成多seed/多区间
泛化、GRPO收益、校准不确定性、独立DFT验证、完整MR-TADF/广元素/开壳层/局部骨架
生成。主线代码修复和这些研究结论分开追踪，见 `TODO.md`。

已找到[MIM-Flow官方关联发布](https://zenodo.org/records/21429926)，但网页当前
`data.tar.gz`的MD5与本地已完整校验的3.25GB文件不同，尚不能把该URL登记为精确
来源。该版本差异已记录于 `data/mim_flow_archive.json`，没有替换现有实验输入。

## Continued implementation and negative results

The 4096-reference-record generator continuation ran8000 additional steps with
the inherited target scaler. Its64 attempts yielded32 valid structures,5 domain-
eligible/retained structures and0 optimizing-oracle joint successes. An independent
fingerprint model found3 interval hits and1 joint success, illustrating evaluator
disagreement. Mean SA score was3.17 (maximum4.14); this is a heuristic, not a
synthetic route. `experiments/ocelot_generator_continued_20261111.json` preserves
the counts, checkpoint hash and independent results.

The sole retained molecule from the earlier1024-record/4000-step model completed
an explicit GPU4PySCF1.9.1 IP-tuned LC-wPBE/def2-SVP optimization and full S1/T1
TDDFT. Its graph survived; HOMO=-8.81566 eV missed the original target despite the
optimizing surrogate's hit. This is a fixed-cohort negative diagnostic, not a
validated reproduction of OCELOT's undocumented per-molecule protocol. The report
is `experiments/generated_quantum_diagnostic_20261111.json`.

The official DeepMD3.2 Python backend now installs from locked sources with
Torch2.7.1, e3nn and Warp/nvalchemi neighbor lists. Its32 fixed-geometry property
and force outputs differ from the previous experimental runtime by at most
2.86e-6 eV and2.80e-6 eV/Å. Native coordinate gradients pass finite differences,
and a32-candidate Stage5-off replay passes with all tokens and the sole exported
molecule checked. Source snapshots separate this clean installation from the
original experimental environment. Docker execution remains untested.

## Projection-preserving step cap (2026-11-12)

Per-atom displacement clipping rotated a globally projected direction: for
`g=[2,1]`, `d=[1,-2]`, the tangent has `g·d=0`, but clipping each atom to
0.1 gives `g·d=0.1`. The cap now applies one positive scale per molecule using
its largest atom displacement, so direction, frozen coordinates and projection
are preserved. Independent finite-step Oracle acceptance remains in place.
The regression failed on the original clipping and passes after the change.

On eight 30-atom OCELOT structures, real GPU serial and batch refinement agree
on acceptance and interval decisions, with coordinate/property tolerances
1e-5 and raw-force tolerance 1e-4 eV/A. Timings were 13.67 s serial and 2.78 s
batched on the shared server; this is not an uncontended throughput benchmark.
See [the per-molecule comparison](experiments/projection_scale_serial_batch_20261112.json).

## Matched conditioning and parity experiment (2026-11-12)

Both conditioner variants start from the same 4096-record/8000-step structural
prior, reset conditioning, freeze the backbone, and train for 4000 steps on the
same labels with training seed 42. V3 changes both Fourier frequencies and the
interval-sampling distribution; the comparison does not isolate these two
factors. Each of three inference seeds uses 64 attempts, at most 32 external
ranking candidates, 16 requested outputs and 30 refinement steps. All runs use
the same applicability-domain filter and FP32 sampling.

| Conditioner | Terminal even-electron constraint | Valid / 64 (mean) | Optimizing joint successes / 192 | Independent surrogate joint successes / 192 |
| --- | --- | ---: | ---: | ---: |
| v2 reset | off | 30.33 | 6 | 6 |
| v2 reset | on | 56.33 | 6 | 6 |
| v3 smooth | off | 27.00 | 3 | 3 |
| v3 smooth | on | 52.67 | 4 | 4 |

The parity constraint addresses the dominant odd-electron rejection mode and
raises chemical validity, but it does not establish a joint-hit improvement.
V3 smooth conditioning improves local numerical continuity but has not improved
generation success over the matched v2 control. The independent evaluator is
the separately trained fingerprint model, not DFT. The three seeds are inference
seeds of one trained checkpoint, not three independent training replicates.
[Raw per-run summaries, budgets, hashes, independent predictions and SA scores](experiments/conditioning_factorial_20261112.json)
are retained. GRPO follows as a separate matched-budget experiment.

## Explicit tree-depth migration (2026-11-12)

Depth 7 doubles the octree box side from 15.36 to 30.72 A while retaining
0.24 A leaf cells and 0.01 A within-cell resolution. Existing level embeddings
and all other weights remain bitwise identical; the added root row copies the
former root embedding. Checkpoints store the depth explicitly and require
backbone adaptation before conditional inference. Changing physical bin size
is not part of this migration.

The 80-atom OCELOT preparation uses the same scaffold hash split as the original
40-atom preparation (19,902 train / 2,236 valid / 2,275 test). All original
training identities stay in train; 4,619 additional training molecules exceed
40 explicit atoms. The adaptation subset contains 512 molecules, including
146 with more than 40 atoms and maximum diameter 27.1145 A. Every molecule
retains its atom count after encoding; the maximum pair-distance error is
0.013293 A, below sqrt(3) times the 0.01 A resolution. A separate regression
checks rejection of atom loss from duplicate leaf-cell occupancy.

[Representation evidence](experiments/octree_depth7_representation_20261112.json)
validates the mapping and weight migration, not sampled molecular quality.
The migrated model is undergoing 1000 backbone-adaptation steps on these 512
true-label structures. Generalization beyond the original property-model
training domain still requires separate measurement.

## Domain-aligned FP32 GRPO, 50 steps (2026-11-12)

The v3 smooth checkpoint was aligned for 50 steps, 16 proposals per group
(800 training rollouts), with the same terminal even-electron constraint and
Morgan-domain filter used at deployment. OOD structures receive the explicit
invalid reward and do not query the property model. Weights are KL 0.02,
physics 0.2, replay 0.1 and uncertainty 0, with learning rate 2e-5.

On the same three inference seeds and 64-attempt budget with parity enabled,
SFT gives joint successes `[3, 0, 1]` and the aligned model gives `[2,3,3]`: the
mean rate changes from 2.08% to 4.17%. Independent fingerprint joint counts
change from `[2,0,2]` to `[2,4,2]`, also 4/192 to 8/192 overall. Chemical-valid
counts for GRPO are `[53,55,53]`; average domain-eligible count increases from
7.0 to 11.33 per run. These are exploratory observations from one training seed
and three inference seeds, not a demonstrated replicated-training effect.

Disabling the terminal parity mask at inference lowers validity but yields
the same aggregate 8/192 joint successes for this aligned checkpoint. That
mask therefore still primarily addresses chemical rejection.
[Per-run results and independently scored cohorts](experiments/grpo50_factorial_20261112.json)
are archived with input hashes and the alignment reference checkpoint.
