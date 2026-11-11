# MIM-Flow 与 OCELOT 测试数据方案

2026-11-11 根据用户决定，优先使用 MIM-Flow 与 OCELOT。DB_chromophore 和其他
figshare 表暂缓进入测试，保留原文件，待独立清洗。两套优先数据按任务分别使用，
不直接拼接标签或混合统计测试成绩。

## 已完成的准备

MIM-Flow 原压缩包为 3,248,316,986 字节，完整 gzip CRC 检查通过，SHA-256 为
`ded9b275805df1976b3c8e666495f723a4d34c8440890b9f535ee52fc22ca43d`。
已在服务器新目录中提取 downstream 数据，保留原始 LMDB 的 train/valid 划分，
没有加载或改写模型权重。文件清单与逐文件哈希见 `mim_flow_archive.json`。
这确认下载产物的完整性与身份；尚未与发布方独立校验和核对。

| MIM-Flow 部分 | train 记录 | valid 记录 | target 长度 | 跨 train/valid 重叠的规范化 SMILES 数 |
| --- | ---: | ---: | ---: | ---: |
| QM9s | 123,323 | 6,490 | 240 | 12 |
| de_novo_gen / mrtadf_norm_350 | 15,158 | 1,371 | 450 | 0 |
| local_gen / mrtadf_norm_350 | 371 | 41 | 450 | 1 |

上述 target 为向量，伴有 spectrum 字段。坐标保存在构象列表中，需显式处理
`[N×3]` 的容器结构，不能把它误判成损坏的坐标矩阵。光谱横轴、单位、归一化
和 target 的确切物理定义仍须对照发布代码确认，不能根据维度猜测。

OCELOT 原始 CSV 有 25,251 行，README 指明数值单位为 eV。主测试字段为
`homo`、`lumo`、`hl`，并单独保留 `s0s1`、`s0t1`。检查中 `hl` 与
`lumo - homo` 的最大差约为 6e-9 eV，三个量不能当作相互独立的条件随意组合。
数据的官方目录页为 [ALCF/MDF 的 OCELOT 发布记录](https://acdc.alcf.anl.gov/mdf/detail/ocelot_chromophore_v1_v1.1/)，
DOI 为 10.18126/iqex-p27w；本地 CSV 与具体发布文件的对应、下载地址和许可仍需补齐。

已生成小分子兼容子集 `dataset/processed/ocelot_v1_dense40_seed42/`：

| 去向 | 行数 | 处理 |
| --- | ---: | --- |
| train.csv | 15,283 | 训练候选集 |
| valid.csv | 1,622 | 调参与模型选择 |
| test.csv | 1,734 | 固定留出，不能用于调参 |
| excluded.csv 中的词表外元素 | 548 | 单独保留，不映射为碳 |
| excluded.csv 中的冲突标签 | 104 | 同一规范化 SMILES 的性质不一致，整组隔离，不平均 |
| excluded.csv 中的自由基 | 1 | 当前中性闭壳层基线不支持 |
| duplicate_aliases.csv | 6 | 标签一致的重复记录保留别名映射 |
| deferred_size.csv | 5,953 | 超过当前 40 个显式原子上限，保留供扩展实验使用 |

三个划分按 Bemis-Murcko 骨架哈希分组，比例阈值 80/10/10，seed=42；实际比例
受分组大小影响。规范化 SMILES 和骨架的跨划分重叠均为 0。原文件保持不变，
所有处理产物位于 Git 忽略的 dataset；可复现配置和哈希见 `ocelot_split_manifest.json`。
这只是当前表示的兼容子集，不能把成绩外推为完整 OCELOT 的成绩。

## 任务顺序

1. **MIM-Flow 结构适配与小规模 smoke。** 先用 QM9s 检查 atoms、构象列表、
   原子数、氢、电荷、分子图及坐标 token 往返。当前基线词表能覆盖其元素，
   原子数范围为 3–29，但 supplied SMILES 中仍有不可解析和被 RDKit 解释为自由基
   的记录；须核实是否来自上游氢/价态预处理，不可直接当作真实开壳层训练标签。
2. **OCELOT 性质预测与多目标测试。** 先验证 HOMO/LUMO、再检查派生 gap，
   固定上述划分，所有 scaler 只拟合训练集。CSV 只有 SMILES，没有原始 DFT
   几何；若用 ETKDG 构象，明确标为重建输入，不能宣称性质对应这个精确构象，
   也不能据此验证坐标梯度的量化精度。S1/T1 定义单独处理。
3. **MIM-Flow 光谱条件生成。** 确认轴、单位、归一化及损失定义后，设计光谱条件
   编码器/评价指标；不能将向量第一个数当成单个目标性质，或把所有通道机械地
   塞入原有 HOMO/LUMO 结果表。
4. **大分子与局部生成。** mrtadf 子集包含 B、Si、Se 等当前词表外元素；局部
   生成集的原子数全部超过 40。待词表、空间/长度能力和骨架生成扩展验证后接入。
   按当前模型中心化 ±6.4 的坐标范围检查，de_novo 的 train 中有 11,827/15,158
   条超界，local_gen 的 train 中有 364/371 条超界；QM9s train 只有 17 条。
   因此 T22 的空间扩展和 T24 的元素支持是完整 MR-TADF 测试的前置条件。
   不裁原子、不静默补氢、不通过修改 SMILES 使校验“通过”。
5. **正式对照实验。** 先以固定小样本做读取、重载、分子身份和有限步验收，再
   扩大训练。记录尝试/筛选分母、原始划分、模型/数据哈希、耗时、失败类别和
   独立性质评估。MIM-Flow 与 OCELOT 的数值结果分开报告。

## 防泄漏及剩余检查

- MIM-Flow 原始 valid 集先作为预留评估集；从 train 内另建固定 dev 划分。
  原始 train 中与 valid 重叠的规范化 SMILES 整组移出训练，并保留排除索引；
  不能仅删掉一条记录或在调参后继续把同一 valid 集称为独立测试。
- 还需检查原子映射、重复构象、光谱/几何的一一对应，以及 MIM-Flow 与 OCELOT
  的跨库结构重叠；当前给出的重叠数只覆盖各 MIM 子集的原始 train/valid。
- MR-TADF 中大量 supplied SMILES 被解析为自由基且许多记录不含显式 H，属于
  当前表示的适配问题，不能简单推断原始研究中的分子都是自由基或坏数据。
- 精确发布 URL、数据许可、发布方校验和及光谱语义保留为未完成项。最初清单只完成数据检查；随后已完成 QM9s 适配和 OCELOT 训练，见下面的更新。


## 实际执行更新

`adapt_qm9s.py --backend uni3dar_octree` 已完成 2048/256/256 个结构的适配。
光谱仍为独立 NPZ，不解释为 HOMO/LUMO。生成器的双性质条件来自 OCELOT 训练
Oracle 的伪标签，输出 SDF 明确记录该来源。

OCELOT 指纹模型在上述完整兼容训练划分训练；3D adapter pilot 使用固定哈希
选出的 512/64/64 条数据，失败构象保留记录。实际配置、指标和局限见
[实验报告](../docs/qm9s-ocelot-experiments-2026-11-11.md)。上述40原子及±6.4Å覆盖率
是旧 dense-grid 的审计数据，不能作为新八叉树的空间合同；MR-TADF 依然受预训练
词表和数据域限制，尚未运行完整测试。
