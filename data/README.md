# 数据与目标定义

原始/处理数据放在根目录 `dataset/`，不进入 Git。本目录只保存来源、校验和、
划分、目标窗口和必要的统计。当前研究使用 MIM-Flow 与 OCELOT；DB_chromophore
和其他 figshare 表未进入本轮实验，原文件保留，待后续决定是否使用。

## 来源

- **OCELOT chromophore v1**：官方 ZIP 的 CSV 字节校验已通过；源 CSV SHA-256
  为 `9ccde14ace63e63f288655c47ab32d14a44ab478319e48ccc0a2c27ce8bd1985`。
  [下载与条款记录](ocelot_download.json)；使用受 OCELOT 非商业研究及署名条款约束，
  不随源码分发原始数据。
- **MIM-Flow Forward Data and Model**：官方 [Zenodo 21887607](https://zenodo.org/records/21887607)，
  CC BY 4.0。3,248,316,986 字节原包匹配发布方 MD5，gzip CRC 通过；SHA-256 为
  `ded9b275805df1976b3c8e666495f723a4d34c8440890b9f535ee52fc22ca43d`。
  [发布方记录](mim_flow_publisher_record.json)、[包清单](mim_flow_archive.json)、
  [LMDB 字段清单](mim_flow_lmdb_inventory.json)。Inverse Design 包是另一份产物，
  不能用它替代包含 QM9s 的 Forward 包。
- `provenance.json` 是本地数据文件清单；未使用数据的缺失来源/许可字段不代表
  优先数据的已核验来源也缺失。获取/校验入口为 `scripts/manage_data.py`。

## 实际数据与坐标

| 数据 | 用途与规模 | 坐标和标签 |
| --- | --- | --- |
| QM9s | 2048 train / 256 dev / 256 test；test 来自官方 valid | 直接读取源坐标；240 通道光谱单独保存 |
| MR-TADF | 512 个经身份/闭壳层检查的含硼结构适配 | 直接读取源坐标；450 通道光谱不冒充轨道性质 |
| OCELOT | 骨架划分 20356 / 2282 / 2322 | 原 CSV 为 SMILES＋性质，没有原始量子坐标 |
| DPA/HyFF 性质比较 | 同一批 8192 train / 512 valid / 512 test，另512校准 | 已完成实验统一用 ETKDGv3＋UFF，早期 MMFF 仅作初始集成 |
| OCELOT 生成适配 | 8188 个可完整编码训练构象，511 个验证结构 | 沿用上述固定结构，不重新分配测试数据 |

OCELOT 按种子42与 Bemis–Murcko 骨架哈希划分，跨集合 SMILES/骨架重叠均为零。
保留冲突标签排除和尺寸延后记录；最多80个显式原子。[完整划分](ocelot_wide80_split_manifest.json)

元素为 H/C/N/O/F/B/Si/P/S/Cl/Br/I/Se/As/Te。最终512测试分子含氢总原子数
13–79，均值34.35、中位数32；重原子数9–58、中位数21。总原子数21–30和
31–40分别占32.8%和38.1%。稀有元素测试分子数为B13、Si7、Se1、As1、Te4；
这不支持每种元素都有可靠精度的结论。[训练/验证/测试完整组成](ocelot_matched_atom_inventory.json)

上述构象处理属于已完成的 OCELOT 实验；后续数据采用方向见[方案调整](../docs/design_changes.md)。

## 候选源三维数据

以下为候选数据的发布规模与计算协议，不表示已经下载、适配或训练。采用方向及
实施状态由[方案调整 D01](../docs/design_changes.md#活动调整)维护，执行协议见
[实验 E0](../docs/experiments.md#e0-源三维数据与目标冻结)。

| 数据 | 发布规模与分子大小 | 坐标与性质计算等级 |
| --- | --- | --- |
| [OE62](https://www.nature.com/articles/s41597-020-0385-y) | 61489 个分子，最多174个总原子/92个重原子，16元素 | 提供气相 PBE/PBE0 轨道标签，源几何为 PBE＋vdW；FHI-aims Tier2/tight，不能以 PySCF/def2-SVP 冒称同级复现 |
| [原始 QM9](https://www.nature.com/articles/sdata201422) | 133885 个分子，最多9个重原子/29个总原子 | B3LYP/6-31G(2df,p) 几何与轨道标签，原轨道单位为 Ha |
| [QMugs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174255/) | 665911 个分子、1992984 个构象，3–100重原子、平均约30.6 | GFN2-xTB 几何＋ωB97X-D/def2-SVP 单点性质，不称为 DFT 优化几何 |
| [GEOM](https://www.nature.com/articles/s41597-022-01288-4) | 约45万分子、3700万构象；药物主子集平均44.4总原子、24.9重原子 | CREST/xTB，不能默认含本项目四目标的同级标签 |

发布规模不等于当前可用样本量，筛选范围、划分及目标冻结协议见
[实验设计](../docs/experiments.md#下一轮最小实验)。不同计算等级的轨道标签分别记录。

已经下载的 MIM-Flow 仍可用于源三维结构与光谱研究。远程 LMDB 扫描所得规模如下；
原子分布以训练部分为准，未按图去重或做适用性筛选：

| 数据 | train / valid 记录 | 所提供坐标的原子数范围 | 中位数；Q25–Q75 |
| --- | ---: | --- | --- |
| QM9s | 123323 / 6490 | 3–29 | 18；16–20 |
| MR-TADF de novo | 15158 / 1371 | 14–86 | 38；34–44 |
| MR-TADF local | 371 / 41 | 44–87 | 66；60.5–72 |

QM9s 训练集重原子数1–9、均值8.80；de novo 有12347条训练记录不含显式氢坐标，
local 训练集全部不含显式氢，因此不能把三行都称作完整含氢原子数。
这些数据的当前目标是240/450通道光谱，不能直接承担新的 HOMO/LUMO 监督。
[源文件与哈希](mim_flow_archive.json)、[记录数与字段](mim_flow_lmdb_inventory.json)

## 目标

`homo`、`lumo`、`s0s1`、`s0t1` 单位均为 eV。后两项为源 S0 参考几何上的
垂直激发能，不能当作各激发态优化后的绝热能量。`hl=lumo-homo` 和
`s0s1-s0t1` 是逐分子派生量，不作为任意独立组合的额外标签。

[轨道窗口](wide_target_windows.json)、[训练标签选窗协议](wide_target_windows_protocol.json)、
[光学窗口](optical_target_windows.json) 定义固定生成任务。不同数据集的相似字段名
不保证相同理论级别。MIM-Flow 的谱向量与 OCELOT 轨道/激发态标签不拼接成同一真值。

[旧窗口训练覆盖](ocelot_target_coverage.json) 给出每个边界/中心的分位数和逐分子交集。
完整训练集的 Original/Central/lower_gap/Optical/Four-target 覆盖分别为
17.72%/34.79%/17.46%/29.52%/0.57%。新任务的目标选择方法及冻结状态见
[实验 E0](../docs/experiments.md#e0-源三维数据与目标冻结)。

## 准备命令

```bash
python scripts/prepare_ocelot.py \
  --source dataset/raw/ocelot_chromophore_v1/ocelot_chromophore_v1.csv \
  --output dataset/processed/ocelot_wide80_seed42 \
  --max-atoms 80 --seed 42 --elements H C N O F B Si Se As Te S P Cl Br I
python scripts/adapt_qm9s.py --source /data/mim-flow/qm9s --output dataset/processed/qm9s_octree
```

模型训练和实验运行使用已冻结的具体配置。`configs/` 是可编辑模板，不能用其
默认值替代原实验快照。性质模型训练、验证和测试的 SourceID 保存在实验结果中。
