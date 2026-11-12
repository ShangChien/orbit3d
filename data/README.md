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

当前构象处理可继续支持受控比较；不把全面重算 DFT 当作开发前置条件。
下一轮比较同一初始构象的 UFF/DPA/HyFF 优化，并分别重训性质头，见
[实验方案](../docs/experiments.md#下一轮最小实验)。

## 目标

`homo`、`lumo`、`s0s1`、`s0t1` 单位均为 eV。后两项为源 S0 参考几何上的
垂直激发能，不能当作各激发态优化后的绝热能量。`hl=lumo-homo` 和
`s0s1-s0t1` 是逐分子派生量，不作为任意独立组合的额外标签。

[轨道窗口](wide_target_windows.json)、[训练标签选窗协议](wide_target_windows_protocol.json)、
[光学窗口](optical_target_windows.json) 定义固定生成任务。不同数据集的相似字段名
不保证相同理论级别。MIM-Flow 的谱向量与 OCELOT 轨道/激发态标签不拼接成同一真值。

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
