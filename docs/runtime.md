# 运行与模型接口

## 环境

主线使用 Python 3.11/3.12、Torch 2.7.1。`uv.lock` 固定依赖；主线 GPU
执行已在 Linux/A800 上验证。macOS 只用于 CPU 测试、打包与文稿。

```bash
uv sync --frozen --extra octree --extra deepmd
uv run --frozen --extra octree --extra deepmd orbit3d --config configs/qm9s_ocelot_octree.json
```

DeepMD 3.2 的发布 C++ wheel 与所用 Torch 版本不兼容，项目使用其官方 Python
PyTorch 后端，源码构建设置 `DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=0`；
这不禁用 CUDA 推理。设置已写入 pyproject。慢速网络可使用镜像安装同一锁定
版本和哈希，不替换模型或依赖版本。GPU 确定性运行需在 CUDA 初始化之前设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，仍应使用声明的数值容差比较输出。

HyFF 用 `--extra hyff`；其源码、注册表和许可证随 wheel 打包，真实权重单独获取。
量化 CPU 依赖为 `--extra quantum`；Linux CUDA12 依赖为 `--extra quantum-gpu`。
固定的pyscf-dispersion 1.3.0仅发布x86_64轮子，安装声明按架构限定；Apple Silicon保留CPU单点能力，既定D3BJ几何优化使用已验证的远程x86_64环境。
实际 GPU 量化使用独立环境，以避免与主模型的二进制依赖混用。GPU4PySCF 1.9.1、
GPU Libxc 0.9.0、CuPy 13.4.1、cuTENSOR 2.2.0、PySCF 2.12.1及pyscf-dispersion 1.3.0的组合已有
实测；不把它写成所有可选依赖合装或 Docker 镜像的验收结论。

远程为平台容器，Docker daemon/cache 由平台管理。模型包和必要归档保存在
EFS，私有路径见 Git 忽略的 `local_dev/README.md`。Dockerfile 仅为可选配方。

## 输入、模式和输出

配置必须明确声明 `generator_config.backend`：`uni3dar_octree` 或
`scaffold_graph_growth`。不再提供 dense-grid 训练/生成入口。

- `train`：正数 SFT 步数。使用 SDF 实测标签且 GRPO 为零时，不加载 Oracle/力场。
- `align`：加载已有适配检查点，更新 conditioner；开启 replay 时需提供训练 SDF。
- `predict`：加载适配检查点、性质模型和力场，无需训练数据或上游预训练文件。

检查点保存元素支持集、空间分辨率/深度、条件版本、目标名称/单位、scaler 和权重。
通过 `migrate_octree_*.py` 显式扩展深度、元素、条件或目标；迁移后按约定适配训练。
保持参数形状不等于模型已学会新语义。当前已验证生成任务为中性闭壳层有机分子。

生成输出包括 `candidates.csv`、`generated.sdf`、`qualified.sdf`、
`relaxation_traces.json`、`summary.json` 和代码/输入清单。`generated.sdf` 为保留
候选，`qualified.sdf` 还需通过性质区间、适用域和原始力阈值；失败/筛除候选仍在
全部尝试分母中。训练、验证、测试及 scaler 拟合范围不能混用。

骨架工作流的输入 SDF 必须带三维坐标和 `OpenAtomIndices` JSON 数组。
开放位点通过显式氢替换连接片段，保留原图、原子映射和固定世界坐标。
低层图 API 支持键型、新开放位点和环闭合；这不是已训练的 octree 骨架补全。
输入实测性质被保留为源属性，不作为新生成分子的真值标签。

## OE62点值与GRPO对照

先完成 `prepare_oe62.py` 源坐标适配、`train_source_oracle.py` 性质头训练，以及 `initialize_point_octree.py` / `train_octree_structural.py` 先验适配。以下路径为已准备好的本机或服务器目录；私有服务器路径不写进配置模板。

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
ORBIT_DATA=/path/to/oe62
ORBIT_PRIOR=/path/to/structural_prior/octree_structural.pt
ORBIT_DPA=/path/to/oracle_dpa
ORBIT_FP=/path/to/oracle_fingerprint
ORBIT_FORCEFIELD=/path/to/DPA4-Mini-OMol25-100M-v20260820.pt
ORBIT_RUN=/path/to/point_homo_seed42
ORBIT_EVAL=/path/to/eval_homo_seed42

for stage in sft arms; do
  python scripts/run_point_comparison.py --stage "$stage" \
    --data "$ORBIT_DATA" --prior "$ORBIT_PRIOR" --dpa "$ORBIT_DPA" \
    --forcefield "$ORBIT_FORCEFIELD" --target homo --seed 42 \
    --device cuda:0 --output "$ORBIT_RUN"
done

python scripts/evaluate_point_campaign.py --campaign "$ORBIT_RUN" \
  --output "$ORBIT_EVAL" --device cuda:0 --batch-size 64 --candidates 128 \
  --frozen "$ORBIT_RUN/sft/weights/octree_conditional.pt" \
  --continued "$ORBIT_RUN/continue/weights/octree_conditional.pt" \
  --independent "$ORBIT_FP/oracle" \
  --arms frozen continue400 grpo200 --time-control

python scripts/summarize_point_comparison.py --campaign "$ORBIT_EVAL" \
  --alignment-campaign "$ORBIT_RUN" --sft-campaign "$ORBIT_RUN" \
  --manifest "$ORBIT_DATA/manifest.json" --target homo \
  --output "$ORBIT_EVAL/comparison.json"
```

`run_point_comparison.py`仅负责训练，阶段明确为`sft`、`arms`或单独`grpo`；已删除重复的逐池冷启动评价路径。`evaluate_point_campaign.py`统一复用只读模型，`--time-control`按本次GRPO实耗时间训练并评价额外SFT臂。原始池不排序、不弛豫；输出中仍保留失败提案。所有输出目录均为新目录，历史结果绑定当时保存的源码快照。

LUMO使用独立输出目录并改`--target lumo`。正式seed43/44需从共同上游权重重新初始化和适配先验，再分别训练两个性质，不能只换采样seed。两张物理卡各绑定一个进程，内部统一`cuda:0`。本次已验证batch8至64在HOMO/LUMO各64个固定候选上的轨迹一致性；变更其他预算/批大小仍需另验。

GRPO的`octree_resume_*.pt`保存当前策略、原始参考策略标识、AdamW状态、随机流和累计更新进度。对齐配置用`mount_paths.alignment_resume_path`恢复，而非把中途策略当成新的参考模型；已完成的100+100组恢复与400次优化器更新有实际记录。

## OE62 PySCF量化复核

使用已安装的独立量化环境，协议为PBE0/def2-TZVP及重元素对应的标量ECP（I/Te为28芯电子，逐结果记录）、密度拟合def2-universal-jkfit、grid level 3、SCF容差1e-8。`--optimize`增加PBE-D3BJ/def2-TZVP几何优化和PBE0末端单点；优化后的图与坐标仍走共享化学验收。计算不使用ω调参或旧OCELOT的LC-wPBE设置。

```bash
export PYTHONPATH=src
ORBIT_FORMAL=/path/to/formal_campaign_root
ORBIT_QC=/path/to/new_quantum_cohort
ORBIT_QC_RESULTS=/path/to/new_quantum_results
ORBIT_QC_PYTHON=/path/to/quantum/python_or_runtime_wrapper

python scripts/select_point_quantum_cohort.py --root "$ORBIT_FORMAL" --output "$ORBIT_QC"
python scripts/run_oe62_quantum.py --input "$ORBIT_QC/cohort.sdf" \
  --output "$ORBIT_QC_RESULTS" --python "$ORBIT_QC_PYTHON" --gpus 0 1 --optimize
python scripts/summarize_point_quantum.py --manifest "$ORBIT_QC/manifest.json" \
  --results "$ORBIT_QC_RESULTS" --output "$ORBIT_QC/comparison.json"
```

默认候选选择固定正式GRPO200和等时SFT、三个中央目标及匹配训练/推理seed；其他已冻结协议通过`--arms`、`--campaign-prefix`和`--assignments quantile:training_seed:inference_seed ...`指定。量化结果不会改变队列。GPU编号由启动参数指定，一张卡一个进程。`--engine cpu --cpu-workers 4`可在训练期间异步运行CPU队列，每个worker使用4个PySCF线程；单点与几何优化通过`--optimize`区分。队列将`TMPDIR`设置为输出目录下的`scratch/`，PySCF临时积分文件随实验落盘，避免占满容器根盘。`completed.json`表示队列退出，逐结构的SCF收敛、几何收敛和保图结果需要分别查看；失败保留分母。已有源参照直接复用，元素覆盖扩展由`select_point_quantum_references.py`固定源开发结构。

## 性质模型合同

主实验使用 DPA/HyFF 描述符加 bootstrap 属性头，以及独立指纹模型。训练的目标
顺序、单位、scaler、构象准备方法和适用域随模型保存。普通预测、可微预测、
导出复查必须对应同一个模型。多头分歧是诊断量，不能自动当成校准置信区间。

OE62独立评价已验证官方Uni-Mol v1与Uni-Mol2；按开发集选中的Uni-Mol2 1.1B使用 `InternalUniMol2Oracle.load_from_dir(model_dir, device)`，目录须含 `model.pt`、`oracle_meta.json`。`predict_3d_mols(mols, batch_size=...)` 直接读取输入源坐标，沿官方路径移除显式氢，不嵌入、不裁剪；本轮共同支持集为14元素、源分子≤80原子。返回的零spread只表示单模型接口，不代表零误差，区间由独立校准残差给出。官方v2内部detach距离，不提供坐标性质梯度。

训练入口为 `train_unimol2_source.py`；`select_unimol2_development.py` 只读开发集，`profile_dual_oracle.py --selection ...` 在选模冻结后运行源测试，`reevaluate_dual_oracle.py --profile profile_unimol2` 评价原A已冻结的候选与选择。`run_unimol2_evaluation.py` 串联这三步及优化后量化几何复评。该B通过独立实验入口使用，没有加入生成CLI评估器工厂或奖励函数。精度、适用范围、权重哈希及耗时见[最终profile](experiments/oe62_unimol2_evaluation_20261113.json)。

可选Uni-QSAR外部桥接仍与该原生Uni-Mol2路径区分：

- Uni-Mol 必须加载匹配的真实预训练骨干；fold scaler 只在对应训练折拟合。
- Uni-QSAR 按训练配置加载所有成员/fold、scaler 和 stacking；归一化空间中的
  聚合后再逆变换，保持原模型的点预测定义。物理单位下的 fold 分散度单独计算。
- 非线性 scaler 不能用一个 scale 系数变换标准差。非可微 stacking 使用原黑盒
  预测进行验收，不拿辅助头或均值替代原模型。
- 外部 imputer、LightGBM 等必须使用与训练产物相符的真实运行时；当前测试中的
  序列化小模型验证不代表完整外部训练目录或 GPU 后端已经验收。

## 可迁移包与重放

```bash
python scripts/export_inference_bundle.py \
  --run /path/to/completed_run --output /persistent/inference_bundle \
  --wheel dist/orbit3d-1.0.0-py3-none-any.whl \
  --requirements /path/to/requirements.lock.txt \
  --asset-root /path/to/original_workdir --archive /persistent/inference_bundle.tar
```

包包含适配生成器、性质头/scaler、力场、适用域、wheel、锁定依赖说明和整批
参考输出，活动路径改为相对路径。安装 wheel 后，在包目录执行 `python
verify_bundle.py`。这是可迁移模型/代码包，不是操作系统镜像。

已经验证的四目标与固定 seed-42/20000 步双目标包，分别重放全部 64 个候选和
6/9 个导出结构；每包 23 个文件哈希通过。历史包始终绑定其自带 wheel；不要用
新源码覆盖后继续沿用旧验证结论。证据见 [四目标包](experiments/relocated_artifact_verification_20261112.json)
和 [双目标包](experiments/orbital20k_artifact_verification_20261112.json)。

### DeePMD 多卡初始化

DPA 的设备须在 DeePMD 首次导入前确定：库内部分常量张量不属于 PyTorch 注册 buffer，事后 `model.to(device)` 不会移动它们。`DPAPotentialGuidance` 依据显式 device 设置当前 GPU 和 DeePMD 的 `LOCAL_RANK` 后再导入；同一 worker 绑定一个设备，两张卡使用两个进程。也可在进程启动时用 `CUDA_VISIBLE_DEVICES` 选择物理卡并统一使用逻辑 `cuda:0`。

## 图特征模型版本

新训练的指纹模型及Morgan域使用 `rdkit_normalize_canonical_v1`：在图副本上进行官能团标准化和规范异构SMILES转换，原图与三维坐标不被改写。转换必须保持总电荷；无法构建特征的候选保留为评价失败。指纹metadata记录v2及特征契约，域文件记录schema版本与同一契约。旧v1文件按原预处理重载；更换特征需要重新训练指纹参数、重建及校准域文件，不能只替换预处理函数。

## 已冻结Uni-Mol2的主线复评

`evaluate_formal_unimol2.py`接受同一训练seed的HOMO/LUMO正式campaign路径、模型目录和开发集选型清单；先核对权重及scaler元数据哈希，再评价保存的原始轨迹。示例：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/evaluate_formal_unimol2.py \
  --sources /path/to/formal_eval_homo_seed42 /path/to/formal_eval_lumo_seed42 \
  --oracle /path/to/unimol2_11b/oracle --selection /path/to/selection.json \
  --manifest /path/to/oe62/manifest.json --output /path/to/new_reevaluation
```

`summarize_independent_3d_campaign.py`和`summarize_independent_3d_repeats.py`按报告中的评价器身份汇总；不能把Uni-Mol2结果标为HyFF。正式与初试预算分别统计。`bridge_quantum_unimol2.py`复用既有量化队列和对应输入SDF，在完全相同坐标重评，不新增量化样本。
