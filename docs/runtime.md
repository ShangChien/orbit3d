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
实际 GPU 量化使用独立环境，以避免与主模型的二进制依赖混用。GPU4PySCF 1.9.1、
GPU Libxc 0.9.0、CuPy 13.4.1、cuTENSOR 2.2.0 和 PySCF 2.12.1 的组合已有
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

## 性质模型合同

主实验使用 DPA/HyFF 描述符加 bootstrap 属性头，以及独立指纹模型。训练的目标
顺序、单位、scaler、构象准备方法和适用域随模型保存。普通预测、可微预测、
导出复查必须对应同一个模型。多头分歧是诊断量，不能自动当成校准置信区间。

可选 Uni-Mol/Uni-QSAR 独立桥接模块保留，尚未接入当前双后端 CLI 的评估器工厂，也不属于当前主实验的 GPU 精度结论：

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
