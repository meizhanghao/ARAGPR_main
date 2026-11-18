# ARAGPR 代码使用说明

[返回英文主页](../README.md)

对应论文：**Auditing Residual Attribution with Geometric Prototype References**。

ARAGPR 从训练区域的轮廓构造几何原型，通过冻结图像嵌入预测原型响应，
并将残差分类分数分解为逐坐标的有符号贡献。每个坐标保留对应的训练区域
与轮廓作为视觉参考。DCBM 是主要实验平台，代码也提供普通冻结 CLIP
线性分类器上的分支入口。

## 包含内容

- 五种轮廓描述子、训练区域原型和形状响应的构造。
- 冻结图像特征上的 decoder、残差分类头和逐坐标归因。
- 普通仿射残差、随机投影、真实/打乱配对、常数项和固定头 Teacher 替换对照。
- 贡献方向、纠错保留、Top-k 重叠、有效 Teacher 分组和病例分区诊断。
- 普通 CLIP 线性分类器上的分支验证、解释素材导出与命名辅助工具。
- 测试、依赖、来源说明和一个不含真实数据的运行示例。

训练所需的数据集、区域掩码、图像嵌入和基准检查点需单独准备。
仓库提供五个数据集的类别词表，以及用于理解方法和输出的论文图示。
SAM2 及其模型权重从官方项目获取。

## 结合论文图理解方法

[英文主页](../README.md)展示框架图、聚类原型和完整贡献审计图。
[图文说明](VISUAL_GUIDE.md)还包括残差接口、纠正案例、误改案例及验证曲线，
并提供每张图对应的代码入口。

- 框架图：区分训练时的形状监督与推理时无需 mask 的 decoded 分支。
- 聚类结果图：对照原图、局部区域和轮廓，理解坐标的训练参考内容。
- 贡献审计图：将全部 128 项、Top-5 参考及其正负贡献对应起来；符号始终针对明确的类别对比。
- 纠正与误改图：同时查看有益和有害的预测变化，结合 Other、Bias 检查完整分解。

聚类成员是用于展示的选例，不是全簇纯度统计；参考图不定位测试图像证据。
自动名称仅辅助阅读，不参与训练、预测或贡献计算。

## 安装与快速开始

建议使用 Python 3.10 或 3.11，并根据 CPU/CUDA 环境安装 PyTorch。
在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m aragpr doctor
python -X utf8 -m unittest discover -s tests -v
python -m aragpr demo --output outputs/synthetic_demo
```

`synthetic_demo` 使用合成输入，导出全部 128 项坐标贡献并验证类别间隔分解，
无需下载数据集或模型检查点。CLIP、SAM2 和基线所需的额外依赖在
`requirements-optional.txt` 中。

## 数据准备与训练

完整输入结构见 [DATA.md](DATA.md)。准备好数据后，可运行：

```bash
python train.py --dataset derm7pt --use_shape --shape_input geo --K_shape 128 --shape_mode decoded --freeze_semantic_from semantic_checkpoints/derm7pt_semantic_baseline.pt
```

`geo` 对应 11 维的 geo11 描述子。三种分支模式分别为：

- `decoded`：从图像嵌入预测形状响应，推理时无需测试掩码。
- `oracle`：Direct teacher 的命令行名称，使用掩码生成的目标响应进行诊断。
- `decoded_rel`：使用验证集相关性进行加权的 decoded 变体。

代码和数据可以分开存放。通过 `train.py --project-root 数据目录 ...`
指定数据根目录；配对对照入口也支持该参数。使用已缓存的实验时，可通过
`ARAGPR_DATA_ROOT` 指定输入数据位置，加载器会校验特征文件的 SHA256、
样本顺序和类别标签。

## 对照实验与诊断

入口对应表在 [ENTRY_POINTS.md](ENTRY_POINTS.md)，
训练和统计口径在 [PROTOCOLS.md](PROTOCOLS.md)。
主要入口包括：

```bash
python -m experiments.controls.run_residual_controls --help
python -m experiments.controls.run_decoder_baselines --help
python -m experiments.diagnostics.run_posthoc_shape_diagnostics --help
python -m experiments.diagnostics.run_teacher_alignment_audit --help
python -m tools.plotting.plot_contributions --help
```

运行时指定输入缓存、模型路径和输出目录。残差对照、解码器基线、固定头
Teacher 替换和贡献对应性分析使用各自的输入与统计规则。

## 结果与统计口径

论文主要分类结果为最后 15 轮测试准确率的算术平均；固定模型诊断使用
验证集选定的检查点；配对实验的标准差来自不同分支种子的重复拟合。
复现前者需要对应的检查点序列，后两者需要保存的模型和输入缓存。

归因针对指定的两个类别计算：正值支持目标类别，负值支持参考类别。
完整类别间隔由基准间隔、全部坐标贡献和偏置项共同构成。Top-k 按贡献
绝对值排序，贡献大小同时取决于响应和分类头权重。

训练参考用于查看坐标关联的区域与轮廓，不等同于测试图像中的证据定位。
医学实验的保留分区与独立病例分区是不同设置，具体条件见
[实验协议](PROTOCOLS.md)与[数据说明](DATA.md)。

## 原型辅助命名

命名流程使用 CLIP 对候选词排序，再由 GPT-4o 结合区域原图、RGB 局部图
与轮廓图进行选择、精简或弃权。`tools/naming/` 提供 CLIP 排序工具，
[命名说明](prototype_naming/README_zh-CN.md)提供 GPT-4o 提示词和运行记录模板。
每次运行保存模型信息、候选词表、排序、提示词与输出。

名称用于辅助理解训练参考内容，不参与模型训练、预测或贡献计算。

## 目录与运行方式

主训练入口为 `train.py`；核心实现位于 `aragpr/training.py`。
`experiments/` 下分别存放基线、拟合对照和固定模型诊断；预处理在
`scripts/preprocessing/`；素材筛选、完整性核验、命名和绘图在 `tools/` 对应子目录。
辅助入口从仓库根目录以 `python -m 包.模块` 运行，例如：

```bash
python -m experiments.controls.run_decoder_baselines --help
python -m experiments.diagnostics.run_teacher_alignment_audit --help
python -m tools.plotting.plot_contributions --help
```

完整命令及检查点兼容性说明见 [ENTRY_POINTS.md](ENTRY_POINTS.md)。

## 引用与许可

论文标题和作者信息见 [CITATION.cff](../CITATION.cff)。DCBM 相关实现改编自
[官方代码](https://github.com/KathPra/DCBM)，引用信息见[英文主页](../README.md#citation-and-acknowledgements)。
代码许可及第三方来源分别见 [LICENSE](../LICENSE) 和
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。数据集和第三方模型权重
遵循各自提供方的使用条款。

联系邮箱：zqmeng@gxu.edu.cn。
