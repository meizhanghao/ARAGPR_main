# ARAGPR 代码使用说明

[返回英文主页](../README.md)

对应论文：**Auditing Residual Attribution with Geometric Prototype References**。

本目录是当前研究代码的独立发布副本。原研究工程未修改。为兼容已有模型，
保留部分旧文件名、类名和 `SDR-CBM` 标识；项目对外名称为 ARAGPR。

## 包含内容

- 五种轮廓描述子、训练区域原型和形状响应的构造。
- 冻结图像特征上的 decoder、残差分类头和逐坐标归因。
- 普通仿射残差、随机投影、真实/打乱配对、常数项和固定头 Teacher 替换对照。
- 贡献方向、纠错保留、Top-k 重叠、有效 Teacher 分组和病例分区诊断。
- 普通 CLIP 线性分类器上的分支验证、解释素材导出与命名辅助工具。
- 测试、依赖、来源说明和一个不含真实数据的运行示例。

原始数据集、独立图像/区域库、mask 缓存、embedding、权重和历史实验输出不在副本中。
文档仅收录论文已经使用的合成图示；其余数据文件仍由 `.gitignore` 排除。
第三方 SAM2 框架从官方仓库单独安装。

## 结合论文图理解方法

英文主页已嵌入论文的框架图、聚类原型展示和完整贡献审计图。
更完整的[图文说明](VISUAL_GUIDE.md)还包括残差接口、纠正案例、误改案例及验证曲线，
并标明每张图对应的代码入口。

- 框架图：区分训练时的形状监督与推理时无需 mask 的 decoded 分支。
- 聚类结果图：对照原图、局部区域和轮廓，理解坐标的训练参考内容。
- 贡献审计图：将全部 128 项、Top-5 参考及其正负贡献对应起来；符号始终针对明确的类别对比。
- 纠正与误改图：同时查看有益和有害的预测变化，结合 Other、Bias 检查完整分解。

聚类成员是用于展示的选例，不是全簇纯度统计；参考图不定位测试图像证据。
自动名称仅辅助阅读，不参与训练、预测或贡献计算。

## 开始使用

```bash
python -m pip install -r requirements.txt
python -m aragpr doctor
python -m unittest discover -s tests -v
python -m aragpr demo --output outputs/synthetic_demo
```

`synthetic_demo` 只验证归因公式和 CSV 输出，不是论文实验数据。
CLIP、SAM2、旧基线所需的额外依赖在 `requirements-optional.txt` 中。

完整输入结构见 [DATA.md](DATA.md)。准备好数据后，可运行：

```bash
python train.py --dataset derm7pt --use_shape --shape_input geo --K_shape 128 --shape_mode decoded --freeze_semantic_from semantic_checkpoints/derm7pt_semantic_baseline.pt
```

`geo` 是代码中 geo11 的参数名。`oracle` 是 Direct teacher 的历史参数名，
需要测试 mask，不能作为 mask-free 推理结果。`decoded_rel` 是可选的相关性加权变体。

需要把代码与数据分开存放时，使用 `train.py --project-root 数据目录 ...`。
配对对照入口也支持 `--project-root`。搬迁旧缓存时可设置环境变量
`ARAGPR_DATA_ROOT`，数据文件仍会执行已有 SHA256 校验。

## 实验对应关系

入口对应表在 [ENTRY_POINTS.md](ENTRY_POINTS.md)，
训练和统计口径在 [PROTOCOLS.md](PROTOCOLS.md)。
主表最后 15 轮均值、单检查点诊断、三个分支种子的均值和标准差必须区分。
代码副本不附带服务器检查点序列，不能把本地初跑结果直接当作论文表格的逐项复现。

命名工具提供候选词的 CLIP 排序；VLM 决选应另行记录实际模型、prompt、
输入图像与输出。名称仅辅助阅读，不进入分类或贡献计算。

## 上传 GitHub

上传本目录中的源码和文档，保留 `.gitignore`、`LICENSE`、
`licenses/DCBM-MIT.txt` 和 `THIRD_PARTY_NOTICES.md`。
无需把原工程、论文稿件、临时目录或数据集一起上传。
远端仓库为 [meizhanghao/ARAGPR_main](https://github.com/meizhanghao/ARAGPR_main)。

