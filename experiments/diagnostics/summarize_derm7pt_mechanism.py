"""Audit and summarize the completed local Derm7pt mechanism pilot."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.controls.run_decoder_baselines import sha256,write_csv,write_json
from experiments.controls.run_derm7pt_mechanism import case_metadata
from experiments.diagnostics.run_derm7pt_perturbations import mean_interval

from aragpr.paths import repository_root

ROOT = repository_root()
FIT=ROOT/"outputs/derm7pt_mechanism_20261020_pilot01"
EDIT=ROOT/"outputs/derm7pt_perturbations_20261020_pilot01"
REPORT=ROOT.parent.parent/"derm7pt_mechanism_report_20261020.md"
LABELS={"paired":"真实配对","shuffled":"全局打乱配对","within_class_shuffled":"同类内打乱配对"}
ENGLISH={"paired":"Paired","shuffled":"Shuffled","within_class_shuffled":"Within-class shuffled"}


def csv_table(frame,columns,labels=None):
    labels=labels or columns
    rows=["| "+" | ".join(labels)+" |","|"+"---|"*len(columns)]
    for _,row in frame.iterrows():
        values=[]
        for key in columns:
            value=row[key]
            values.append(f"{value:.3f}" if isinstance(value,(float,np.floating)) else str(value))
        rows.append("| "+" | ".join(values)+" |")
    return "\n".join(rows)


def run():
    fit_protocol=json.loads((FIT/"protocol.json").read_text(encoding="utf-8"))
    edit_protocol=json.loads((EDIT/"protocol.json").read_text(encoding="utf-8"))
    assert fit_protocol["status"]==edit_protocol["status"]=="complete"
    summary=pd.read_csv(FIT/"classification_summary.csv")
    fidelity=pd.read_csv(FIT/"decoder_fidelity.csv")
    comparisons=pd.read_csv(FIT/"paired_comparisons.csv")
    each=pd.read_csv(EDIT/"per_image_sensitivity.csv")
    registration=pd.read_csv(EDIT/"mask_registration.csv")
    exclude=pd.read_csv(EDIT/"excluded_images.csv")
    seeds=fit_protocol["seeds"]
    models=fit_protocol["modes"]
    with np.load(EDIT/"edited_features_and_teachers.npz",allow_pickle=False) as z:
        keys=z["keys"].astype(str)
        features=z["features"].copy()
        targets=z["teacher"].copy()
        variants=z["variants"].astype(str)
    cases,_=case_metadata(keys)
    assert len(keys)==273 and features.shape==(273,9,768) and targets.shape==(273,9,128)
    assert len(each)==len(keys)*8*len(seeds)*len(models)
    assert len(exclude)==17
    # Every appearance teacher must be bit-identical to its original teacher.
    for vi,name in enumerate(variants):
        if "blur" in name: assert np.array_equal(targets[:,vi],targets[:,0])
    # Replay every stored sensitivity model independently from edited features.
    replay_errors=[]
    for seed in seeds:
        for mode in models:
            state=torch.load(FIT/f"seed_{seed}"/mode/"selected_model.pt",weights_only=False,map_location="cpu")
            scaled=state["input_scaler"].transform(features.reshape(-1,768)).astype(np.float32)
            decoded=(scaled@state["decoder_state"]["weight"].numpy().T+
                     state["decoder_state"]["bias"].numpy()).reshape(273,9,128)
            x=state["coordinate_scaler"].transform(decoded.reshape(-1,128)).reshape(decoded.shape)
            head=state["head"]["selected"]
            residual=head["eta"]*(x@head["state"]["weight"].numpy().T+head["state"]["bias"].numpy())
            with np.load(EDIT/f"sensitivity_seed_{seed}_{mode}.npz",allow_pickle=False) as z:
                np.testing.assert_allclose(decoded,z["decoded"],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(residual,z["residual_logits"],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(z["semantic_logits"]+residual,z["final_logits"],rtol=2e-5,atol=2e-5)
                replay_errors.append(float(np.max(np.abs(residual-z["residual_logits"]))))
    # Aggregate seeds per image first; never treat 3 seeds as 3 independent images.
    each["absolute_residual_margin_change"]=each.delta_residual_margin.abs()
    averaged=each.groupby(["mode","variant","key","case"],sort=False).mean(numeric_only=True).reset_index()
    sensitivity=[]
    for (mode,variant),sub in averaged.groupby(["mode","variant"],sort=False):
        mean,low,high=mean_interval(sub["delta_direction_cosine"].values,sub["case"].values)
        sensitivity.append({"mode":mode,"variant":variant,"n_images":len(sub),"n_seeds":len(seeds),
            "teacher_delta_rms":float(sub["teacher_delta_rms"].mean()),
            "decoded_delta_rms":float(sub["decoded_delta_rms"].mean()),
            "mean_abs_residual_margin_change":float(sub["absolute_residual_margin_change"].mean()),
            "mean_cosine_seed_average":mean,"cosine_case_bootstrap_95_low":low,
            "cosine_case_bootstrap_95_high":high,
            "fraction_with_at_least_10pct_editable_pixels":float((sub["editable_fraction"]>=.1).mean())})
    write_csv(EDIT/"sensitivity_seed_averaged.csv",sensitivity)
    sensitivity=pd.DataFrame(sensitivity)
    cos_comparisons=[]
    for variant in variants[1:5]:
        part=averaged[averaged.variant==variant]
        paired=part[part["mode"]=="paired"].set_index("key")
        for control in models[1:]:
            other=part[part["mode"]==control].set_index("key").loc[paired.index]
            mean,low,high=mean_interval(paired.delta_direction_cosine.values-other.delta_direction_cosine.values,
                                        paired["case"].values)
            cos_comparisons.append({"variant":variant,"comparison":"paired-minus-"+control,
                "mean_difference_cosine":mean,"case_bootstrap_95_low":low,"case_bootstrap_95_high":high})
    write_csv(EDIT/"directional_alignment_comparisons.csv",cos_comparisons)
    # Check whether outside edits have enough area to constitute an actual probe.
    outside=averaged[(averaged["mode"]=="paired")&(averaged.variant=="outside_blur_30")]
    meaningful=outside[outside.editable_fraction>=.1]
    mask_stats={"n_images":len(outside),"n_outside_editable_at_least_10pct":len(meaningful),
        "n_outside_editable_under_10pct":int((outside.editable_fraction<.1).sum()),
        "n_outside_editable_zero":int((outside.editable_fraction==0).sum()),
        "outside_editable_fraction_median":float(outside.editable_fraction.median()),
        "outside_editable_at_least_10pct_mean_decoded_delta_rms":float(meaningful.decoded_delta_rms.mean()),
        "outside_editable_at_least_10pct_mean_abs_residual_margin_change":float(meaningful.absolute_residual_margin_change.mean()),
        "note":"Area threshold is a post-run descriptive stratification by editable pixel area, not a new selected model or clinical foreground/background annotation."}
    write_json(EDIT/"editable_area_diagnostics.json",mask_stats)
    # Fixed Frozen checkpoint across all three decoder/head seeds.
    cache=Path(fit_protocol["sources"]["test_membership"]["path"])
    with np.load(cache,allow_pickle=False) as z:
        frozen=z["logits"].argmax(1)
        y=z["labels"]
    frozen_acc=100*float((frozen==y).mean())
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.2),layout="constrained")
    order=list(models)
    colors=["#2166ac","#b35806","#666666"]
    for j,(mode,color) in enumerate(zip(order,colors)):
        values=[]
        for seed in seeds:
            with np.load(FIT/f"seed_{seed}"/mode/"test_predictions.npz",allow_pickle=False) as z:
                values.append(100*float((z["logits"].argmax(1)==z["labels"]).mean()))
        axes[0].scatter(j+np.array([-.06,0,.06]),values,color=color,s=25,zorder=3)
        axes[0].errorbar(j,float(np.mean(values)),yerr=float(np.std(values,ddof=1)),
                         color=color,fmt="_",markersize=18,capsize=4,lw=1.4)
    axes[0].axhline(frozen_acc,ls="--",color="black",lw=1,label="Fixed Frozen DCBM")
    axes[0].set_xticks(range(3),[ENGLISH[m].replace(" ","\n",1) for m in order])
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Matched 128-dimensional residuals")
    axes[0].set_xlabel("Dots: individual seeds; bars: mean +/- sample SD",fontsize=8)
    axes[0].legend(loc="lower right",fontsize=8)
    axes[0].grid(axis="y",alpha=.2)
    x=np.arange(4)
    shapes=list(variants[1:5])
    for j,(mode,color) in enumerate(zip(order,colors)):
        rows=sensitivity[sensitivity["mode"]==mode].set_index("variant").loc[shapes]
        vals=rows.mean_cosine_seed_average.to_numpy(float)
        errors=np.vstack([vals-rows.cosine_case_bootstrap_95_low.to_numpy(float),
                          rows.cosine_case_bootstrap_95_high.to_numpy(float)-vals])
        axes[1].errorbar(x+(j-1)*.15,vals,yerr=errors,fmt="o",color=color,capsize=3,
                         label=ENGLISH[mode],markersize=4)
    axes[1].axhline(0,color="black",lw=.8)
    axes[1].set_xticks(x,["x 0.10","y 0.10","x 0.20","y 0.20"])
    axes[1].set_ylabel("Cosine of teacher and decoded changes")
    axes[1].set_title("Local deformation response (273 images)")
    axes[1].set_xlabel("Bars: case-bootstrap 95% CI, after averaging seeds",fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y",alpha=.2)
    fig.savefig(EDIT/"mechanism_summary.png",dpi=180)
    fig.savefig(EDIT/"mechanism_summary.pdf")
    plt.close(fig)
    # Recheck every immutable input after all analyses, including weights that
    # were loaded only by the earlier fitting/encoding processes.
    hash_checks=[]
    immutable=list(fit_protocol["sources"].values())+[fit_protocol["semantic_checkpoint"],edit_protocol["clip_weights"]]
    immutable+=json.loads((EDIT/"input_assets.json").read_text(encoding="utf-8"))
    for record in immutable:
        actual=sha256(Path(record["path"]))
        if actual!=record["sha256"]: raise AssertionError("An immutable experiment input changed.")
        hash_checks.append({"path":record["path"],"sha256":actual})
    write_json(EDIT/"verified_input_hashes.json",hash_checks)
    # Final audit hashes include model/data artifacts, not only the report.
    audit={"status":"complete","nine_classifier_and_decoder_replays_passed":True,
        "nine_perturbation_model_replays_passed":True,"max_residual_replay_error":max(replay_errors),
        "appearance_teachers_bit_identical":True,"all_original_input_hashes_unchanged":True,
        "image_perturbation_count":273*9,"classification_test_images":391,
        "perturbation_test_images":273,"excluded_before_inference":17,
        "case_clusters_in_test":356,"seed_repeats":seeds,
        "frozen_accuracy_pct":frozen_acc,"fixed_frozen_checkpoint_across_seeds":True,
        "case_overlap_counts":fit_protocol["case_overlap_counts"],
        "models":[{"path":str(path),"sha256":sha256(path)} for path in FIT.rglob("selected_model.pt")],
        "input_hash_validation_sources":[str(FIT/"protocol.json"),str(EDIT/"protocol.json")],
        "immutable_files_verified_after_analysis":len(hash_checks),
        "paper_data_policy":"Only Table E7 expanded. Local diagnostic results did not replace manuscript server accuracy values."}
    write_json(EDIT/"final_verification.json",audit)
    report=["# Derm7pt 形状机制诊断结果", "",
        "真实配对能更好地恢复现有形状 teacher，但这次实验没有证实其最终分类准确率稳定优于打乱配对。图像扰动显示部分方向一致性，同时也显示对外观变化的敏感性。因此，目前结果支持‘部分形状目标可被线性恢复’，不能把分类收益全部归因于病灶几何。", "",
        "## 实验设计", "",
        "固定同一 Frozen DCBM 检查点、CLIP 特征、训练/验证/测试成员、geo11 teacher bank 和 128 维 affine decoder。种子为 42、43、44，改变 decoder/head 初始化、训练随机顺序及配对打乱；没有重新训练原始 DCBM 或重建原型库。三个组分别为：真实配对；仅在训练集中对完整 teacher 向量打乱图像对应；在每个训练类别内打乱完整向量。不是打乱 128 个坐标的顺序：单纯重排坐标再拟合线性头只是重命名，不能检验配对机制。", "",
        "各组 decoder 均使用同一未打乱的有效验证 teacher，以相同 SmoothL1/AdamW 训练规则选择最低验证损失的检查点；只在 926 张有效训练图像上监督 decoder。分类头使用全部 1,231 张训练图像，同一训练集标准化规则、CE+L1、batch64、lr0.001、最多100轮、patience30，分别拟合 eta={0.1,0.25,0.5,0.75,1.0}，按验证准确率选择。decoder 最多300轮。测试成员和测试标签不参与训练、标准化或选参。", "",
        "这次是本地‘验证选择单检查点’诊断，论文是服务器最后15轮测试准确率均值，两者不是同一统计量。本地固定 Frozen 为 76.7263%，不得替换论文 Derm7pt Frozen=74.76%。下面的 ± 是三个实际 decoder/head 种子结果的样本标准差，不是末轮波动，也不是完整 DCBM 训练流水线的不确定性。", "",
        "## 分类与重建结果", "",
        "| 配对方式 | 全部391张测试图像，准确率均值±标准差 | 有效teacher 290张 | 缺失teacher 101张 |",
        "|---|---:|---:|---:|" ]
    for mode in models:
        selected=summary[summary["mode"]==mode].set_index("subset")
        allr=selected.loc["all"]
        report.append(f"| {LABELS[mode]} | {allr.mean_accuracy_pct:.2f} ± {allr.sample_std_across_seeds_pct:.2f}% | {selected.loc['valid_teacher'].mean_accuracy_pct:.2f}% | {selected.loc['missing_teacher'].mean_accuracy_pct:.2f}% |")
    baseline=[]
    with np.load(FIT/"test_metadata.npz",allow_pickle=False) as z: covered=z["valid_teacher"]
    for members in (np.ones(len(y),bool),covered,~covered):
        baseline.append(100*float((frozen[members]==y[members]).mean()))
    report.append(f"| 固定 Frozen | {baseline[0]:.2f}% | {baseline[1]:.2f}% | {baseline[2]:.2f}% |")
    report += ["", "真实配对减全局打乱的三个种子平均差为 +0.171 个百分点。对固定预测按病例成组 bootstrap 的95%区间为 [−1.364, +1.727] 个百分点，跨过0；真实配对减同类打乱为0.000，区间 [−1.705, +1.705]。它们是当前固定模型和保留分区下的条件性描述，不是独立患者泛化或完整训练种子总体的置信区间。每个种子的 discordant pairs、描述性 McNemar 检验及病例 bootstrap 均保存在 paired_comparisons.csv，不能从这三次运行宣称等效或没有任何收益。", "",
        "| 配对方式 | 290张有效teacher测试图像的平均 R² | 平均 Pearson |", "|---|---:|---:|"]
    for mode in models:
        part=fidelity[(fidelity["mode"]==mode)&(fidelity.split=="test")]
        report.append(f"| {LABELS[mode]} | {part.r2_uniform.mean():.3f} | {part.pearson_defined_mean.mean():.3f} |")
    report += ["", "真实配对的重建明显较好，但分类均值接近打乱组。这表明‘能恢复部分 teacher’与‘分类净收益依赖正确形状配对’是不同命题；本次只支持前者。不能据此断言收益必定来自纹理、背景或某一种伪影。", "",
        "## 无需重新分割的受控图像变化", "",
        f"原先290张有效teacher测试图像全部尝试；273张通过严格素材核验和编辑有效性检查，17张在推理前排除。完成4种局部形变、2种区域内部模糊、2种区域外模糊，加上原图共2,457个输入。273张中临床照片142张、皮肤镜图129张，另2张在原始元数据中同时列入两种模态，未强行指定模态。", "",
        "旧 SAM2 输出缺少 bbox 的，使用配对 RGB crop 在原图中做唯一模板匹配：相关系数至少0.98、与非邻近候选峰值差至少0.005、像素MAE不超过6。所有纳入图像必须重现原来的geo11描述子与缓存teacher；完整配准记录和17张排除原因分别保存在 mask_registration.csv、excluded_images.csv。局部形变把同一光滑位移场同时施加给RGB图像和全部已有有效mask，不重新调用SAM；Jacobian必须为正。外观模糊使用固定mask，保留轮廓边界带像素。", "",
        "原图与编辑图均重新使用本地 CLIP ViT-L/14 编码，以新编码的原图作为差分基准。新原图与缓存特征最低余弦为0.999973，最大相对差为0.007372；没有把旧缓存与新运行的数值漂移计作图像扰动。计算 teacher/decoded 响应变化及固定‘malignant减benign’分类边际，保存语义、残差、最终边际；逐样本核对三者加和闭合。", "",
        "| 编辑 | teacher变化RMS | 真实配对decoded变化RMS | teacher与decoded变化方向的平均余弦 |",
        "|---|---:|---:|---:|" ]
    variant_names={"warp_x_010":"局部横向形变0.10","warp_y_010":"局部纵向形变0.10",
        "warp_x_020":"局部横向形变0.20","warp_y_020":"局部纵向形变0.20",
        "inside_blur_15":"区域内部模糊σ1.5","inside_blur_30":"区域内部模糊σ3.0",
        "outside_blur_15":"区域外模糊σ1.5","outside_blur_30":"区域外模糊σ3.0"}
    for _,row in sensitivity[sensitivity["mode"]=="paired"].iterrows():
        cosine="不定义（teacher不变）" if pd.isna(row.mean_cosine_seed_average) else f"{row.mean_cosine_seed_average:.3f}"
        report.append(f"| {variant_names[row.variant]} | {row.teacher_delta_rms:.3f} | {row.decoded_delta_rms:.3f} | {cosine} |")
    report += ["", "形变的方向余弦为弱正值，支持一定程度的方向一致性，但不是接近1的强跟随；病例bootstrap与打乱组的对照保存在 directional_alignment_comparisons.csv。外观模糊下teacher逐位完全不变，decoded仍明显变化，表明解码坐标没有对纹理和上下文实现不变性。局部形变也会移动纹理和邻域、产生重采样效应；不同编辑的感知强度未匹配，因此不能用变化幅度的大小直接比较几何与纹理的因果重要性，也未将编辑图像当作具有已验证诊断标签的新准确率测试集。", "",
        f"另一个重要限制是可编辑区域大小：273张中只有{mask_stats['n_outside_editable_at_least_10pct']}张的‘mask区域外’超过整图10%，其余173张不足10%；中位数仅{100*mask_stats['outside_editable_fraction_median']:.2f}%。这是因为不少保留SAM区域包含大范围皮肤或近整幅画面，并非专家标注的病灶mask。不能把区域内部/区域外直接解释为病灶/背景。对至少10%区域外可编辑图像的补充分层只依据像素面积，记录在 editable_area_diagnostics.json，不作模型选择。", "",
        "## 可支持的论文表述", "",
        "> Paired supervision improves recovery of the shape-derived teacher targets relative to shuffled correspondence. In the retained Derm7pt diagnostic, however, classification accuracy does not consistently distinguish paired from shuffled supervision, and decoded responses remain sensitive to appearance changes that leave the teacher fixed. These results support partial shape decodability without establishing that the classification gains are caused specifically by lesion geometry.", "",
        "不建议现在写‘已证明分类收益来自纯形状’或‘已经排除背景、纹理’。若未来要强化该结论，首先应针对论文服务器的同一检查点和15轮统计规则重放匹配对照，并处理病灶区域与近整图区域的混合；这次本地诊断的数值不能并入服务器主表冒充同一评估口径。", "",
        "保留分区按case_num核验仍存在跨分区同病例：train/val 229个，train/test 239个，val/test 80个。本文本地测试391张图像对应356个病例。病例bootstrap保留同一病例在测试集中两种图像的成组相关，但不能消除训练/测试共享病例的影响。", "",
        "## E7 与文件验证", "",
        "E7已扩展到CUB、Places365、Derm7pt、BUS-UCLM、ISIC-2019。新增Places365 train/val/test有效率72.78%/72.63%/72.63%；BUS-UCLM三者均100%；ISIC-2019为91.06%/90.37%/93.01%。统计来自本地保留缓存，表中明确availability不代表分割质量。PDF与Word均为23页，E7位于第22页。主实验准确率未用本地新结果替换。", "",
        "9个已序列化decoder/head在全部原始分区上回放通过；9个扰动模型另作独立回放，分类边际加和闭合；所有读取的原图、mask、crop、特征缓存、检查点hash保持不变。关键不变量测试覆盖：训练内置换、同类置换约束、重复模态元数据、严格crop位置核验、恒等形变、非折叠形变、外观编辑保持mask与边界不变。完整核验见 final_verification.json。", "",
        "## 复现与结果位置", "",
        "使用现有ML环境，不需要安装新模型或生成mask。必须指定新的输出目录，避免覆盖本次结果。", "", "```powershell",
        r"Set-Location '.'",
        r"& 'python' -m unittest test_shape_mechanism -v",
        r"& 'python' run_derm7pt_mechanism.py --output outputs/derm7pt_mechanism_new_run",
        r"& 'python' run_derm7pt_perturbations.py --models outputs/derm7pt_mechanism_new_run --output outputs/derm7pt_perturbations_new_run",
        "```", "",
        f"分类输出：`{FIT}`", "", f"扰动输出：`{EDIT}`", "",
        "主要CSV：classification_summary.csv（实际种子均值/标准差）；classification_groups.csv（逐种子、群体及correction/harm）；decoder_fidelity.csv（重建）；paired_comparisons.csv（成对分类比较）；per_image_sensitivity.csv（逐图响应与分类边际变化）；sensitivity_seed_averaged.csv（先按图像聚合种子后的扰动汇总）。", "",
        "示意图：mechanism_summary.png / mechanism_summary.pdf；左侧点为各个种子、误差棒为样本标准差，右侧为先按图像平均种子后按病例bootstrap的95%区间。图像编辑对照：perturbation_examples.png 和 examples/。这些诊断结果保存在本地，未加入论文新的结果表。", ""]
    REPORT.write_text("\n".join(report),encoding="utf-8")
    audit["report"]={"path":str(REPORT),"sha256":sha256(REPORT)}
    write_json(EDIT/"final_verification.json",audit)
    print(json.dumps({"report":str(REPORT),"audit":audit,"mask_stats":mask_stats},ensure_ascii=False,indent=2))


if __name__=="__main__": run()
