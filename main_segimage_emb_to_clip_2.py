# 第二步，将第一步分割好的图片传入CLIP里，segcrop_SAM2_RN50.torch
import torch,clip,glob,os,argparse,sys,subprocess
import pandas as pd

from data.Segments.scripts.utilities.CLIP import fix_embeddings, compute_embeddings_batched
from data.Segments.scripts.utilities.data_loading import load_segments

# ---------------------------------------------------------------------------
#  derm7pt 二分类标签工具
# ---------------------------------------------------------------------------

def _build_derm7pt_label_map():
    """
    从 meta.csv 建立二分类标签映射。

    返回:
        label_map : dict  {原始图像文件名(小写): 'benign'|'malignant'}
        meta_csv_path : str  (供日志输出)
    """
    meta_csv_path = "./dataset_all/derm7pt/meta/meta.csv"
    if not os.path.exists(meta_csv_path):
        raise FileNotFoundError(f"找不到 meta.csv: {meta_csv_path}")

    meta_df = pd.read_csv(meta_csv_path)

    def _to_binary(diagnosis):
        d = str(diagnosis).lower()
        if 'melanoma' in d or 'basal cell carcinoma' in d:
            return 'malignant'
        return 'benign'

    label_map = {}
    for _, row in meta_df.iterrows():
        binary_label = _to_binary(row['diagnosis'])
        for col in ['clinic', 'derm']:
            rel_path = str(row[col]) if pd.notna(row[col]) else ''
            if rel_path:
                filename = os.path.basename(rel_path).lower()
                label_map[filename] = binary_label

    n_benign = sum(1 for v in label_map.values() if v == 'benign')
    n_malignant = sum(1 for v in label_map.values() if v == 'malignant')
    print(f"  derm7pt 二分类标签映射: {len(label_map)} 文件 "
          f"(benign={n_benign}, malignant={n_malignant})")
    return label_map


def _build_isic2019_label_map():
    """
    从 ISIC_2019_Training_GroundTruth.csv 和 ISIC_2019_Test_GroundTruth.csv
    建立 9 分类标签映射。

    CSV 格式: image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK  (one-hot, 值为 1.0 的列即标签)
    测试集 CSV 末尾有额外的 score_weight, validation_weight 列, 不影响读取。

    返回:
        label_map : dict  {原始图像文件名(小写, 含扩展名): 'MEL'|'NV'|...}
    """
    CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    label_map = {}

    for split, csv_rel_path in [
        ('train', './dataset_all/ISIC-2019/train/ISIC_2019_Training_GroundTruth.csv'),
        ('test',  './dataset_all/ISIC-2019/test/ISIC_2019_Test_GroundTruth.csv'),
    ]:
        if not os.path.exists(csv_rel_path):
            print(f"  [ISIC-2019] 跳过 {split}: 找不到 {csv_rel_path}")
            continue
        df = pd.read_csv(csv_rel_path)
        for _, row in df.iterrows():
            image_id = str(row['image']).strip().lower()
            for cls in CLASSES:
                if float(row[cls]) == 1.0:
                    label_map[image_id + '.jpg'] = cls
                    break

    from collections import Counter
    dist = Counter(label_map.values())
    print(f"  ISIC-2019 标签映射: {len(label_map)} 文件, {len(dist)} 类 "
          f"({dict(dist)})")
    return label_map


def _build_bus_uclm_label_map():
    """
    从 BUS-UCLM_labels.csv 建立三分类标签映射。

    CSV 格式: filename,label,split,case_prefix,mask_path
    label 取值: benign, malignant, normal

    返回:
        label_map : dict  {原始图像文件名(小写): 'benign'|'malignant'|'normal'}
    """
    csv_path = "./dataset_all/BUS-UCLM/BUS-UCLM_labels.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 BUS-UCLM_labels.csv: {csv_path}，请先运行 "
                                f"dataset_all/BUS-UCLM/BUS_UCLM_csv.py")

    df = pd.read_csv(csv_path)
    label_map = {}
    for _, row in df.iterrows():
        filename = str(row['filename']).strip().lower()
        label = str(row['label']).strip()
        label_map[filename] = label

    from collections import Counter
    dist = Counter(label_map.values())
    print(f"  BUS-UCLM 标签映射: {len(label_map)} 文件, {len(dist)} 类 ({dict(dist)})")
    return label_map


def _extract_original_basename(seg_key):
    """
    从段键提取原始图像文件名（小写）。

    段键格式: Aal002_OUT_CROP_0_X_.jpg  →  原始图像: Aal002.JPG
    返回: "aal002" (小写 basename，不含扩展名)
    """
    for marker in ['_OUT_CROP_', '_OUT_SHAPE_', '_OUT_MASK_']:
        if marker in seg_key:
            return seg_key.split(marker)[0].lower()
    # 无已知标记，返回整个键（去除扩展名）
    return os.path.splitext(seg_key)[0].lower()


def _fix_segment_labels(torch_path, label_map):
    """
    加载已生成的 .torch 文件，将段键映射为正确的类别标签，原地覆盖保存。

    段键 → 提取原始图像 basename → 查找 label_map → 真实类别名

    匹配策略（按优先级）:
      1. 完整 basename + 扩展名  (如 "apartment_building/outdoor/0001.jpg")
      2. 纯文件名 + 扩展名        (如 "0001.jpg"，用于 BUS_UCLM/derm7pt 等扁平日录数据集)
      3. 完整 basename 直接匹配
      4. 纯文件名直接匹配
    """
    print(f"  修正标签 (label_map 共 {len(label_map)} 条)...")
    data = torch.load(torch_path, weights_only=False, map_location="cpu")
    fixed = 0
    missing = 0
    for seg_key in list(data.keys()):
        full_basename = _extract_original_basename(seg_key)
        # 同时准备纯文件名 (去掉目录前缀)，用于扁平日录数据集 (BUS_UCLM, derm7pt 等)
        filename_only = full_basename.split('/')[-1] if '/' in full_basename else full_basename
        candidates = [full_basename, filename_only] if filename_only != full_basename else [full_basename]

        new_label = None
        for cand in candidates:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if cand + ext in label_map:
                    new_label = label_map[cand + ext]
                    break
            if new_label is None and cand in label_map:
                new_label = label_map[cand]
            if new_label is not None:
                break

        if new_label is not None:
            old_emb = data[seg_key][0]
            data[seg_key] = [old_emb, new_label]
            fixed += 1
        else:
            missing += 1

    torch.save(data, torch_path)
    print(f"    修正: {fixed} 段, 未匹配: {missing}")
    return data


# ---------------------------------------------------------------------------

def load_emb_model(emb, device):
    # --emb 参数直接对应 CLIP 模型名 (RN50, ViT-L/14, ViT-B/16, ...)
    # 格式: "CLIP-{model_name}" → 去掉 "CLIP-" 前缀得到实际模型名
    if emb.startswith("CLIP-"):
        clip_model = emb[5:]  # e.g. "CLIP-RN50" → "RN50", "CLIP-ViT-L14" → "ViT-L14"
    else:
        clip_model = emb
    model, preprocess = clip.load(clip_model, device=device)
    model_name = f"CLIP-{clip_model}".replace("/", "")
    model.eval()
    return model, preprocess, model_name

def embed_segment_type(seg_dir, seg_type, dataset, seg_model, model_name, prompt, model, preprocess, device, label_map=None):
    """对一种分割类型（crops / shapes）下的图片做 CLIP 嵌入，保存为 .torch 文件。
    当 label_map 不为 None 时（derm7pt 二分类），生成文件后修正标签。"""
    # 确定输出文件名
    if seg_type == "shape":
        # shape 强制命名为 segshape_...（不从目录名派生，因为目录名是 "shapes" 复数形式）
        if seg_model in ["GDINO", "SemSAM"]:
            out_file = f"data/Segments/Seg_embs/segshape_{dataset}_{seg_model}_{prompt}_{model_name}.torch"
        else:
            out_file = f"data/Segments/Seg_embs/segshape_{dataset}_{seg_model}_{model_name}.torch"
        print("out_file:", out_file)
    else:
        # 原有逻辑：从 seg_type（如 "crops"）去掉末尾 's' 拼文件名，如 segcrop_...
        if seg_model in ["GDINO", "SemSAM"]:
            out_file = f"data/Segments/Seg_embs/seg{seg_type[:-1]}_{dataset}_{seg_model}_{prompt}_{model_name}.torch"
        else:
            out_file = f"data/Segments/Seg_embs/seg{seg_type[:-1]}_{dataset}_{seg_model}_{model_name}.torch"
        print("out_file:", out_file)

    # 收集该目录下所有 jpg 图片 (recursive: 支持多级类目录如 crops/apartment_building/outdoor/)
    segs = glob.glob(seg_dir + "/**/*.jpg", recursive=True)
    print(seg_dir + "/**/*.jpg")

    if not os.path.exists(out_file):
        print(f"computing embeddings for {seg_model} segments of type {seg_type}: {len(segs)} segments")
        compute_embeddings_batched(segs, model, preprocess, out_file, device, batch_size=64)
        # derm7pt: 修正标签为二分类
        if label_map is not None:
            _fix_segment_labels(out_file, label_map)
    else:
        embs = torch.load(out_file, weights_only=False, map_location="cuda:0")
        # 检查是否有已知的损坏文件跳过列表
        skip_path = out_file.replace('.torch', '_corrupted_skip.txt')
        skip_set = set()
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                skip_set = set(line.strip() for line in f if line.strip())
        # 从 segs 中排除已知损坏文件
        segs_filtered = [s for s in segs if s not in skip_set]
        if len(embs) == len(segs_filtered):
            # derm7pt: 即使数量匹配也检查是否需要修正标签
            if label_map is not None:
                _fix_segment_labels(out_file, label_map)
            else:
                torch.save(embs, out_file)
            print(f"Embeddings already exist for {seg_type} segments of {dataset} using {seg_model} and {model_name}")
            if skip_set:
                print(f"  (已跳过 {len(skip_set)} 个已知损坏文件)")
        else:
            print(f"Fixing embeddings for {seg_model} segments of type {seg_type}.")
            seg_dict = {i.replace('\\', '/').split("/")[-1]: i for i in segs}
            missing_keys = [i for i in seg_dict.keys() if i not in embs.keys()]
            print(f"缺失嵌入: {len(missing_keys)}")
            seg2emb = {i: seg_dict[i] for i in missing_keys}
            missing_embs = fix_embeddings(seg2emb, model, preprocess, device)
            embs.update(missing_embs)
            # 记录永久损坏的文件 (fix_embeddings 跳过但未成功处理的)
            still_missing = [k for k in missing_keys if k not in embs]
            if still_missing:
                new_skips = {seg_dict[k] for k in still_missing}
                print(f"⚠ {len(still_missing)} 个文件无法读取 (已损坏), 将跳过:")
                for p in sorted(new_skips):
                    print(f"  - {p}")
                skip_set.update(new_skips)
                with open(skip_path, 'w') as f:
                    f.write('\n'.join(sorted(skip_set)))
                print(f"损坏文件列表已保存至: {skip_path}")
            torch.save(embs, out_file)
            # derm7pt: 修正标签为二分类
            if label_map is not None:
                _fix_segment_labels(out_file, label_map)
            print(f"Embeddings updated for {seg_type} segments of {dataset} using {seg_model} and {model_name}")


def _build_places365_label_map():
    """
    从 categories_places365.txt 读取类别名, 建立 {类别名: 类别名} 的恒等映射。

    places365_part 的 crops 目录结构为 crops/{class_name}/*.jpg,
    compute_embeddings_batched 已从父目录名自动提取类别标签,
    此处仅用于验证类别列表的完整性。

    返回:
        label_map : dict  {class_name: class_name}
        class_list : list  所有类别名
    """
    categories_path = "./dataset_all/places365_part/categories_places365.txt"
    if not os.path.exists(categories_path):
        raise FileNotFoundError(f"找不到类别文件: {categories_path}")

    classes = []
    with open(categories_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)

    label_map = {c: c for c in classes}
    print(f"  places365_part 类别: {len(classes)} 类")
    return label_map


def main(emb, dataset, seg_model, prompt, device):
    # load embedding model
    model, preprocess, model_name = load_emb_model(emb, device)

    # derm7pt / ISIC-2019 / BUS_UCLM: 从 CSV 构建标签映射, 后续修正嵌入文件的标签
    # places365_part / CUB_200_2011: 标签来自 crops 目录结构 (crops/{class_name}/), 无需修正
    label_map = None
    if dataset == "derm7pt":
        label_map = _build_derm7pt_label_map()
    elif dataset == "ISIC-2019":
        label_map = _build_isic2019_label_map()
    elif dataset == "BUS_UCLM":
        label_map = _build_bus_uclm_label_map()
    elif dataset == "places365_part":
        _build_places365_label_map()  # 验证类别文件存在, 标签由目录结构自动提供
        # label_map 保持 None — compute_embeddings_batched 从父目录名提取类别

    # 处理 crops
    seg_paths = load_segments(dataset, seg_model, prompt)
    for i in seg_paths:
        i_norm = i.replace('\\', '/')
        seg_type = i_norm.split("/")[-1]
        if seg_type in ["crops"]:
            embed_segment_type(i_norm, seg_type, dataset, seg_model, model_name, prompt,
                               model, preprocess, device, label_map=label_map)


# ══════════════════════════════════════════════════════════════════════
#  多 GPU 支持
# ══════════════════════════════════════════════════════════════════════

def _split_list(lst, n):
    """将列表尽量均匀地分成 n 份."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def _collect_crop_files(dataset, seg_model, prompt):
    """收集所有 crop JPEG 路径, 返回列表."""
    seg_paths = load_segments(dataset, seg_model, prompt)
    all_files = []
    for i in seg_paths:
        i_norm = i.replace('\\', '/')
        seg_type = i_norm.split("/")[-1]
        if seg_type == "crops":
            files = glob.glob(i_norm + "/**/*.jpg", recursive=True)
            all_files.extend(files)
    return all_files


def _run_embed_worker(args):
    """Worker 进程: 绑定指定 GPU, 处理分配给它的 crop 子集."""
    device = f"cuda:{args.worker_gpu}"
    model, preprocess, model_name = load_emb_model(args.emb, device)

    # 读取分配的 crop 列表
    with open(args.worker_list, 'r') as f:
        crop_files = [line.strip() for line in f if line.strip()]

    if not crop_files:
        print(f"[GPU {args.worker_gpu}] 空列表, 跳过")
        return

    # 确定输出文件名
    if args.worker_prompt == "None":
        seg_model_name = args.seg_model
    else:
        seg_model_name = f"{args.seg_model}_{args.worker_prompt}"

    out_file = (f"data/Segments/Seg_embs/"
                f"segcrop_{args.dataset}_{seg_model_name}_{model_name}"
                f"_part{args.worker_part}.torch")

    print(f"[GPU {args.worker_gpu}] {len(crop_files)} crops → {out_file}")
    compute_embeddings_batched(crop_files, model, preprocess, out_file, device, batch_size=64)

    # derm7pt / ISIC-2019 / BUS_UCLM: 修正标签
    if args.dataset == "derm7pt":
        label_map = _build_derm7pt_label_map()
        _fix_segment_labels(out_file, label_map)
    elif args.dataset == "ISIC-2019":
        label_map = _build_isic2019_label_map()
        _fix_segment_labels(out_file, label_map)
    elif args.dataset == "BUS_UCLM":
        label_map = _build_bus_uclm_label_map()
        _fix_segment_labels(out_file, label_map)

    print(f"[GPU {args.worker_gpu}] 完成 → {out_file}")


def run_multi_gpu(args):
    """多 GPU 主控: 收集 crops → 划分 → 启动 worker → 合并."""
    gpu_ids = [int(x.strip()) for x in args.multi_gpu.split(',')]
    num_gpus = len(gpu_ids)
    print(f"\n{'='*60}")
    print(f"  Multi-GPU 模式: {num_gpus} GPUs — {gpu_ids}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    # 1. 收集所有 crop 文件
    all_crops = _collect_crop_files(args.dataset, args.seg_model, args.prompts)
    print(f"总 crop 文件数: {len(all_crops)}")
    if len(all_crops) == 0:
        print("没有找到 crop 文件，退出。")
        return

    # 2. 均匀划分
    splits = _split_list(all_crops, num_gpus)

    # 3. 确定输出文件名 (与单 GPU 模式一致)
    model, preprocess, model_name = load_emb_model(args.emb, "cuda:0")
    if args.prompts == "None":
        seg_model_name = args.seg_model
    else:
        seg_model_name = f"{args.seg_model}_{args.prompts}"
    final_out = f"data/Segments/Seg_embs/segcrop_{args.dataset}_{seg_model_name}_{model_name}.torch"

    if os.path.exists(final_out):
        print(f"输出文件已存在: {final_out}")
        print("如需重新生成请先删除此文件。")
        return

    # 4. 写入临时列表文件, 启动 worker
    os.makedirs("_temp_embed_lists", exist_ok=True)
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        list_file = f"_temp_embed_lists/{args.dataset}_part{i}.txt"
        with open(list_file, 'w') as f:
            f.write('\n'.join(splits[i]))

        cmd = [
            sys.executable, __file__,
            '--dataset', args.dataset,
            '--emb', args.emb,
            '--seg_model', args.seg_model,
            '--prompts', args.prompts,
            '--worker_gpu', str(gpu_id),
            '--worker_list', list_file,
            '--worker_part', str(i),
            '--worker_prompt', args.prompts,
        ]
        print(f"  [GPU {gpu_id}] {len(splits[i])} crops")
        p = subprocess.Popen(cmd)
        processes.append((i, gpu_id, p))

    # 5. 等待所有 worker 完成
    print(f"\n等待 {num_gpus} 个 worker 完成...\n")
    for i, gpu_id, p in processes:
        ret = p.wait()
        status = "OK" if ret == 0 else f"ERR({ret})"
        print(f"  [GPU {gpu_id}] worker {i} → {status}")

    # 6. 合并部分文件
    print(f"\n合并临时文件 → {final_out}")
    merged = {}
    for i in range(num_gpus):
        part_file = (f"data/Segments/Seg_embs/"
                     f"segcrop_{args.dataset}_{seg_model_name}_{model_name}"
                     f"_part{i}.torch")
        if not os.path.exists(part_file):
            print(f"  ⚠ 缺少: {part_file}")
            continue
        part = torch.load(part_file, weights_only=False, map_location="cpu")
        overlap = set(merged.keys()) & set(part.keys())
        if overlap:
            print(f"  ⚠ 重复键: {len(overlap)} 个, 跳过旧值")
            for k in overlap:
                del part[k]
        merged.update(part)
        os.remove(part_file)
        print(f"  已合并 part {i}: {len(part)} entries")

    torch.save(merged, final_out)
    print(f"\n✅ 合并完成: {len(merged)} embeddings → {final_out}")

    # 7. 清理临时文件
    for i in range(num_gpus):
        list_file = f"_temp_embed_lists/{args.dataset}_part{i}.txt"
        if os.path.exists(list_file):
            os.remove(list_file)
    try:
        os.rmdir("_temp_embed_lists")
    except OSError:
        pass

    print("\nMulti-GPU 嵌入完成。\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute embeddings')
    parser.add_argument('--emb', type=str,
                        help='CLIP model: CLIP-RN50, CLIP-ViT-B16, CLIP-ViT-L14')
    parser.add_argument('--dataset', type=str, help='Dataset segments to embed')
    parser.add_argument('--seg_model', type=str,
                        help='Segmentation model: SAM, SAM2, SemSAM, GDINO')
    parser.add_argument('--prompts', default='None', type=str,
                        help='Prompts used for segmentation (if applicable)')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='单 GPU 模式下的设备 (default: cuda:1)')

    # 多 GPU 参数
    parser.add_argument('--multi_gpu', type=str, default=None,
                        help='多 GPU 模式: 逗号分隔的 GPU 编号, 如 "0,2,3"')

    # 内部 worker 参数 (由 run_multi_gpu 自动传递, 用户无需关心)
    parser.add_argument('--worker_gpu', type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_list', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_part', type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_prompt', type=str, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ── 分发 ──
    if args.worker_gpu is not None:
        # Worker 模式 (内部使用)
        _run_embed_worker(args)
    elif args.multi_gpu is not None:
        # 多 GPU 主控模式
        run_multi_gpu(args)
    else:
        # 原始单 GPU 模式 (完全向后兼容)
        main(args.emb, args.dataset, args.seg_model, args.prompts, args.device)

    # 用法:

    #   单 GPU:  python main_segimage_emb_to_clip_2.py --dataset CUB_200_2011 --device cuda:0 --emb CLIP-ViT-L/14 --seg_model SAM2
    #   多 GPU:  python main_segimage_emb_to_clip_2.py --dataset places365_part --multi_gpu "0,2,3" --emb CLIP-ViT-L/14 --seg_model SAM2
