# 使用SAM2对数据集进行分割,输出data/Segments/Seg_embs/stats/{dataset}_SAM2_STATS.torch和分割好的crops/masks
import numpy as np
import os, sys, torch, glob, tqdm, argparse, subprocess, uuid
from collections import defaultdict
from data.Segments.scripts.utilities.SAM2 import generate_mask
from data.Segments.scripts.utilities.segment import load_mask_manifest

torch.cuda.empty_cache()


_STATS_META_KEYS = {
    "total_no_masks_count", "total_img_wo_masks", "failed_images",
    "processed_count",
}


def image_output_identity(img_path, dataset):
    """Return the exact output folder, basename and stable image key.

    Places categories retain their complete category path. This prevents leaf
    names such as ``indoor`` or ``outdoor`` from merging distinct classes.
    """
    normalized = img_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot derive output identity from: {img_path}")
    basename = os.path.splitext(parts[-1])[0]

    if dataset == "places365_part":
        root_indices = [
            i for i, part in enumerate(parts[:-1])
            if part.lower() == "places365_part"
        ]
        split_index = root_indices[-1] + 1 if root_indices else None
        if (split_index is None or split_index >= len(parts) - 1
                or parts[split_index] not in {"train", "val", "test"}):
            split_indices = [
                i for i, part in enumerate(parts[:-1])
                if part in {"train", "val", "test"}
            ]
            split_index = split_indices[-1] if split_indices else None
        if split_index is None:
            raise ValueError(f"Missing split directory in places365_part path: {img_path}")
        folder = "/".join(parts[split_index + 1:-1])
    elif dataset == "places365":
        root_indices = [
            i for i, part in enumerate(parts[:-1])
            if part.lower() == "places365"
        ]
        train_index = root_indices[-1] + 1 if root_indices else None
        if (train_index is None or train_index >= len(parts) - 1
                or parts[train_index] != "train"):
            train_indices = [i for i, part in enumerate(parts[:-1]) if part == "train"]
            train_index = train_indices[-1] if train_indices else None
        if train_index is None:
            raise ValueError(f"Missing train directory in Places365 path: {img_path}")
        folder = "/".join(parts[train_index + 1:-1])
    else:
        folder = parts[-2]

    if not folder:
        raise ValueError(f"Empty output folder for: {img_path}")
    key = f"{folder}/{basename}"
    return folder, basename, key


def _audit_output_identities(train_paths, dataset):
    """Deduplicate source images that map to the same output key.

    When train/val/test splits contain the same basename under the same class
    (e.g. ``airfield/00000113.jpg`` in both train and val), only the first
    occurrence is kept — later duplicates are dropped.  This is safe because
    crops/masks are written once per (class, basename) and the existing output
    already represents whichever split was processed first.
    """
    by_key = defaultdict(list)
    for path in train_paths:
        by_key[image_output_identity(path, dataset)[2]].append(path)
    collisions = {k: v for k, v in by_key.items() if len(set(v)) > 1}
    if collisions:
        n_dropped = sum(len(v) - 1 for v in collisions.values())
        examples = list(collisions.items())[:3]
        detail = "\n".join(
            f"  {key}: keeping {paths[0]}, dropping {len(paths)-1} duplicate(s)"
            for key, paths in examples
        )
        print(f"⚠  {len(collisions)} output keys have {n_dropped} duplicate "
              f"source images (same basename in multiple splits).  "
              f"Keeping first occurrence only.\n{detail}")
    # Return deduplicated path list (keep first occurrence per key)
    deduped = [paths[0] for paths in by_key.values()]
    return deduped


def _has_any_mask_files(mask_root, folder, basename):
    """Check whether ANY _OUT_MASK_*_X_.png file exists for (folder, basename).

    This is the last-resort fallback when neither a *_MASKS_DONE.json manifest
    nor a usable stats record is available (e.g. old-format stats with flat
    basename keys when basenames collide across multi-level categories).
    """
    mask_dir = os.path.join(mask_root, *folder.split("/")) if folder else mask_root
    if not os.path.isdir(mask_dir):
        return False
    pattern = os.path.join(mask_dir, f"{basename}_OUT_MASK_*_X_.*")
    return len(glob.glob(pattern)) > 0


def _legacy_mask_set_matches(mask_root, folder, basename, expected_count):
    """Validate legacy outputs that predate per-image completion manifests."""
    mask_dir = os.path.join(mask_root, *folder.split("/")) if folder else mask_root
    actual_names = set()
    for ext in ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG"):
        actual_names.update(
            os.path.basename(path)
            for path in glob.glob(
                os.path.join(mask_dir, f"{basename}_OUT_MASK_*_X_.{ext}")
            )
        )
    expected_names = set()
    for i in range(expected_count):
        found = False
        for ext in ("png", "PNG", "jpg", "JPG"):
            name = f"{basename}_OUT_MASK_{i}_X_.{ext}"
            if os.path.exists(os.path.join(mask_dir, name)):
                expected_names.add(name)
                found = True
                break
        if not found:
            return False
    return actual_names == expected_names


def _record_dict(stats):
    return {
        k: v for k, v in stats.items()
        if k not in _STATS_META_KEYS and isinstance(v, dict)
    }


def merge_stats_state(old_stats, new_stats):
    """Merge stats while allowing successful retries to clear stale failures."""
    old_stats = dict(old_stats or {})
    new_stats = dict(new_stats or {})
    old_records = _record_dict(old_stats)
    new_records = _record_dict(new_stats)
    records = dict(old_records)
    records.update(new_records)

    def zero_segment_keys(records):
        result = set()
        for key, value in records.items():
            try:
                if int(value.get("num_segments", -1)) == 0:
                    result.add(key)
            except (TypeError, ValueError):
                continue
        return result

    def positive_segment_keys(records):
        result = set()
        for key, value in records.items():
            try:
                if int(value.get("num_segments", -1)) > 0:
                    result.add(key)
            except (TypeError, ValueError):
                continue
        return result

    old_no_seg = (
        set(old_stats.get("total_img_wo_masks", []))
        | zero_segment_keys(old_records)
    )
    new_no_seg = zero_segment_keys(new_records)
    successful = positive_segment_keys(new_records)
    no_seg = (old_no_seg - successful) | new_no_seg

    failed = set(old_stats.get("failed_images", []))
    failed.update(new_stats.get("failed_images", []))
    failed.difference_update(new_records.keys())

    merged = records
    merged["total_img_wo_masks"] = sorted(no_seg)
    merged["total_no_masks_count"] = len(no_seg)
    if failed:
        merged["failed_images"] = sorted(failed)
    return merged


# ══════════════════════════════════════════════════════════════════════
#  SAM2 模型加载
# ══════════════════════════════════════════════════════════════════════

def load_model(device):
    from sam2.build_sam import build_sam2
    import urllib.request

    # Config: relative to the sam2 package (Hydra resolves via pkg://sam2)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Checkpoint: ./seg_pt/, auto-download if missing
    os.makedirs("seg_pt", exist_ok=True)
    sam2_checkpoint = os.path.join("seg_pt", "sam2.1_hiera_large.pt")
    if not os.path.exists(sam2_checkpoint):
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        print(f"[SAM2] Checkpoint not found. Downloading to {sam2_checkpoint} ...")
        print(f"[SAM2] URL: {url}")
        urllib.request.urlretrieve(url, sam2_checkpoint)
        print("[SAM2] Download complete.")

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2 = sam2.to(device)
    return sam2


# ══════════════════════════════════════════════════════════════════════
#  数据集路径加载 (支持 CUB_200_2011 / derm7pt / BUS_UCLM / ISIC-2019)
# ══════════════════════════════════════════════════════════════════════

def segmentation_paths(dataset, model="SAM2", output_dir=None):
    if output_dir:
        root = os.path.abspath(output_dir)
        return root, os.path.join(root, "stats.torch")
    return (f"data/Segments/Seg_embs/{dataset}_{model}",
            f"data/Segments/Seg_embs/stats/{dataset}_{model}_STATS.torch")


def load_data(dataset_name, model, prompt_specs, check_existing,
              image_root=None, output_dir=None):
    if image_root:
        if dataset_name != "CUB_200_2011":
            raise ValueError("--image_root is currently supported for CUB_200_2011 only")
        from pathlib import Path
        root = Path(image_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CUB image directory not found: {root}")
        train_paths = sorted(str(p).replace('\\', '/') for p in root.rglob('*')
                             if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not train_paths:
            raise ValueError(f"No CUB images found in {root}")
        print(f"Reading CUB source images from {root}")
    elif dataset_name == "CUB_200_2011":
        with open("./dataset_all/CUB_200_2011/images.txt") as f:
            train_imgs = f.readlines()
        img_dir = "./dataset_all/CUB_200_2011/images/"
        train_paths = [
            os.path.normpath(
                os.path.join(
                    img_dir,
                    line.strip().split(' ', 1)[-1]
                )
            )
            for line in train_imgs
            if line.strip() and len(line.strip().split(' ', 1)) > 1
        ]
        train_paths = [path.replace('\\', '/') for path in train_paths]
        print("正在读取CUB数据集...")
    elif dataset_name == "derm7pt":
        img_dir = os.path.abspath("./dataset_all/derm7pt/images/")
        train_paths = glob.glob(os.path.join(img_dir, "*", "*.jpg")) + \
                      glob.glob(os.path.join(img_dir, "*", "*.JPG")) + \
                      glob.glob(os.path.join(img_dir, "*", "*.JPEG")) + \
                      glob.glob(os.path.join(img_dir, "*", "*.jpeg"))
        train_paths = [path.replace('\\', '/') for path in train_paths]
        print("正在读取derm7pt数据集...")
    elif dataset_name == "BUS_UCLM":
        img_dir = os.path.abspath("./dataset_all/BUS-UCLM/BUS-UCLM/images/")
        train_paths = glob.glob(os.path.join(img_dir, "*.png"))
        train_paths = [path.replace('\\', '/') for path in train_paths]
        print("正在读取BUS-UCLM数据集...")
    elif dataset_name == "ISIC-2019":
        train_img_dir = os.path.abspath("./dataset_all/ISIC-2019/train/ISIC_2019_Training_Input/")
        train_paths = glob.glob(os.path.join(train_img_dir, "*.jpg")) + \
                      glob.glob(os.path.join(train_img_dir, "*.JPG")) + \
                      glob.glob(os.path.join(train_img_dir, "*.JPEG")) + \
                      glob.glob(os.path.join(train_img_dir, "*.jpeg"))
        test_img_dir = os.path.abspath("./dataset_all/ISIC-2019/test/ISIC_2019_Test_Input/")
        train_paths += glob.glob(os.path.join(test_img_dir, "*.jpg")) + \
                       glob.glob(os.path.join(test_img_dir, "*.JPG")) + \
                       glob.glob(os.path.join(test_img_dir, "*.JPEG")) + \
                       glob.glob(os.path.join(test_img_dir, "*.jpeg"))
        train_paths = [path.replace('\\', '/') for path in train_paths]
        print("正在读取ISIC-2019数据集...")
    elif dataset_name == "places365":
        # 读取 categories_places365.txt: "/a/airfield 0"
        # 只使用 train 集 (test_val 无标签)，下游会自行 6:2:2 划分 train/val/test
        txt_path = os.path.abspath("./dataset_all/Places365/categories_places365.txt")
        train_img_base = os.path.abspath("./dataset_all/Places365/train/")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到类别文件: {txt_path}")
        train_paths = []
        class_count = 0
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rel_path = parts[0].lstrip('/')  # e.g., "a/airfield"
                    class_dir = os.path.join(train_img_base, rel_path)
                    if not os.path.isdir(class_dir):
                        print(f"  ⚠ 类别目录不存在: {class_dir}")
                        continue
                    class_count += 1
                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            train_paths.append(
                                os.path.join(class_dir, fname).replace('\\', '/'))
        print(f"正在读取places365数据集... ({class_count} 个类别, 共 {len(train_paths)} 张图片)")
    elif dataset_name == "places365_part":
        # places365_part: 每类 1000 张, 已分好 train/val/test (6:2:2)
        # categories_places365.txt 只有纯类名，无下标
        # 目录结构: train/{class}/, val/{class}/, test/{class}/
        txt_path = os.path.abspath("./dataset_all/places365_part/categories_places365.txt")
        img_root = os.path.abspath("./dataset_all/places365_part/")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到类别文件: {txt_path}")
        class_names = []
        seen_classes = set()
        with open(txt_path, 'r') as f:
            for line in f:
                name = line.strip().replace('\\', '/').strip('/')
                if name:
                    if any(part in {"", ".", ".."} for part in name.split('/')):
                        raise ValueError(f"非法 Places365 类别路径: {name}")
                    if name in seen_classes:
                        raise ValueError(f"重复 Places365 类别: {name}")
                    seen_classes.add(name)
                    class_names.append(name)
        train_paths = []
        class_count = 0
        for cls in class_names:
            found = False
            for split in ["train", "val", "test"]:
                class_dir = os.path.join(img_root, split, cls)
                if os.path.isdir(class_dir):
                    if not found:
                        class_count += 1
                        found = True
                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            train_paths.append(
                                os.path.join(class_dir, fname).replace('\\', '/'))
        print(f"正在读取places365_part数据集... ({class_count} 个类别, 共 {len(train_paths)} 张图片)")
    else:
        raise ValueError(
            f"未知数据集: {dataset_name}，"
            f"支持: CUB_200_2011, derm7pt, BUS_UCLM, ISIC-2019, places365, places365_part")
    train_paths = _audit_output_identities(train_paths, dataset_name)
    basename_counts = defaultdict(int)
    for _, basename, _ in (
        image_output_identity(path, dataset_name) for path in train_paths
    ):
        basename_counts[basename] += 1

    if check_existing:
        # ═══════════════════════════════════════════════════════════════
        #  Completion check based on BINARY PNG MASKS (not crops).
        #  Old JPG masks in crops/ are NOT a valid completion signal.
        #
        #  A current manifest is authoritative. Legacy outputs are accepted only
        #  when stats identify the image unambiguously and the on-disk mask set
        #  is exactly {0, ..., num_segments - 1}.
        # ═══════════════════════════════════════════════════════════════
        asset_root, stats_path = segmentation_paths(dataset_name, model, output_dir)
        try:
            stats = torch.load(stats_path)
        except Exception:
            stats = {}

        masks_dir = os.path.join(asset_root, "masks")

        records = _record_dict(stats)
        legacy_no_seg = set(stats.get("total_img_wo_masks", []))
        complete_keys = set()
        no_seg_keys = set()

        # Pre-scan manifests (single rglob → fast, avoids N disk I/O calls)
        manifest_cache = {}
        for mf in glob.glob(os.path.join(masks_dir, "**", "*_MASKS_DONE.json"),
                             recursive=True):
            mf = mf.replace("\\", "/")
            # Parse path: masks_dir/{folder}/{basename}_MASKS_DONE.json
            rel = mf[len(masks_dir) + 1:]  # "{folder}/{basename}_MASKS_DONE.json"
            fname = os.path.basename(rel)
            basename = fname.rsplit("_MASKS_DONE.json", 1)[0]
            folder = os.path.dirname(rel)
            if folder == ".":
                folder = ""
            # Lightweight check: just verify the file exists, don't full-validate
            manifest_cache[(folder, basename)] = True

        train_paths_ver = []
        for path in tqdm.tqdm(train_paths, desc="  check_existing",
                               disable=len(train_paths) < 1000):
            folder, basename, key = image_output_identity(path, dataset_name)
            if (folder, basename) in manifest_cache:
                complete_keys.add(key)
                continue

            record = records.get(key)
            if record is None and basename_counts[basename] == 1:
                record = records.get(basename)

            complete = False
            if record is not None:
                try:
                    expected_count = int(record["num_segments"])
                except (KeyError, TypeError, ValueError):
                    expected_count = -1
                if expected_count >= 0:
                    complete = _legacy_mask_set_matches(
                        masks_dir, folder, basename, expected_count
                    )
                    if complete and expected_count == 0:
                        no_seg_keys.add(key)

            if not complete:
                listed_no_seg = key in legacy_no_seg
                if basename_counts[basename] == 1:
                    listed_no_seg = listed_no_seg or basename in legacy_no_seg
                if listed_no_seg:
                    complete = _legacy_mask_set_matches(
                        masks_dir, folder, basename, 0
                    )
                    if complete:
                        no_seg_keys.add(key)

            # ▸ 最终兜底：直接检查磁盘上的 mask 文件 (manifest 缺失 + stats
            #   旧格式 key 时，record 可能为 None，但 mask 可能实际存在)
            if not complete:
                complete = _has_any_mask_files(masks_dir, folder, basename)
                if complete:
                    complete_keys.add(key)
                    continue

            if complete:
                complete_keys.add(key)
            else:
                train_paths_ver.append(path)

        n_skipped_noseg = len(no_seg_keys)
        n_skipped_masks = len(complete_keys - no_seg_keys)
        print(f"  check_existing: {n_skipped_masks} done (masks), "
              f"{n_skipped_noseg} done (no segments)")
    else:
        train_paths_ver = train_paths
    print("为", len(train_paths_ver), "个images进行分割")
    if len(train_paths_ver) == 0:
        print("没有待分割图片，将检查已有输出。")
        return [], dataset_name
    return train_paths_ver, dataset_name


# ══════════════════════════════════════════════════════════════════════
#  核心分割循环 (单 GPU / worker 共用)
# ══════════════════════════════════════════════════════════════════════

def run_segmentation_loop(sam2, train_paths, out_path, dataset, create_stats,
                          create_segments, mask_only=False, checkpoint_tag=None):
    """遍历 train_paths，逐张调用 SAM2 分割，返回 (stats_dict, total_no_masks, total_img_wo).

    每 500 张图片自动保存一次 stats checkpoint，防止长时运行中断导致数据丢失。
    """
    stats = {}
    total_no_masks_count = 0
    total_img_wo_masks = []
    stats_dir = os.path.join(out_path, "checkpoints")
    os.makedirs(stats_dir, exist_ok=True)
    suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    checkpoint_path = (
        f"{stats_dir}/{dataset}_SAM2_STATS_checkpoint{suffix}.torch"
    )
    failed_images = []

    for idx, img_path in enumerate(tqdm.tqdm(train_paths, desc="Segmenting")):
        try:
            ifolder, expected_basename, stats_key = image_output_identity(
                img_path, dataset
            )
            s, ipath, no_masks_count, img_wo_masks = generate_mask(
                sam2, img_path, out_path, create_stats, create_segments, dataset,
                ifolder=ifolder, mask_only=mask_only)
            if ipath != expected_basename:
                raise RuntimeError(
                    f"Output basename mismatch: {ipath} != {expected_basename}"
                )
            stats[stats_key] = s
            total_no_masks_count += no_masks_count
            total_img_wo_masks += img_wo_masks
        except Exception as e:
            print(f"\n  ⚠ 分割失败 [{img_path}]: {e}")
            try:
                failed_key = image_output_identity(img_path, dataset)[2]
            except ValueError:
                failed_key = img_path
            failed_images.append(failed_key)
            continue

        # 每 500 张保存一次断点
        if (idx + 1) % 500 == 0:
            ckpt = dict(stats)
            ckpt["total_no_masks_count"] = total_no_masks_count
            ckpt["total_img_wo_masks"] = total_img_wo_masks
            ckpt["failed_images"] = failed_images
            ckpt["processed_count"] = idx + 1
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(ckpt, checkpoint_path)

    if failed_images:
        print(f"\n  ⚠ 共 {len(failed_images)} 张图片分割失败")
        stats["failed_images"] = failed_images

    # 清理 checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return stats, total_no_masks_count, total_img_wo_masks


# ══════════════════════════════════════════════════════════════════════
#  单 GPU 模式 (原始入口)
# ══════════════════════════════════════════════════════════════════════

def main_single(dataset, create_segments, create_stats, device, check_existing,
                 use_lunkuo=False, mask_only=False, image_root=None, output_dir=None):
    prompt_specs = None
    train_paths, dataset_name = load_data(dataset, "SAM2", prompt_specs, check_existing,
                                         image_root=image_root, output_dir=output_dir)
    out_path, stats_path = segmentation_paths(dataset_name, output_dir=output_dir)
    os.makedirs(out_path, exist_ok=True)

    if len(train_paths) > 0:
        sam2 = load_model(device)
        stats, total_no, total_wo = run_segmentation_loop(
            sam2, train_paths, out_path, dataset, create_stats, create_segments,
            mask_only=mask_only)
    else:
        print("No images to segment, skipping SAM2 load.")
        stats = {}; total_no = 0; total_wo = []

    print("Mask generation completed，抠图完毕.")
    print(f"Total stats collected，一共收集了多少数据: {len(stats)}")
    print("Total no masks count，有多少图片未抠图出来:", total_no)
    stats["total_no_masks_count"] = total_no
    stats["total_img_wo_masks"] = total_wo

    # 保存 / 合并 stats
    stats_dir = os.path.dirname(stats_path)
    os.makedirs(stats_dir, exist_ok=True)
    old_stats = torch.load(stats_path) if os.path.exists(stats_path) else {}
    stats = merge_stats_state(old_stats, stats)
    torch.save(stats, stats_path)
    print(f"Stats saved: {stats_path} ({len(_record_dict(stats))} images)")

    # 轮廓导出 (单卡)
    if use_lunkuo:
        from pathlib import Path
        from data.Segments.scripts.utilities.segment import export_contours_from_masks
        contour_stats = export_contours_from_masks(
            Path(out_path) / "masks",
            Path(out_path) / "contours",
        )
        print(f"Contour export: {contour_stats}")


# ══════════════════════════════════════════════════════════════════════
#  多 GPU 支持
# ══════════════════════════════════════════════════════════════════════

def split_list(lst, n):
    """将列表尽量均匀地分成 n 份."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def merge_worker_stats(dataset, num_parts, run_id, output_dir=None):
    """合并 worker 产生的临时 stats_part_*.torch → 最终 stats 文件, 并清理临时文件."""
    part_paths = [
        f"stats/stats_{dataset}_SAM2_{run_id}_part{i}.torch"
        for i in range(num_parts)
    ]
    missing = [path for path in part_paths if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"Missing worker stats files: {missing}")

    merged_new = {}
    for part_path in part_paths:
        part = torch.load(part_path)
        duplicate_keys = set(_record_dict(merged_new)) & set(_record_dict(part))
        if duplicate_keys:
            raise RuntimeError(
                f"Duplicate image keys across workers: {sorted(duplicate_keys)[:5]}"
            )
        merged_new = merge_stats_state(merged_new, part)

    # 与已存在的旧 stats 合并
    _, final_path = segmentation_paths(dataset, output_dir=output_dir)
    stats_dir = os.path.dirname(final_path)
    os.makedirs(stats_dir, exist_ok=True)
    old = torch.load(final_path) if os.path.exists(final_path) else {}
    merged = merge_stats_state(old, merged_new)

    torch.save(merged, final_path)
    for part_path in part_paths:
        os.remove(part_path)
    print(
        f"\n合并完成: {len(_record_dict(merged))} images, "
        f"{merged['total_no_masks_count']} without masks → {final_path}"
    )
    return merged


def run_as_worker(args):
    """Worker 进程: 绑定指定 GPU, 处理分配给它的图像子集."""
    device = f"cuda:{args.worker_gpu}"

    # 读取分配的图像列表
    with open(args.worker_list, 'r') as f:
        train_paths = [line.strip() for line in f if line.strip()]

    out_path, _ = segmentation_paths(args.dataset, output_dir=args.output_dir)
    os.makedirs(out_path, exist_ok=True)

    sam2 = load_model(device)
    stats, total_no, total_wo = run_segmentation_loop(
        sam2, train_paths, out_path, args.dataset,
        args.create_stats, args.create_segments,
        mask_only=args.mask_only,
        checkpoint_tag=f"{args.worker_run_id}_part{args.worker_part}")

    stats["total_no_masks_count"] = total_no
    stats["total_img_wo_masks"] = total_wo

    os.makedirs("stats", exist_ok=True)
    part_file = (
        f"stats/stats_{args.dataset}_SAM2_{args.worker_run_id}_"
        f"part{args.worker_part}.torch"
    )
    torch.save(stats, part_file)
    print(f"\n[GPU {args.worker_gpu}] 完成: {len(stats) - 2} images, "
          f"{total_no} without masks → {part_file}")


def run_multi_gpu(args):
    """多 GPU 主控: 划分图像 → 启动 worker → 合并结果."""
    gpu_ids = [int(x.strip()) for x in args.multi_gpu.split(',') if x.strip()]
    if not gpu_ids:
        raise ValueError("--multi_gpu must contain at least one GPU id")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--multi_gpu contains duplicate GPU ids")
    num_gpus = len(gpu_ids)
    print(f"\n{'='*60}")
    print(f"  Multi-GPU 模式: {num_gpus} GPUs — {gpu_ids}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    # 1. 加载全部图像路径 (check_existing 过滤已在主进程完成)
    prompt_specs = None
    train_paths, dataset_name = load_data(args.dataset, "SAM2", prompt_specs,
                                          args.check_existing, args.image_root, args.output_dir)

    if len(train_paths) == 0:
        print("所有图像已处理完毕，无需分割。")
        # 即使没有待分割图片，仍然可能需要对已有 mask 导出轮廓
        if args.use_lunkuo:
            from pathlib import Path
            from data.Segments.scripts.utilities.segment import export_contours_from_masks
            out_path, _ = segmentation_paths(args.dataset, output_dir=args.output_dir)
            contour_stats = export_contours_from_masks(
                Path(out_path) / "masks",
                Path(out_path) / "contours",
            )
            print(f"Contour export: {contour_stats}")
        return

    print(f"待处理: {len(train_paths)} images, "
          f"每 GPU 约 {len(train_paths) // num_gpus} images\n")

    # 2. 均匀划分
    splits = split_list(train_paths, num_gpus)
    jobs = [(gpu_id, split) for gpu_id, split in zip(gpu_ids, splits) if split]
    run_id = uuid.uuid4().hex[:12]

    # 3. 写入临时列表文件
    os.makedirs("_temp_seg_lists", exist_ok=True)
    processes = []
    list_files = []
    for i, (gpu_id, split) in enumerate(jobs):
        list_file = f"_temp_seg_lists/{args.dataset}_{run_id}_part{i}.txt"
        with open(list_file, 'w') as f:
            f.write('\n'.join(split))
        list_files.append(list_file)

        # 构建 worker 命令
        cmd = [
            sys.executable, __file__,
            '--dataset', args.dataset,
            '--worker_gpu', str(gpu_id),
            '--worker_list', list_file,
            '--worker_part', str(i),
            '--worker_run_id', run_id,
            '--create_segments', str(args.create_segments),
            '--create_stats', str(args.create_stats),
            '--mask_only', str(args.mask_only),
        ]
        if args.output_dir:
            cmd.extend(['--output_dir', args.output_dir])
        # worker 不需要 check_existing (主进程已过滤)
        print(f"  [GPU {gpu_id}] {len(split)} images")
        p = subprocess.Popen(cmd)
        processes.append((i, gpu_id, p))

    # 4. 等待所有 worker 完成
    print(f"\n等待 {len(processes)} 个 worker 完成...\n")
    failures = []
    for i, gpu_id, p in processes:
        ret = p.wait()
        status = "OK" if ret == 0 else f"ERR({ret})"
        print(f"  [GPU {gpu_id}] worker {i} → {status}")
        if ret != 0:
            failures.append((i, gpu_id, ret))

    for list_file in list_files:
        if os.path.exists(list_file):
            os.remove(list_file)

    if failures:
        for i in range(len(jobs)):
            part_path = f"stats/stats_{args.dataset}_SAM2_{run_id}_part{i}.torch"
            if os.path.exists(part_path):
                os.remove(part_path)
        raise RuntimeError(
            f"Segmentation workers failed; stats were not merged: {failures}"
        )

    # 5. 合并 stats
    print(f"\n合并临时 stats 文件...")
    merge_worker_stats(args.dataset, len(jobs), run_id, args.output_dir)

    # 6. 清理临时文件
    try:
        os.rmdir("_temp_seg_lists")
    except OSError:
        pass

    print("\nMulti-GPU 分割完成。\n")

    # 轮廓导出 (多卡，由主进程统一调用一次)
    if args.use_lunkuo:
        from pathlib import Path
        from data.Segments.scripts.utilities.segment import export_contours_from_masks
        out_path, _ = segmentation_paths(args.dataset, output_dir=args.output_dir)
        contour_stats = export_contours_from_masks(
            Path(out_path) / "masks",
            Path(out_path) / "contours",
        )
        print(f"Contour export: {contour_stats}")


# ══════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 分割 — 单GPU / 多GPU")

    # 通用参数
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名: CUB_200_2011, derm7pt, BUS_UCLM, ISIC-2019, ...')
    parser.add_argument('--create_segments', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='是否生成 crops/masks 图片 (default: True)')
    parser.add_argument('--create_stats', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='是否生成 stats (default: True)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='单 GPU 模式下的设备 (default: cuda:0)')
    parser.add_argument('--image_root', type=str, default=None,
                        help='CUB source image directory; searched recursively, with class folders retained')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Separate SAM2 asset directory; writes crops/, masks/, contours/ and stats.torch here')
    parser.add_argument('--check_existing', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='是否跳过已分割的图像 (default: True)')

    # 轮廓导出
    parser.add_argument('--use_lunkuo', '--export_contours',
                        dest='use_lunkuo', action='store_true', default=False,
                        help='分割完成后，将 masks 中的二值 PNG 转换为黑色轮廓、白色背景的轮廓图')

    # Mask-only 模式 (只重建二值 PNG mask，不覆盖已有 crops，避免 BGR 通道交换)
    parser.add_argument('--mask_only', type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help='仅生成/更新二值 PNG mask，跳过 crop 保存 (default: False)。'
                             '用于将旧 JPG mask 升级为二值 PNG，不改变已有 crops。')

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
    parser.add_argument('--worker_run_id', type=str, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.mask_only and not args.create_segments:
        parser.error("--mask_only requires --create_segments true")
    if args.worker_gpu is not None and not args.worker_run_id:
        parser.error("internal worker mode requires --worker_run_id")

    # ── 分发 ──
    if args.worker_gpu is not None:
        # Worker 模式 (内部使用)
        run_as_worker(args)
    elif args.multi_gpu is not None:
        # 多 GPU 主控模式
        run_multi_gpu(args)
    else:
        # 原始单 GPU 模式 (完全向后兼容)
        main_single(args.dataset, args.create_segments, args.create_stats,
                    args.device, args.check_existing, args.use_lunkuo,
                    mask_only=args.mask_only, image_root=args.image_root,
                    output_dir=args.output_dir)

    # 用法示例:
    #   python main_segment_1.py --dataset CUB_200_2011 --device cuda:0 --use_lunkuo                     # 单 GPU
    #   python main_segment_1.py --dataset ISIC-2019 --multi_gpu "0,2,3"                    # 多 GPU
    #   python main_segment_1.py --dataset ISIC-2019 --multi_gpu "0,2,3" --device cuda:0    # --device 在 multi_gpu 模式会被忽略
0
