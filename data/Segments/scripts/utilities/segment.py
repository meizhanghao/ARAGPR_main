from .data_loading import load_CUB_bb
import torch
import cv2
import numpy as np
import json
import os
import uuid
import hashlib
from pathlib import Path


MASK_PIPELINE_VERSION = 2


def asset_sha256(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def load_paired_mask_records(mask_root):
    """Validate a paired binary-mask bank before rebuilding a shape teacher."""
    mask_root = Path(mask_root).resolve()
    asset_root = mask_root.parent
    records = {}
    manifests = sorted(mask_root.rglob("*_MASKS_DONE.json"))
    if not manifests:
        raise ValueError(f"No paired-mask manifests in {mask_root}")
    for path in manifests:
        basename = path.name.removesuffix("_MASKS_DONE.json")
        folder = path.parent.relative_to(mask_root).as_posix()
        manifest = load_mask_manifest(mask_root, folder, basename)
        if not manifest or not manifest.get("paired_outputs"):
            raise ValueError(f"Missing or invalid paired outputs: {path}")
        regions = manifest.get("regions", [])
        if len(regions) != manifest["num_masks"]:
            raise ValueError(f"Incomplete region provenance: {path}")
        for region, expected_id in zip(regions, manifest["region_ids"]):
            if region.get("region_id") != expected_id:
                raise ValueError(f"Region ID mismatch: {path}")
            mask = path.parent / f"{basename}_OUT_MASK_{expected_id}_X_.png"
            crop = asset_root / "crops" / folder / f"{basename}_OUT_CROP_{expected_id}_X_.jpg"
            if not mask.is_file() or not crop.is_file():
                raise ValueError(f"Missing paired mask/crop: {mask}")
            raw = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(str(crop))
            if raw is None or raw.ndim != 2 or not np.isin(raw, [0, 255]).all():
                raise ValueError(f"Expected a grayscale binary PNG: {mask}")
            if rgb is None or rgb.shape[:2] != raw.shape:
                raise ValueError(f"Crop/mask dimensions differ: {mask}")
            size = [raw.shape[1], raw.shape[0]]
            if size != region.get("size_wh") or size != region.get("bbox_xywh", [])[2:]:
                raise ValueError(f"BBox/crop dimensions differ: {mask}")
            mask_hash, crop_hash = asset_sha256(mask), asset_sha256(crop)
            if mask_hash != region.get("mask_sha256") or crop_hash != region.get("crop_sha256"):
                raise ValueError(f"Paired output changed after segmentation: {mask}")
            records[mask] = {
                "mask_relpath": mask.relative_to(asset_root).as_posix(),
                "crop_relpath": crop.relative_to(asset_root).as_posix(),
                "mask_sha256": mask_hash, "crop_sha256": crop_hash,
            }
    actual = {p.resolve() for p in mask_root.rglob("*_OUT_MASK_*_X_.*")}
    if actual != set(records):
        raise ValueError(f"Uncommitted or mixed-version masks in {mask_root}")
    if not records:
        raise ValueError(f"No usable paired masks in {mask_root}")
    print(f"[Paired masks] verified {len(records)} binary PNG/crop pairs, {len(manifests)} images")
    return records


def mask_manifest_path(mask_root, ifolder, ipath):
    return Path(mask_root) / ifolder / f"{ipath}_MASKS_DONE.json"


def load_mask_manifest(mask_root, ifolder, ipath):
    """Return a validated per-image mask manifest, or None when incomplete."""
    manifest_path = mask_manifest_path(mask_root, ifolder, ipath)
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("pipeline_version") != MASK_PIPELINE_VERSION:
            return None
        expected_ids = [int(i) for i in manifest.get("region_ids", [])]
        num_masks = int(manifest.get("num_masks", -1))
        if num_masks < 0 or expected_ids != list(range(num_masks)):
            return None
        mask_dir = manifest_path.parent
        # Accept both PNG (new pipeline) and JPG (legacy masks)
        actual_names = set()
        for ext_pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
            for p in mask_dir.glob(f"{ipath}_OUT_MASK_*_X_.{ext_pat.split('.')[-1]}"):
                actual_names.add(p.name)
        expected_names = set()
        for i in expected_ids:
            seen = False
            for ext in (".png", ".PNG", ".jpg", ".JPG"):
                if (mask_dir / f"{ipath}_OUT_MASK_{i}_X_{ext}").exists():
                    expected_names.add(f"{ipath}_OUT_MASK_{i}_X_{ext}")
                    seen = True
                    break
            if not seen:
                return None
        if actual_names != expected_names:
            return None
        return manifest
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def filter_valid_masks(masks, image_shape):
    """Normalize SAM bboxes and discard degenerate or empty mask regions."""
    image_h, image_w = image_shape[:2]
    valid = []
    for mask in masks:
        try:
            x, y, w, h = [float(v) for v in mask["bbox"]]
            x0 = max(0, int(np.floor(x)))
            y0 = max(0, int(np.floor(y)))
            x1 = min(image_w, int(np.ceil(x + w)))
            y1 = min(image_h, int(np.ceil(y + h)))
            if x1 <= x0 or y1 <= y0:
                continue
            segmentation = np.asarray(mask["segmentation"], dtype=bool)
            if segmentation.shape[:2] != (image_h, image_w):
                continue
            if not segmentation[y0:y1, x0:x1].any():
                continue
            normalized = dict(mask)
            normalized["bbox"] = [x0, y0, x1 - x0, y1 - y0]
            normalized["segmentation"] = segmentation
            valid.append(normalized)
        except (KeyError, TypeError, ValueError):
            continue
    return valid

def create_statistics(boxes, img_path, dataset_name):
    sizes = [b[2]* b[3] for b in boxes]
    s = {"num_segments": len(boxes), "segments": boxes, "sizes": sizes, "overlap with gt": []}
    # Calculate if pred is overlapping with ground truth (boolean)
    if dataset_name == "CUB":
        bb_gt = load_CUB_bb()
        img_id = img_path.split("/")[-1].split(".")[0]
        gt = bb_gt[img_id]
        for b in boxes:
            overlap = False
            if b[0] <= gt[0] or b[2] >= gt[2] and b[1] <= gt[1] or b[3] >= gt[3]:
                overlap = True
            s["overlap with gt"].append(overlap)
    return s

def create_crops(boxes, phrases, image_source, out_path, ifolder, ipath):

    # Loop through the masks and display/save them
    for i, box in enumerate(boxes):
        try:
            # Extract the bounding box and mask
            h, w, _ = image_source.shape
            box = box * torch.Tensor([w, h, w, h])
            x, y, w, h = box.cpu().numpy().astype(int)
            # 跳过退化的零面积 bbox
            if w <= 0 or h <= 0:
                continue
            # create crop of images
            cropped = image_source[y:y+h, x:x+w]
            if cropped.size == 0:
                continue
            file_name = f"{out_path}/crops/{ifolder}/{ipath}_OUT_CROP_{i}_{phrases[i]}.jpg"
            cv2.imwrite(file_name, cropped)
        except Exception as e:
            print(f"  ⚠ 保存 crop 失败 [{ipath}] box={box}: {e}")
            # print error in try except block to avoid stopping the script

    return ipath

def check_non_seg(boxes, ipath):
    no_masks_count = 0
    img_wo_masks = []
    if len(boxes) == 0:
        no_masks_count += 1
        img_wo_masks.append(ipath)
    return no_masks_count, img_wo_masks


def create_crops_masks(masks, image, out_path, ifolder, ipath, mask_only=False,
                       source_metadata=None):
    """Transactionally replace one image's crops and binary PNG masks.

    A completion manifest is written only after every expected output succeeds.
    An interrupted or failed write therefore remains retryable.
    """
    mask_dir = Path(out_path) / "masks" / ifolder
    crop_dir = Path(out_path) / "crops" / ifolder
    mask_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    manifest_final = mask_manifest_path(
        Path(out_path) / "masks", ifolder, ipath
    )
    # The manifest is the commit marker. Invalidate the previous run before
    # replacing any output so a failed retry can never appear complete.
    manifest_final.unlink(missing_ok=True)
    mask_outputs = []
    crop_outputs = []
    temp_paths = []
    manifest_tmp = None
    try:
        for i, mask in enumerate(masks):
            x, y, w, h = map(int, mask["bbox"])
            seg_mask = np.asarray(
                mask["segmentation"][y:y + h, x:x + w], dtype=bool
            )
            if seg_mask.shape != (h, w) or not seg_mask.any():
                raise ValueError(f"Invalid cropped segmentation for region {i}")

            binary_mask = seg_mask.astype(np.uint8) * 255
            mask_final = mask_dir / f"{ipath}_OUT_MASK_{i}_X_.png"
            mask_tmp = mask_dir / f".{ipath}_{i}_{token}.tmp.png"
            if not cv2.imwrite(str(mask_tmp), binary_mask):
                raise IOError(f"Failed to write temporary mask: {mask_tmp}")
            mask_outputs.append((mask_tmp, mask_final))
            temp_paths.append(mask_tmp)

            if not mask_only:
                cropped = image[y:y + h, x:x + w]
                if cropped.shape[:2] != (h, w) or cropped.size == 0:
                    raise ValueError(f"Invalid crop for region {i}")
                # image is already BGR (caller guarantees this);
                # save directly without extra colour conversion.
                crop_final = crop_dir / f"{ipath}_OUT_CROP_{i}_X_.jpg"
                crop_tmp = crop_dir / f".{ipath}_{i}_{token}.tmp.jpg"
                if not cv2.imwrite(str(crop_tmp), cropped):
                    raise IOError(f"Failed to write temporary crop: {crop_tmp}")
                crop_outputs.append((crop_tmp, crop_final))
                temp_paths.append(crop_tmp)

        expected_mask_names = {final.name for _, final in mask_outputs}
        for tmp, final in mask_outputs:
            os.replace(tmp, final)
        for stale in mask_dir.glob(f"{ipath}_OUT_MASK_*_X_.png"):
            if stale.name not in expected_mask_names:
                stale.unlink()

        if not mask_only:
            expected_crop_names = {final.name for _, final in crop_outputs}
            for tmp, final in crop_outputs:
                os.replace(tmp, final)
            for stale in crop_dir.glob(f"{ipath}_OUT_CROP_*_X_.jpg"):
                if stale.name not in expected_crop_names:
                    stale.unlink()

        manifest = {
            "pipeline_version": MASK_PIPELINE_VERSION,
            "num_masks": len(mask_outputs),
            "region_ids": list(range(len(mask_outputs))),
            "binary_values": [0, 255],
            "paired_outputs": not mask_only,
            "processed_size_wh": [image.shape[1], image.shape[0]],
            "source": source_metadata or {},
            "regions": [
                {
                    "region_id": i,
                    "bbox_xywh": list(map(int, mask["bbox"])),
                    "size_wh": list(map(int, mask["bbox"][2:])),
                    "mask_sha256": asset_sha256(mask_outputs[i][1]),
                    "crop_sha256": asset_sha256(crop_outputs[i][1]) if not mask_only else None,
                }
                for i, mask in enumerate(masks)
            ],
        }
        manifest_tmp = manifest_final.with_name(
            f".{manifest_final.name}.{token}.tmp"
        )
        with manifest_tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(manifest_tmp, manifest_final)
        return ipath, len(crop_outputs), 0, len(mask_outputs), 0
    except Exception:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if manifest_tmp is not None:
            try:
                manifest_tmp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def export_contours_from_masks(mask_root, contour_root, thickness=1, overwrite=False):
    """从 masks/ 中的二值 PNG 提取主轮廓，输出黑线白底轮廓图到 contours/。

    严格顺序：
      1. 读取二值 mask（白=前景 255，黑=背景 0）；
      2. 二值化（>127=255）以防旧数据污染；
      3. findContours(RETR_EXTERNAL) → 取最大外轮廓（与形状描述符一致）；
      4. 在白画布(255)上画黑色轮廓(0)。

    Args:
        mask_root:    masks 目录路径
        contour_root: contours 输出目录路径
        thickness:    轮廓线宽 (默认 1，特征提取; 论文展示可用 2-3)
        overwrite:    是否覆盖已有轮廓文件

    Returns:
        dict: {"total": int, "processed": int, "skipped": int, "failed": int}
    """
    from pathlib import Path

    mask_root = Path(mask_root)
    contour_root = Path(contour_root)

    mask_paths = sorted(mask_root.rglob("*_OUT_MASK_*_X_.png"))
    if not mask_paths:
        # Fallback: old pipeline versions saved masks as JPG
        mask_paths = sorted(mask_root.rglob("*_OUT_MASK_*_X_.jpg"))
    if not mask_paths:
        print(f"[Contour Export] No mask files (PNG/JPG) found in {mask_root}")
        return {
            "total": 0, "processed": 0, "skipped": 0, "failed": 0,
            "incomplete": 0, "unverified": 0,
        }

    processed, skipped, failed, incomplete, unverified = 0, 0, 0, 0, 0
    manifest_cache = {}

    for mask_path in mask_paths:
        relative = mask_path.relative_to(mask_root)
        folder = relative.parent.as_posix()
        if folder == ".":
            folder = ""
        basename = mask_path.name.split("_OUT_MASK_", 1)[0]
        group_key = (folder, basename)
        if group_key not in manifest_cache:
            manifest_cache[group_key] = load_mask_manifest(
                mask_root, folder, basename
            )
        # Fallback: if manifest is missing (masks from old pipeline),
        # still generate contours — just track as unverified.
        if manifest_cache[group_key] is None:
            unverified += 1
            # fall through to process the mask
        elif manifest_cache[group_key].get("num_masks", -1) == 0:
            # Manifest says zero masks — genuinely nothing to draw
            incomplete += 1
            continue

        output_name = mask_path.name.replace("_OUT_MASK_", "_OUT_CONTOUR_")
        # Always save contours as .png (lossless) regardless of source mask format
        output_name = os.path.splitext(output_name)[0] + ".png"
        output_path = contour_root / relative.parent / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            # Re-generate if mask is newer than contour (mask was re-exported)
            if mask_path.stat().st_mtime <= output_path.stat().st_mtime:
                skipped += 1
                continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            failed += 1
            continue

        binary = (mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        contours = [c for c in contours if len(c) >= 3 and cv2.contourArea(c) > 0]
        if not contours:
            failed += 1
            continue

        main_contour = max(contours, key=cv2.contourArea)

        canvas = np.full(binary.shape, 255, dtype=np.uint8)
        cv2.drawContours(
            canvas,
            [main_contour],
            contourIdx=-1,
            color=0,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

        if cv2.imwrite(str(output_path), canvas):
            processed += 1
        else:
            failed += 1

    print(f"[Contour Export] total={len(mask_paths)} processed={processed} "
          f"skipped={skipped} failed={failed} incomplete={incomplete} "
          f"unverified={unverified}")
    return {"total": len(mask_paths), "processed": processed,
            "skipped": skipped, "failed": failed,
            "incomplete": incomplete, "unverified": unverified}
