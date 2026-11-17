"""
SAM2 分割 + 轮廓绘制工具
输入一张图片, 对符合筛选条件的区域用红色轮廓圈出, 保存结果。

用法:
    python -m scripts.preprocessing.sam2_contour --image ./path/to/image.jpg
    python -m scripts.preprocessing.sam2_contour --image ./image.jpg --min_area_ratio 0.2 --output result.jpg
    python -m scripts.preprocessing.sam2_contour --image ./image.jpg --top_k 5                  # 只圈最大的5个
    python -m scripts.preprocessing.sam2_contour --image ./image.jpg --compactness_min 0.4      # 只圈紧凑形状
"""
import argparse, os, sys, cv2, numpy as np, torch

# SAM2 is an optional external dependency.
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_sam2_model(device="cuda:0"):
    """加载 SAM2 模型 (与 main_segment_1.py 同逻辑)."""
    import urllib.request

    # Config: relative to the sam2 package (Hydra resolves via pkg://sam2)
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Checkpoint: ./seg_pt/, auto-download if missing
    os.makedirs("seg_pt", exist_ok=True)
    checkpoint = os.path.join("seg_pt", "sam2.1_hiera_large.pt")
    if not os.path.exists(checkpoint):
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        print(f"[SAM2] Checkpoint not found. Downloading to {checkpoint} ...")
        print(f"[SAM2] URL: {url}")
        urllib.request.urlretrieve(url, checkpoint)
        print("[SAM2] Download complete.")

    print(f"加载 SAM2: {checkpoint}")
    sam2 = build_sam2(cfg, checkpoint, device=device, apply_postprocessing=False)
    sam2 = sam2.to(device)
    return sam2


def compute_region_properties(mask, bbox):
    """从掩膜 + bbox 计算区域的几何属性."""
    x, y, w, h = bbox
    area = np.sum(mask)
    # 紧致度: 4π × area / perimeter², 圆形=1
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        perimeter = 1
        compactness = 0
        elongation = 0
    else:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-8)
        # 延长率: 主轴比 (越大越细长)
        if len(cnt) >= 5:
            (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
            elongation = ma / (MA + 1e-8)  # 短轴/长轴, 圆的=1, 细长的≈0
        else:
            elongation = 1

    return {
        "area": int(area),
        "bbox": (x, y, w, h),
        "compactness": compactness,
        "elongation": elongation,
        "contour": contours[0] if contours else None,
    }


def filter_regions(masks, image_area, args):
    """根据用户参数筛选要圈出的区域, 返回 (regions, reason_maps)."""
    regions = []
    for m in masks:
        mask = m["segmentation"]
        bbox_raw = m["bbox"]  # x, y, w, h
        props = compute_region_properties(mask, bbox_raw)
        area_ratio = props["area"] / image_area

        # ── 用户定义的筛选条件 ──
        if area_ratio < args.min_area_ratio:
            continue
        if area_ratio > args.max_area_ratio:
            continue
        if props["compactness"] < args.compactness_min:
            continue
        if props["compactness"] > args.compactness_max:
            continue
        # ────────────────────────────

        regions.append(props)

    # 如果指定了 top_k, 按面积降序取
    if args.top_k is not None and args.top_k > 0:
        regions.sort(key=lambda r: r["area"], reverse=True)
        regions = regions[:args.top_k]

    return regions


def draw_contours(image, regions, color=(0, 0, 255), thickness=3):
    """在原图上绘制红色轮廓, 返回标注后的图片."""
    out = image.copy()
    for r in regions:
        if r["contour"] is not None:
            cv2.drawContours(out, [r["contour"]], -1, color, thickness)
        # 可选: 同时画 bbox 矩形
        # x, y, w, h = r["bbox"]
        # cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="SAM2 分割 + 红圈标注")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径 (默认: 原文件名_contour.jpg)")
    parser.add_argument("--device", type=str, default="cuda:0")

    # 筛选参数
    parser.add_argument("--min_area_ratio", type=float, default=0.005,
                        help="最小面积占全图比例 (default: 0.5%%)")
    parser.add_argument("--max_area_ratio", type=float, default=0.95,
                        help="最大面积占全图比例 (筛掉全图级别的大块)")
    parser.add_argument("--compactness_min", type=float, default=0.0,
                        help="最小紧致度 (圆形=1.0, 越大越圆)")
    parser.add_argument("--compactness_max", type=float, default=1.0,
                        help="最大紧致度")
    parser.add_argument("--top_k", type=int, default=None,
                        help="只保留面积最大的 K 个区域 (None=全部)")

    # SAM2 参数
    parser.add_argument("--points_per_side", type=int, default=32,
                        help="SAM2 采样密度 (default: 32, 越大越密)")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88,
                        help="预测 IoU 阈值")
    parser.add_argument("--stability_score_thresh", type=float, default=0.90,
                        help="稳定性阈值")

    args = parser.parse_args()

    # 读图
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"找不到图片: {args.image}")
    h, w = image.shape[:2]
    image_area = h * w
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"图片: {w}×{h}, 面积={image_area}")

    # 加载 SAM2
    sam2 = load_sam2_model(args.device)

    # 分割
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=0.5,
        min_mask_region_area=100,
    )
    print("正在分割...")
    masks = mask_generator.generate(image_rgb)
    print(f"SAM2 生成了 {len(masks)} 个区域")

    # 筛选
    regions = filter_regions(masks, image_area, args)
    print(f"筛选后保留 {len(regions)} 个区域")
    for i, r in enumerate(regions):
        print(f"  区域{i+1}: 面积{r['area']}px ({100*r['area']/image_area:.1f}%), "
              f"紧致度={r['compactness']:.3f}")

    # 画圈
    result = draw_contours(image, regions, color=(0, 0, 255), thickness=3)

    # 保存
    out_path = args.output or os.path.splitext(args.image)[0] + "_contour.jpg"
    cv2.imwrite(out_path, result)
    print(f"结果保存至: {out_path}")


if __name__ == "__main__":
    main()
    # 用法示例:
    #   python -m scripts.preprocessing.sam2_contour --image ./test.jpg
    #   python -m scripts.preprocessing.sam2_contour --image ./test.jpg --top_k 5
    #   python -m scripts.preprocessing.sam2_contour --image ./test.jpg --min_area_ratio 0.02 --compactness_min 0.3
