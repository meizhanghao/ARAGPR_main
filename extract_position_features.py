"""
从 SAM2 stats 文件中提取每个 segment 的归一化位置特征 (用于 T-RCB 的 Position Branch).

归一化中心坐标 (cx/W, cy/H)、归一化宽高 (w/W, h/H)、归一化面积 (area/(W*H))。

用法:
    python extract_position_features.py --dataset CUB_200_2011
    python extract_position_features.py --dataset derm7pt --output ./features/
"""
import os, argparse, torch, numpy as np, cv2, glob
from collections import defaultdict

STATS_DIR = "data/Segments/Seg_embs/stats"


def get_image_size(img_path):
    """读取单张图像尺寸。"""
    img = cv2.imread(img_path)
    if img is None:
        # 回退: 尝试不同扩展名
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            alt = os.path.splitext(img_path)[0] + ext
            if os.path.exists(alt):
                img = cv2.imread(alt)
                break
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]  # W, H


def extract_position_features(dataset, img_root):
    stats_path = os.path.join(STATS_DIR, f"{dataset}_SAM2_STATS.torch")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats 不存在: {stats_path}\n请先运行 main_segment_1.py")

    stats = torch.load(stats_path, weights_only=False, map_location="cpu")
    print(f"加载 stats: {stats_path}")
    print(f"  共 {sum(1 for k in stats if isinstance(stats[k], dict))} 张图像")

    # 建立文件名 → 绝对路径映射
    if dataset == "CUB_200_2011":
        split_img_dir = os.path.join(img_root, "images")
        if os.path.isdir(split_img_dir):
            all_imgs = glob.glob(os.path.join(split_img_dir, "*/*.jpg"))
        else:
            all_imgs = glob.glob(os.path.join(img_root, "train/*/*.jpg"))
            all_imgs += glob.glob(os.path.join(img_root, "val/*/*.jpg"))
            all_imgs += glob.glob(os.path.join(img_root, "test/*/*.jpg"))
    elif dataset in ("derm7pt", "BUS_UCLM", "ISIC-2019", "places365_part"):
        all_imgs = []
        for root_dir, _, files in os.walk(img_root):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_imgs.append(os.path.join(root_dir, f))
    else:
        all_imgs = glob.glob(os.path.join(img_root, "**/*.jpg"), recursive=True)

    name2path = {}
    for p in all_imgs:
        name2path[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"  图片路径映射: {len(name2path)} 张")

    # 缓存图片尺寸 (同一张图可能出现在 train/val/test 中)
    size_cache = {}

    features = {}  # {image_name: [[cx_n, cy_n, w_n, h_n, area_n], ...]}
    missing_img = 0
    missing_size = 0

    for img_name, info in stats.items():
        if not isinstance(info, dict):
            continue  # 跳过特殊键 (total_no_masks_count 等)

        # 解析 key: 可能为纯 basename 或 "subfolder/basename" (命名冲突时)
        # 提取纯 basename 用于图片路径查找
        if "/" in img_name:
            lookup_basename = img_name.split("/")[-1]
        else:
            lookup_basename = img_name

        # 找这个图像的尺寸
        if img_name in size_cache:
            W, H = size_cache[img_name]
        elif lookup_basename in name2path:
            W, H = get_image_size(name2path[lookup_basename])
            size_cache[img_name] = (W, H)
        elif img_name in name2path:
            # 回退: 直接匹配完整 key (可能 name2path 存的是完整路径)
            W, H = get_image_size(name2path[img_name])
            size_cache[img_name] = (W, H)
        else:
            # 尝试模糊匹配: 用 basename 或完整 key 在路径中查找
            found = False
            for k, v in name2path.items():
                if lookup_basename in k or k in lookup_basename or img_name in k:
                    W, H = get_image_size(v)
                    size_cache[img_name] = (W, H)
                    found = True
                    break
            if not found:
                missing_size += 1
                continue

        if W is None:
            missing_img += 1
            continue

        segs = info.get("segments", [])
        feats = []
        for x, y, w, h in segs:
            cx = x + w / 2.0
            cy = y + h / 2.0
            area = w * h
            feats.append([
                cx / W,          # 归一化中心 X
                cy / H,          # 归一化中心 Y
                w / W,           # 归一化宽度
                h / H,           # 归一化高度
                area / (W * H),  # 归一化面积
            ])
        features[img_name] = feats

    print(f"  提取完成: {len(features)} 张图像的位置特征")
    print(f"  缺失原始图像: {missing_img}, 缺尺寸: {missing_size}")

    # 统计
    all_feats = np.concatenate([np.array(v) for v in features.values()], axis=0)
    print(f"  总 segment 数: {all_feats.shape[0]}")
    print(f"  特征范围:")
    print(f"    cx_n:   [{all_feats[:,0].min():.4f}, {all_feats[:,0].max():.4f}]")
    print(f"    cy_n:   [{all_feats[:,1].min():.4f}, {all_feats[:,1].max():.4f}]")
    print(f"    w_n:    [{all_feats[:,2].min():.4f}, {all_feats[:,2].max():.4f}]")
    print(f"    h_n:    [{all_feats[:,3].min():.4f}, {all_feats[:,3].max():.4f}]")
    print(f"    area_n: [{all_feats[:,4].min():.6f}, {all_feats[:,4].max():.4f}]")

    return features


def main():
    parser = argparse.ArgumentParser(description="提取 SAM2 segment 归一化位置特征")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img_root", type=str, default=None,
                        help="图片根目录 (default: ./{dataset}_split/)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径 (default: ./data/Segments/Seg_embs/{dataset}_pos_features.npy)")
    args = parser.parse_args()

    img_root = args.img_root or f"./{args.dataset}_split/"
    features = extract_position_features(args.dataset, img_root)

    out = args.output or f"./data/Segments/Seg_embs/{args.dataset}_pos_features.npy"
    np.save(out, features)
    print(f"保存至: {out}")


if __name__ == "__main__":
    main()
