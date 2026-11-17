"""DCBM全图上计算激活值，这个文件是部件计算激活值，为什么不行？
DCBM-Local — CUB_200_2011

与原 DCBM 的唯一区别：激活值由「部件→原型」计算，而非「全图→原型」。

原 DCBM:
    activation = softmax( full_image_embedding @ prototypes.T / ||prototypes||² )
    → 每张图一个 CLIP 嵌入向量，直接和所有原型算相似度

DCBM-Local:
    activation[k] = max_{segment in image} softmax( segment_emb @ prototypes.T / ||prototypes||² )
    → 每张图有多个部件（segment），每个部件和原型算相似度，取最大值

聚类、标签 MLP 训练等其余步骤与原 DCBM 完全一致。
模型调用 utils/dcbm.py 的 CBM 类，不改动任何 utils 文件。

用法：
    python dcbm_seglocal_cub.py
"""

import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from utils.dcbm import CBM

# ---------------------------------------------------------------------------
# 路径 & 超参数（与原 main_train_DCBM_cub.py / main_dcbm_5.py 对齐）
# ---------------------------------------------------------------------------

embed_path = "./CUB_200_2011_split/"
dataset = "CUB_200_2011"
class_labels_path = "./data/classes/cub_classes.txt"
segment_path = "./data/Segments/Seg_embs/"

model_name = "CLIP-ViT-L14"
segmentation_technique = "SAM2"
device = "cuda:0"

# --- 聚类 ---
cluster_method = "kmeans"
centroid_method = "median"
clusters = 2048 # 2048
concept_per_class = 50
crop = False

# --- 训练 ---
epochs = 400
lambda_1 = 1e-4
lr = 5e-4
batch_size = 32
one_hot = False

# ---------------------------------------------------------------------------
# 工具函数：段键 → 图像键 映射
# ---------------------------------------------------------------------------

def extract_image_key(seg_key, image_keys_set):
    """从段键提取原始图像名，在 image_keys_set 中匹配。

    段键格式: Black_Footed_Albatross_0001_796111_OUT_CROP_0_X_.jpg
    图像键:   Black_Footed_Albatross_0001_796111.jpg
    """
    for marker in ['_OUT_CROP_', '_OUT_SHAPE_', '_OUT_MASK_']:
        if marker in seg_key:
            basename = seg_key.split(marker)[0]
            break
    else:
        return None
    for ext in ['.jpg', '.JPG', '.JPEG', '.jpeg', '.png', '.PNG']:
        candidate = basename + ext
        if candidate in image_keys_set:
            return candidate
    if basename in image_keys_set:
        return basename
    return None


def build_segment_image_map(segment_names, data_train_raw, data_val_raw,
                            data_test_raw):
    """为每个段找到它所属的图像键。返回 list[str|None]，长度 = M."""
    all_image_keys = set()
    for raw_dict in [data_train_raw, data_val_raw, data_test_raw]:
        if raw_dict is not None:
            all_image_keys.update(raw_dict.keys())

    mapping = []
    unmatched = 0
    for seg_key in segment_names:
        img_key = extract_image_key(seg_key, all_image_keys)
        mapping.append(img_key)
        if img_key is None:
            unmatched += 1

    print(f"  段→图像映射: {len(mapping)} 个段, {unmatched} 个未能匹配")
    return mapping


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DCBM-Local — CUB_200_2011（部件级激活）")
    print(f"  K={clusters}, lambda_1={lambda_1}, lr={lr}")
    print("=" * 60)

    # ---- 1. 初始化 CBM（自动加载图像嵌入、标签）----
    print("\n[1/5] 初始化 CBM...")
    cbm = CBM(
        embeddings_path=embed_path,
        dataset=dataset,
        model=model_name,
        class_labels_path=class_labels_path,
        device=device,
    )

    # ---- 2. 加载部件嵌入（与原 DCBM 一致）----
    print("\n[2/5] 加载部件嵌入 (segcrop)...")
    cbm.load_concepts(
        segment_path=segment_path,
        segmentation_technique=segmentation_technique,
        concept_name=None,
        selected_image_concepts=None,
        concept_per_class=concept_per_class,
        crop=crop,
    )
    # cbm.image_segments     : [M, 768]  部件 CLIP 嵌入
    # cbm.image_segments_names : list[str]  部件键
    # ⚠ 保存扁平副本！聚类后 image_segments_names 会被覆盖为 dict
    flat_seg_names = list(cbm.image_segments_names)

    # ---- 3. 聚类 → 原型向量（与原 DCBM 一致）----
    print("\n[3/5] 聚类...")
    if clusters is not None:
        cbm.cluster_image_concepts(
            clustering_method=cluster_method,
            n_clusters=clusters,
            pca=True,
        )
        cbm.centroid_concepts(centroid_method=centroid_method)
    else:
        cbm.clustered_concepts = cbm.image_segments.copy()
        print("跳过聚类，直接使用原始部件作为概念原型。")

    # cbm.clustered_concepts : [K, 768]  概念原型向量

    # ---- 4. 部件级激活计算（替代原 preprocess_data）----
    print("\n[4/5] 部件级激活计算 (GPU)...")

    concepts = cbm.clustered_concepts  # [K, 768]
    seg_embs = cbm.image_segments       # [M, 768]
    K = concepts.shape[0]
    M = seg_embs.shape[0]

    # 4a. 每段 → 每原型：softmax 激活（与原 DCBM preprocess_module 公式一致）
    print(f"  计算 {M} 个部件 → {K} 个原型的 softmax 激活 (GPU)...")

    concepts_t = torch.tensor(concepts, dtype=torch.float32, device=device)   # [K, 768]
    norms_c2_t = torch.norm(concepts_t, dim=1, keepdim=True).pow(2)           # [K, 1]

    GPU_BATCH = 4096
    seg_act_parts = []
    for start in range(0, M, GPU_BATCH):
        end = min(start + GPU_BATCH, M)
        batch_seg = torch.tensor(seg_embs[start:end], dtype=torch.float32, device=device)
        # softmax( seg @ C.T / ||C||² )
        raw = torch.matmul(batch_seg, concepts_t.T) / norms_c2_t.T         # [B, K]
        raw_max = raw.max(dim=1, keepdim=True).values
        raw_exp = torch.exp(raw - raw_max)
        softmax_batch = raw_exp / raw_exp.sum(dim=1, keepdim=True)          # [B, K]
        seg_act_parts.append(softmax_batch.cpu().numpy())
        if (end % (GPU_BATCH * 4) == 0) or end == M:
            print(f"    GPU softmax: {end}/{M} 部件完成")

    seg_act = np.concatenate(seg_act_parts, axis=0).astype(np.float32)     # [M, K]
    del concepts_t, norms_c2_t, seg_act_parts
    torch.cuda.empty_cache()

    # 4b. 段→图像 映射
    print("  构建段→图像映射...")
    seg_to_img = build_segment_image_map(
        flat_seg_names,
        cbm.data_train_raw,
        cbm.data_val_raw,
        cbm.data_test_raw,
    )

    # 4c. 反向索引：图像键 → 段索引列表
    img_to_seg = {}
    for seg_idx, img_key in enumerate(seg_to_img):
        if img_key is not None:
            img_to_seg.setdefault(img_key, []).append(seg_idx)

    # 4d. 对每个 split 做 max-pool（GPU 加速）
    def max_pool_for_split(raw_dict, split_name):
        """返回 [N, K] 的每图激活矩阵（GPU max-pool）。"""
        if raw_dict is None:
            return None
        image_keys = list(raw_dict.keys())
        N_img = len(image_keys)

        all_img_seg = []
        for i, img_key in enumerate(image_keys):
            indices = img_to_seg.get(img_key, [])
            if len(indices) > 0:
                all_img_seg.append((i, indices))

        act = np.zeros((N_img, K), dtype=np.float32)
        if len(all_img_seg) == 0:
            print(f"  [{split_name}] 0/{N_img} 张图有部件!")
            return act

        POOL_BATCH = 256
        for b_start in range(0, len(all_img_seg), POOL_BATCH):
            b_end = min(b_start + POOL_BATCH, len(all_img_seg))
            batch_items = all_img_seg[b_start:b_end]

            flat_indices = []
            seg_counts = []
            for _, seg_idx_list in batch_items:
                flat_indices.extend(seg_idx_list)
                seg_counts.append(len(seg_idx_list))

            seg_pool_t = torch.tensor(seg_act[flat_indices], dtype=torch.float32, device=device)
            max_vals = []
            offset = 0
            for cnt in seg_counts:
                if cnt > 0:
                    max_vals.append(seg_pool_t[offset:offset + cnt].max(dim=0).values)
                else:
                    max_vals.append(torch.zeros(K, device=device))
                offset += cnt
            max_t = torch.stack(max_vals)  # [batch, K]

            for j, (img_idx, _) in enumerate(batch_items):
                act[img_idx] = max_t[j].cpu().numpy()

        skipped = N_img - len(all_img_seg)
        if skipped > 0:
            print(f"  [{split_name}] {skipped}/{N_img} 张图无部件 (激活=0)")
        return act

    X_train_local = max_pool_for_split(cbm.data_train_raw, "train")
    X_val_local   = max_pool_for_split(cbm.data_val_raw,   "val")
    X_test_local  = max_pool_for_split(cbm.data_test_raw,  "test")

    # 4e. StandardScaler（与原 DCBM 一致）
    print("  StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train_local)
    cbm.scaler = scaler

    cbm.X_train = scaler.transform(X_train_local).astype(np.float32)
    cbm.X_val   = scaler.transform(X_val_local).astype(np.float32) if X_val_local is not None else None
    cbm.X_test  = scaler.transform(X_test_local).astype(np.float32) if X_test_local is not None else None

    # Labels（与原 DCBM 一致）
    if one_hot:
        cbm.y_train = cbm.train_labels_int
        cbm.y_val   = cbm.val_labels_int
        cbm.y_test  = cbm.test_labels_int
    else:
        cbm.y_train = cbm.train_labels_one_hot
        cbm.y_val   = cbm.val_labels_one_hot
        cbm.y_test  = cbm.test_labels_one_hot

    print(f"  train: {cbm.X_train.shape}, val: {cbm.X_val.shape}, "
          f"test: {cbm.X_test.shape}")

    # ---- 5. 训练 Label MLP（与原 DCBM 完全一致）----
    print("\n[5/5] 训练 Label MLP...")
    cbm.train(
        lambda_1=lambda_1,
        lr=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        device=device,
        project="DCBM_Local_CUB",
        to_print=True,
        early_stopping_patience=None,
        one_hot=one_hot,
        use_wandb=False,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
