## conda env: clip_ex
# load packages
import torch,glob,tqdm,clip,argparse,os,numpy as np
import pandas as pd
from PIL import Image
from data.Segments.scripts.utilities.CLIP import compute_embeddings, compute_embeddings_batched

# ---------------------------------------------------------------------------
#  derm7pt 二分类标签工具
# ---------------------------------------------------------------------------

def _build_derm7pt_label_map():
    """从 meta.csv 建立 {文件名小写: 'benign'|'malignant'} 映射。"""
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
    print(f"derm7pt 二分类标签映射: {len(label_map)} 文件 "
          f"(benign={n_benign}, malignant={n_malignant})")
    return label_map


def _fix_derm7pt_labels(torch_path, label_map):
    """加载已生成的 .torch 文件，将标签修正为二分类，原地保存。"""
    print(f"  修正 {torch_path} 的标签为二分类...")
    data = torch.load(torch_path, weights_only=False, map_location="cpu")
    fixed = 0
    missing = 0
    for key in list(data.keys()):
        emb = data[key][0]
        key_lower = key.lower()
        if key_lower in label_map:
            data[key] = [emb, label_map[key_lower]]
            fixed += 1
        else:
            missing += 1
    torch.save(data, torch_path)
    print(f"    修正: {fixed}, 未匹配: {missing}")
    if missing > 0:
        print(f"    ⚠ {missing} 个文件在 meta.csv 中找不到标签!")
    return data


def load_dataset(dataset):
    if dataset == "CUB_200_2011":
        path = "CUB_200_2011"
    elif dataset == "derm7pt":
        path = "derm7pt"
    elif dataset == "BUS_UCLM":
        path = "BUS_UCLM"
    return path


def _load_bus_uclm_split():
    """Read BUS-UCLM CSV, return {split: [(img_path, label), ...]}"""
    import pandas as pd
    csv_path = "./dataset_all/BUS-UCLM/BUS-UCLM_labels.csv"
    df = pd.read_csv(csv_path)
    img_base = os.path.abspath("./dataset_all/BUS-UCLM/BUS-UCLM/images")
    splits = {}
    for split in ["train", "test"]:
        sub = df[df["split"] == split]
        paths = [(os.path.join(img_base, row["filename"]), row["label"])
                 for _, row in sub.iterrows()]
        splits[split] = paths
    return splits


def _load_isic2019_data():
    """读取 ISIC-2019 train/test GroundTruth CSV, 返回 {split: [(img_path, label), ...]}

    CSV 格式: image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK  (one-hot, 值为 1.0 的列即标签)
    测试集 CSV 末尾有额外的 score_weight, validation_weight 列, 不影响读取。
    """
    CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    splits = {}

    for split, csv_path, img_dir in [
        ('train', './dataset_all/ISIC-2019/train/ISIC_2019_Training_GroundTruth.csv',
         './dataset_all/ISIC-2019/train/ISIC_2019_Training_Input'),
        ('test', './dataset_all/ISIC-2019/test/ISIC_2019_Test_GroundTruth.csv',
         './dataset_all/ISIC-2019/test/ISIC_2019_Test_Input'),
    ]:
        if not os.path.exists(csv_path):
            print(f"  [ISIC-2019] 跳过 {split}: 找不到 {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        img_dir_abs = os.path.abspath(img_dir)
        items = []
        skipped = 0
        for _, row in df.iterrows():
            image_id = str(row['image']).strip()
            img_path = os.path.join(img_dir_abs, image_id + '.jpg')
            if not os.path.exists(img_path):
                skipped += 1
                continue
            # 找到 one-hot 中值为 1.0 的类别
            label = None
            for cls in CLASSES:
                if float(row[cls]) == 1.0:
                    label = cls
                    break
            if label is None:
                skipped += 1
                continue
            items.append((img_path, label))
        splits[split] = items
        print(f"  ISIC-2019 {split}: {len(items)} images" + (f" (跳过 {skipped})" if skipped else ""))

    label_set = sorted(set(lbl for _, lbl in splits.get('train', [])))
    print(f"  ISIC-2019 类别 ({len(label_set)}): {label_set}")
    return splits


def _embed_batch_with_labels(img_paths, labels, model, preprocess, device, batch_size=64):
    """批量嵌入全图并直接赋予指定标签, 返回 {filename: [embedding, label]}"""
    data = {}
    for i in tqdm.tqdm(range(0, len(img_paths), batch_size), desc="Embedding images"):
        batch_paths = img_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_imgs = []
        valid_labels = []
        valid_keys = []
        for p, lbl in zip(batch_paths, batch_labels):
            try:
                image = preprocess(Image.open(p).convert('RGB')).unsqueeze(0)
                batch_imgs.append(image)
                valid_labels.append(lbl)
                valid_keys.append(os.path.basename(p))
            except Exception as e:
                print(f"  ⚠ 跳过 {os.path.basename(p)}: {e}")
        if not batch_imgs:
            continue
        batch_tensor = torch.cat(batch_imgs).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch_tensor).cpu()
        for j, key in enumerate(valid_keys):
            data[key] = [embs[j], valid_labels[j]]
    return data


def main(dataset, emb, device):
    # load model
    device = torch.device(device)
    model, preprocess = clip.load(emb, device=device)
    model_name = f"CLIP-{emb}".replace("/", "")
    model.eval()
    model.to(device)

    # ── BUS-UCLM: 合并原始 train+test, 分层重切为 train/val/test (60:20:20) ──
    if dataset == "BUS_UCLM":
        import pandas as pd
        from sklearn.model_selection import train_test_split

        csv_path = "./dataset_all/BUS-UCLM/BUS-UCLM_labels.csv"
        df = pd.read_csv(csv_path)
        img_base = os.path.abspath("./dataset_all/BUS-UCLM/BUS-UCLM/images")
        out_dir = "./BUS_UCLM_split"
        os.makedirs(out_dir, exist_ok=True)

        # 收集所有 (path, label)
        all_items = [(os.path.join(img_base, row["filename"]), row["label"])
                     for _, row in df.iterrows()]
        all_labels = [lbl for _, lbl in all_items]

        label_set = sorted(set(all_labels))
        print(f"BUS-UCLM: {len(all_items)} images, {len(label_set)} classes: {label_set}")

        # 展示原始分布
        from collections import Counter
        orig_dist = Counter(all_labels)
        print(f"  全量分布: {dict(orig_dist)}")

        # Step 1: train+val (80%) vs test (20%), stratified
        trval_items, test_items, trval_labels, test_labels = train_test_split(
            all_items, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

        # Step 2: train (75% of trval = 60% total) vs val (25% of trval = 20% total)
        train_items, val_items = train_test_split(
            trval_items, test_size=0.25, random_state=42, stratify=trval_labels)

        for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            split_labels = [l for _, l in items]
            split_dist = Counter(split_labels)
            print(f"  {split_name}: {len(items)} samples {dict(split_dist)}")

            paths = [p for p, _ in items]
            labels = [l for _, l in items]

            data = {}
            for img_path, label in zip(paths, labels):
                try:
                    image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb_vec = model.encode_image(image).cpu()
                    key = os.path.basename(img_path)
                    data[key] = [emb_vec, label]
                except Exception as e:
                    print(f"    ⚠ 跳过 {os.path.basename(img_path)}: {e}")
            out_file = os.path.join(out_dir, f"images_BUS_UCLM_{split_name}_{model_name}.torch")
            torch.save(data, out_file)
            print(f"    → {out_file} ({len(data)} entries)")
        return

    # ── ISIC-2019: pre-split, read GroundTruth CSV, 9:1 train→val ──
    if dataset == "ISIC-2019":
        splits = _load_isic2019_data()
        out_dir = "./ISIC-2019_split"
        os.makedirs(out_dir, exist_ok=True)

        # 输出类别文件 (供后续训练步骤读取)
        train_labels = sorted(set(lbl for _, lbl in splits.get('train', [])))
        print(f"ISIC-2019: {len(train_labels)} classes: {train_labels}")
        class_file = "./data/classes/ISIC-2019_classes.txt"
        os.makedirs(os.path.dirname(class_file), exist_ok=True)
        with open(class_file, 'w') as f:
            for c in train_labels:
                f.write(f"{c}\n")
        print(f"  类别文件 → {class_file}")

        # 嵌入 test (原封不动)
        test_items = splits.get('test', [])
        test_paths = [p for p, _ in test_items]
        test_labels = [l for _, l in test_items]
        print(f"  test: {len(test_paths)} images")
        data_test = _embed_batch_with_labels(test_paths, test_labels, model, preprocess, device)
        out_test = os.path.join(out_dir, f"images_ISIC-2019_test_{model_name}.torch")
        torch.save(data_test, out_test)
        print(f"    → {out_test} ({len(data_test)} entries)")

        # 训练集 9:1 分为 train / val
        train_items = splits.get('train', [])
        np.random.seed(42)
        idx = np.random.permutation(len(train_items))
        split_n = int(len(train_items) * 0.9)
        train_sub = [train_items[i] for i in idx[:split_n]]
        val_sub   = [train_items[i] for i in idx[split_n:]]

        for split_name, items_sub in [("train", train_sub), ("val", val_sub)]:
            paths = [p for p, _ in items_sub]
            labels = [l for _, l in items_sub]
            print(f"  {split_name} (from train): {len(paths)} images")
            data = _embed_batch_with_labels(paths, labels, model, preprocess, device)
            out_file = os.path.join(out_dir, f"images_ISIC-2019_{split_name}_{model_name}.torch")
            torch.save(data, out_file)
            print(f"    → {out_file} ({len(data)} entries)")
        return

    # ── places365_part: 已预分割, 直接从 dataset_all 嵌入全图 ──
    #  注意: places365_part 各类下图片为纯数字命名 (00000001.jpg),
    #  不同类之间存在 basename 冲突, 因此 .torch 文件 key 使用
    #  "{class_name}/{basename}" 格式以避免覆盖丢失数据.
    if dataset == "places365_part":
        img_root = "./places365_part_split"
        out_dir = "./places365_part_split"
        os.makedirs(out_dir, exist_ok=True)

        # 读取类别列表：优先从 categories_places365.txt，否则从 split 目录推导
        categories_path = os.path.join(img_root, "categories_places365.txt")
        if os.path.exists(categories_path):
            classes = []
            with open(categories_path, 'r') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        classes.append(name)
        else:
            # Fallback: derive class list from train/ subdirectories
            train_dir = os.path.join(img_root, "train")
            if not os.path.isdir(train_dir):
                raise FileNotFoundError(
                    f"找不到类别文件: {categories_path}，且 {train_dir} 不存在")
            classes = sorted(
                d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            )
            print(f"  从 {train_dir} 推导出 {len(classes)} 个类别")
        print(f"places365_part: {len(classes)} classes")

        # 写入类别文件供后续训练步骤读取
        class_file = "./data/classes/places365_part_classes.txt"
        os.makedirs(os.path.dirname(class_file), exist_ok=True)
        with open(class_file, 'w') as f:
            for c in sorted(classes):
                f.write(f"{c}\n")
        print(f"  类别文件 → {class_file}")

        # 嵌入 train / val / test
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(img_root, split)
            # 收集 (full_path, class_name, unique_key) 三元组
            img_items = []  # [(path, class_name, "{class}/{basename}")]
            for cls in classes:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_items.append((
                                os.path.join(cls_dir, fname),
                                cls,
                                f"{cls}/{fname}"  # 唯一 key, 避免跨类 basename 冲突
                            ))
            if not img_items:
                print(f"  ⚠ {split}: 没有找到图片, 跳过")
                continue
            print(f"  {split}: {len(img_items)} images")

            # 使用带自定义 key 的嵌入 (不用 compute_embeddings_batched)
            out_file = os.path.join(out_dir, f"images_places365_part_{split}_{model_name}.torch")
            data = {}
            batch_size = 64
            for i in tqdm.tqdm(range(0, len(img_items), batch_size), desc=f"  Embedding {split}"):
                batch = img_items[i:i + batch_size]
                batch_imgs = []
                batch_keys = []
                batch_labels = []
                for path, label, key in batch:
                    try:
                        img = preprocess(Image.open(path).convert('RGB')).unsqueeze(0)
                        batch_imgs.append(img)
                        batch_keys.append(key)
                        batch_labels.append(label)
                    except Exception as e:
                        print(f"  ⚠ 跳过 {os.path.basename(path)}: {e}")
                if not batch_imgs:
                    continue
                batch_tensor = torch.cat(batch_imgs).to(device)
                with torch.no_grad():
                    embs = model.encode_image(batch_tensor).cpu()
                for j, key in enumerate(batch_keys):
                    data[key] = [embs[j], batch_labels[j]]
            torch.save(data, out_file)
            print(f"    → {out_file} ({len(data)} entries)")
        return

    # ── CUB / derm7pt: original flow ──
    dataset_path = load_dataset(dataset)
    split_dir = f"{dataset_path}_split"

    # derm7pt: 构建二分类标签映射
    label_map = None
    if dataset == "derm7pt":
        label_map = _build_derm7pt_label_map()

    ## train set ##
    train_list = glob.glob(os.path.join(split_dir, "train", "*", "*"))
    output_train = os.path.join(split_dir, f"images_{dataset}_train_{model_name}.torch")
    compute_embeddings_batched(train_list, model, preprocess, output_train, device)
    if label_map is not None:
        _fix_derm7pt_labels(output_train, label_map)
    output_train = None

    ## val set ##
    val_list = glob.glob(os.path.join(split_dir, "val", "*", "*"))
    output_val = os.path.join(split_dir, f"images_{dataset}_val_{model_name}.torch")
    compute_embeddings_batched(val_list, model, preprocess, output_val, device)
    if label_map is not None:
        _fix_derm7pt_labels(output_val, label_map)
    output_val = None

    ## test set ##
    test_list = glob.glob(os.path.join(split_dir, "test", "*", "*"))
    print(len(test_list))
    output_test = os.path.join(split_dir, f"images_{dataset}_test_{model_name}.torch")
    compute_embeddings_batched(test_list, model, preprocess, output_test, device)
    if label_map is not None:
        _fix_derm7pt_labels(output_test, label_map)
    output_test = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--emb', type=str, help='embedding model')
    parser.add_argument('--device', type=str, help='device')
    args = parser.parse_args()
    main(args.dataset, args.emb, args.device)
    # --dataset CUB_200_2011 --emb ViT-L/14 --device cuda:0
    # --dataset derm7pt --emb ViT-L/14 --device cuda:0
    # --dataset BUS_UCLM --emb ViT-L/14 --device cuda:0
    # --dataset ISIC-2019 --emb ViT-L/14 --device cuda:0
    # --dataset places365_part --emb ViT-L/14 --device cuda:0