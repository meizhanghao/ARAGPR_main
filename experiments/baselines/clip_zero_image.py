"""
CLIP 零样本分类 — 支持 derm7pt / CUB_200_2011 / BUS_UCLM / ISIC-2019 / places365_part / places365

用法:
    python -m experiments.baselines.clip_zero_image --dataset derm7pt --model ViT-L/14 --prompt medical
    python -m experiments.baselines.clip_zero_image --dataset BUS_UCLM --model ViT-L/14 --prompt ultrasound
    python -m experiments.baselines.clip_zero_image --dataset CUB_200_2011 --model ViT-L/14 --prompt generic --dataset_path ./dataset_all/CUB_200_2011
    python -m experiments.baselines.clip_zero_image --dataset ISIC-2019 --model ViT-L/14 --prompt isic
    python -m experiments.baselines.clip_zero_image --dataset ISIC-2019 --model ViT-L/14 --prompt isic --include_unk
    python -m experiments.baselines.clip_zero_image --dataset places365_part --model ViT-L/14 --prompt places
    python -m experiments.baselines.clip_zero_image --dataset places365 --model ViT-L/14 --prompt places --max_per_class 50
"""

import torch
import clip
import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path


# ======================== 配置 ========================

AVAILABLE_MODELS = [
    "ViT-L/14", "ViT-L/14@336px",
    "ViT-B/32", "ViT-B/16",
    "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
]

PROMPT_TEMPLATES = {
    "generic": [
        "a photo of a {}.",
        "a picture of a {}.",
        "an image of a {}.",
        "a close-up of a {}.",
    ],
    "medical": [
        "a dermoscopic image of {}.",
        "a dermatoscope photo showing {}.",
        "a skin lesion diagnosed as {}.",
        "dermoscopy of {}.",
        "a clinical photograph of {}.",
    ],
    "ultrasound": [
        "an ultrasound image of {}.",
        "a breast ultrasound showing {}.",
        "an ultrasonography scan of {}.",
        "a sonogram of {}.",
        "ultrasound imaging of {}.",
    ],
    "simple": [
        "{}.",
    ],
    "scene": [
        "a photo of a {}.",
        "a photo of the {}.",
        "a picture of a {} scene.",
        "a picture of the {} scene.",
        "a view of a {}.",
        "an image taken in a {}.",
        "this is a {}.",
        "this is a photo of a {}.",
    ],
    "places": [
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a {}, a type of place.",
        "a photo of the {}, a type of place.",
        "a photo of a {}, a type of scene.",
        "a photo of the {}, a type of scene.",
        "a picture of a {}.",
        "a picture of the {}.",
        "a {} scene.",
        "this is a photo of a {}.",
    ],
    "isic": [
        "a dermoscopic image of {}.",
        "a dermoscopy image of {}.",
        "a skin lesion diagnosed as {}.",
        "a clinical image of {}.",
    ],
}

# derm7pt 二分类描述短语（prompt ensembling）
DERM7PT_CLASS_DESCRIPTIONS = {
    "benign": [
        "a benign skin lesion",
        "a benign mole or nevus",
        "a harmless skin growth",
        "a common mole on the skin",
    ],
    "malignant": [
        "a malignant skin cancer",
        "a melanoma",
        "a dangerous skin tumor",
        "a skin cancer that needs treatment",
    ],
}


BUS_UCLM_CLASS_DESCRIPTIONS = {
    "benign": [
        "a benign breast tumor on ultrasound",
        "a benign breast mass",
        "a non-cancerous breast lesion",
    ],
    "malignant": [
        "a malignant breast cancer on ultrasound",
        "a breast carcinoma",
        "a cancerous breast tumor",
    ],
    "normal": [
        "normal breast tissue on ultrasound",
        "healthy breast ultrasound",
        "normal breast anatomy",
    ],
}

ISIC2019_CLASS_DESCRIPTIONS = {
    "MEL": ["melanoma"],
    "NV": ["melanocytic nevus"],
    "BCC": ["basal cell carcinoma"],
    "AK": ["actinic keratosis"],
    "BKL": ["benign keratosis-like lesion"],
    "DF": ["dermatofibroma"],
    "VASC": ["vascular lesion"],
    "SCC": ["squamous cell carcinoma"],
    "UNK": ["unclassified skin lesion"],
}


# ======================== 数据加载 ========================

def _to_binary(diagnosis):
    d = str(diagnosis).lower()
    if 'melanoma' in d or 'basal cell carcinoma' in d:
        return 'malignant'
    return 'benign'


def load_derm7pt_data():
    """derm7pt 测试集：二分类标签。"""
    root = "./dataset_all/derm7pt"
    meta = pd.read_csv(f"{root}/meta/meta.csv")
    test_idx = pd.read_csv(f"{root}/meta/test_indexes.csv")
    col = test_idx.columns[0]
    test_cases = set(test_idx[col].values.flatten())
    test_meta = meta[meta['case_num'].isin(test_cases)]
    images_dir = f"{root}/images"

    image_paths, labels = [], []
    for _, row in test_meta.iterrows():
        binary_label = _to_binary(row['diagnosis'])
        for col_name in ['clinic', 'derm']:
            rel_path = row[col_name]
            if pd.notna(rel_path):
                full_path = os.path.join(images_dir, rel_path)
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                    labels.append(binary_label)

    class_names = ["benign", "malignant"]
    class_descriptions = DERM7PT_CLASS_DESCRIPTIONS
    print(f"derm7pt 测试集: {len(image_paths)} 张图片, 二分类")
    print(f"  benign: {labels.count('benign')}, malignant: {labels.count('malignant')}")
    return image_paths, labels, class_names, class_descriptions


def load_cub_data(dataset_path):
    """CUB_200_2011 全部图片：零样本分类（无需 train/test 划分）。

    dataset_path 示例: ./dataset_all/CUB_200_2011
    结构: {dataset_path}/images/{class_folder}/*.jpg
          {dataset_path}/images.txt  (id path 格式)
    """
    images_txt = os.path.join(dataset_path, "images.txt")
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_txt):
        raise FileNotFoundError(f"找不到 images.txt: {images_txt}")

    # 读取 images.txt（格式: "1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"）
    with open(images_txt, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 收集所有图片路径，同时记录类文件夹名
    image_paths, class_folder_names = [], []
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        rel_path = parts[1]  # "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
        full_path = os.path.join(images_dir, rel_path)
        if os.path.exists(full_path):
            image_paths.append(full_path)
            class_folder_names.append(os.path.dirname(rel_path))  # "001.Black_footed_Albatross"

    # 去重得到有序类名列表
    unique_folders = sorted(set(class_folder_names))
    # "001.Black_footed_Albatross" → "Black footed Albatross"
    class_names = []
    folder_to_cls = {}
    for folder in unique_folders:
        parts = folder.split(".", 1)
        clean = parts[1].replace("_", " ") if len(parts) == 2 else folder.replace("_", " ")
        class_names.append(clean)
        folder_to_cls[folder] = clean

    labels = [folder_to_cls[f] for f in class_folder_names]
    class_descriptions = {cn: [cn] for cn in class_names}

    print(f"CUB_200_2011: {len(image_paths)} 张图片, {len(class_names)} 类")
    return image_paths, labels, class_names, class_descriptions


def load_bus_uclm_data():
    """BUS-UCLM 全量数据：3分类 (benign/malignant/normal)"""
    csv_path = "./dataset_all/BUS-UCLM/BUS-UCLM_labels.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run BUS_UCLM_csv.py first. Missing: {csv_path}")
    df = pd.read_csv(csv_path)
    img_dir = os.path.abspath("./dataset_all/BUS-UCLM/BUS-UCLM/images")

    image_paths, labels = [], []
    for _, row in df.iterrows():
        full_path = os.path.join(img_dir, row["filename"])
        if os.path.exists(full_path):
            image_paths.append(full_path)
            labels.append(row["label"])

    class_names = sorted(df["label"].unique())
    class_descriptions = BUS_UCLM_CLASS_DESCRIPTIONS
    print(f"BUS-UCLM 全量: {len(image_paths)} 张图片, {len(class_names)} 类: {class_names}")
    for c in class_names:
        print(f"  {c}: {labels.count(c)}")
    return image_paths, labels, class_names, class_descriptions


def load_isic2019_data(include_unk=False):
    """ISIC-2019 测试集：8 分类 (+ 可选 UNK 兜底类)。

    从 GroundTruth CSV 读取 one-hot 标签, 只保留测试集。
    include_unk=True 时将 UNK 作为第 9 类 (泛指"未分类皮肤病变")。
    UNK 是异质集合体 (瘢痕/囊肿/感染/正常皮肤), 不是单一视觉概念,
    零样本准确率通常显著低于 8 个已知类, 但可避免丢弃 ~25% 数据。
    """
    CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    if include_unk:
        CLASSES = CLASSES + ['UNK']
    csv_path = './dataset_all/ISIC-2019/test/ISIC_2019_Test_GroundTruth.csv'
    img_dir = os.path.abspath('./dataset_all/ISIC-2019/test/ISIC_2019_Test_Input')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 GroundTruth CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    image_paths, labels = [], []
    skipped_unk = 0

    for _, row in df.iterrows():
        image_id = str(row['image']).strip()
        img_path = os.path.join(img_dir, image_id + '.jpg')
        if not os.path.exists(img_path):
            continue
        # 从 one-hot 列解析标签
        label = None
        is_unk = False
        for cls in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']:
            if float(row.get(cls, 0)) == 1.0:
                if cls == 'UNK':
                    is_unk = True
                    if include_unk:
                        label = 'UNK'
                else:
                    label = cls
                break
        if label is None:
            if is_unk:
                skipped_unk += 1
            continue
        image_paths.append(img_path)
        labels.append(label)

    class_names = CLASSES
    class_descriptions = ISIC2019_CLASS_DESCRIPTIONS
    unk_tag = " 9类(含UNK)" if include_unk else " 8类(排除UNK)"
    print(f"ISIC-2019 测试集: {len(image_paths)} 张图片, {len(class_names)} 类{unk_tag}: {class_names}")
    for c in class_names:
        print(f"  {c}: {labels.count(c)}")
    if skipped_unk:
        print(f"  UNK (排除): {skipped_unk} 张")
    return image_paths, labels, class_names, class_descriptions


def load_places365_part_data():
    """places365_part 测试集：365 类场景识别（含多级类如 apartment_building/outdoor）。

    类别列表从 categories_places365.txt 读取，图片位于 test/{class_name}/ 下。
    由于类别名已是自然语言描述 (如 "airfield", "bazaar/outdoor")，
    使用 scene 提示模板 + 类别名本身作为描述句进行零样本分类。
    """
    root = "./dataset_all/places365_part"
    categories_path = os.path.join(root, "categories_places365.txt")
    if not os.path.exists(categories_path):
        raise FileNotFoundError(f"找不到类别文件: {categories_path}")

    classes = []
    with open(categories_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)

    test_dir = os.path.join(root, "test")
    image_paths, labels = [], []
    missing = 0
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            missing += 1
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)  # 完整类名, 含可能的多级路径

    # 对多级类名 (如 "apartment_building/outdoor")，将 "/" 和 "_" 替换为空格作为自然语言描述词
    class_descriptions = {}
    for cls in classes:
        desc = cls.replace('/', ' ').replace('_', ' ')
        class_descriptions[cls] = [desc]

    print(f"places365_part 测试集: {len(image_paths)} 张图片, {len(classes)} 类")
    if missing:
        print(f"  ⚠ {missing} 个类目录缺失")
    return image_paths, labels, classes, class_descriptions


def load_places365_data(max_per_class=None):
    """places365 完整版 — 364 类场景识别，使用 train 集做零样本评估。

    test_val 目录仅含 test_256/val_256 (无标签)，因此以 train 集作为评估集。
    类别名映射自 data/classes/categories_places365_clean.txt（人类可读格式）。

    Args:
        max_per_class: 每类最多加载图片数 (None=全部 ~1.8M 张, 建议 50-100 快速验证)
    """
    root = "./dataset_all/Places365"
    categories_path = os.path.join(root, "categories_places365.txt")
    clean_path = "./data/classes/categories_places365_clean.txt"

    if not os.path.exists(categories_path):
        raise FileNotFoundError(f"找不到类别文件: {categories_path}")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"找不到 clean 类别文件: {clean_path}")

    # 读取人类可读类别名 (364 个, 如 "airplane cabin", "atrium, public")
    with open(clean_path, 'r') as f:
        clean_names = [line.strip() for line in f if line.strip()]

    train_base = os.path.join(root, "train")
    image_paths, labels = [], []
    used_clean = set()

    with open(categories_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path = parts[0].lstrip('/')  # "a/airfield"
            cls_idx = int(parts[1])
            class_dir = os.path.join(train_base, rel_path)
            if not os.path.isdir(class_dir):
                continue
            # 用索引映射到人类可读类别名
            clean_name = clean_names[cls_idx] if cls_idx < len(clean_names) else rel_path
            used_clean.add(clean_name)
            fnames = [fn for fn in os.listdir(class_dir)
                      if fn.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if max_per_class is not None:
                fnames = sorted(fnames)[:max_per_class]
            for fname in fnames:
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(clean_name)

    class_names = sorted(used_clean)
    class_descriptions = {cn: [cn] for cn in class_names}

    limit_info = f", 每类上限 {max_per_class}" if max_per_class else ""
    print(f"places365 完整版 train 集: {len(image_paths)} 张图片, "
          f"{len(class_names)} 类{limit_info}")
    return image_paths, labels, class_names, class_descriptions


# ======================== 零样本分类 ========================

@torch.no_grad()
def zero_shot_classify(model, transform, image_paths, class_names, class_descriptions,
                       prompt_template, device):
    text_features = []
    for cls_name in class_names:
        cls_features = []
        for desc in class_descriptions[cls_name]:
            prompts = [t.format(desc) for t in prompt_template]
            text_tokens = clip.tokenize(prompts).to(device)
            cls_text_feat = model.encode_text(text_tokens)
            cls_text_feat = cls_text_feat / cls_text_feat.norm(dim=-1, keepdim=True)
            cls_features.append(cls_text_feat.mean(dim=0))
        text_features.append(torch.stack(cls_features).mean(dim=0))
    text_features = torch.stack(text_features)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    predictions = []
    for img_path in tqdm(image_paths, desc="零样本分类"):
        image = Image.open(img_path).convert("RGB")
        image_input = transform(image).unsqueeze(0).to(device)
        image_feat = model.encode_image(image_input)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        similarity = (image_feat @ text_features.T).squeeze(0)
        pred_idx = similarity.argmax().item()
        predictions.append(class_names[pred_idx])

    return predictions


# ======================== 评估 ========================

def evaluate(predictions, ground_truth, class_names):
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    overall_acc = correct / len(ground_truth) * 100

    per_class = {}
    for cls in class_names:
        mask = [g == cls for g in ground_truth]
        total_cls = sum(mask)
        if total_cls > 0:
            correct_cls = sum(p == g for p, g, m in zip(predictions, ground_truth, mask) if m)
            per_class[cls] = (correct_cls / total_cls * 100, total_cls)

    return overall_acc, per_class


# ======================== 主函数 ========================
3
def main():
    parser = argparse.ArgumentParser(description="CLIP 零样本分类")
    parser.add_argument("--dataset", type=str, default="derm7pt",
                        choices=["derm7pt", "CUB_200_2011", "BUS_UCLM", "ISIC-2019",
                                 "places365_part", "places365"],
                        help="数据集 (默认 derm7pt)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="原始数据集路径 (CUB必填, e.g. ./dataset_all/CUB_200_2011)")
    parser.add_argument("--model", type=str, default="ViT-L/14",
                        help=f"CLIP 模型 (默认 ViT-L/14)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备 (默认 cuda:0)")
    parser.add_argument("--prompt", type=str, default="generic",
                        choices=["generic", "medical", "ultrasound", "simple",
                                 "scene", "places", "isic"],
                        help="提示词模板集 (derm7pt→medical, BUS_UCLM→ultrasound, "
                             "CUB→generic, places365→places, ISIC-2019→isic)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="places365 完整版: 每类最多加载图片数 (默认全部 ~1.8M, "
                             "建议 50-100 快速验证)")
    parser.add_argument("--include_unk", action="store_true",
                        help="ISIC-2019: 将 UNK 作为第 9 类 (默认排除, "
                             "因为 UNK 是异质集合体, 非单一视觉概念)")
    parser.add_argument("--list_models", action="store_true",
                        help="列出所有可用模型")
    args = parser.parse_args()

    if args.list_models:
        print("可用 CLIP 模型:")
        for m in AVAILABLE_MODELS:
            print(f"  {m}")
        return

    # 1. 加载数据
    print(f"\n{'='*60}")
    if args.dataset == "derm7pt":
        print(f"数据集: derm7pt (二分类: benign/malignant)")
        image_paths, labels, class_names, class_descriptions = load_derm7pt_data()
        if args.prompt == "generic":
            args.prompt = "medical"  # derm7pt 默认用 medical
    elif args.dataset == "BUS_UCLM":
        print(f"数据集: BUS_UCLM (三分类: benign/malignant/normal)")
        image_paths, labels, class_names, class_descriptions = load_bus_uclm_data()
        if args.prompt not in ("ultrasound",):
            args.prompt = "ultrasound"  # BUS-UCLM 默认用超声模板
    elif args.dataset == "ISIC-2019":
        unk_info = "9类(含UNK)" if args.include_unk else "8类(排除UNK)"
        print(f"数据集: ISIC-2019 ({unk_info}皮肤镜)")
        image_paths, labels, class_names, class_descriptions = \
            load_isic2019_data(include_unk=args.include_unk)
        if args.prompt not in ("isic", "medical"):
            args.prompt = "isic"  # ISIC-2019 默认用 isic 皮肤镜专用模板
    elif args.dataset == "places365_part":
        print(f"数据集: places365_part (365类场景识别)")
        image_paths, labels, class_names, class_descriptions = load_places365_part_data()
        if args.prompt not in ("places", "scene"):
            args.prompt = "places"  # places365 默认用 places 模板
    elif args.dataset == "places365":
        max_pc = args.max_per_class
        print(f"数据集: places365 完整版 (365类场景识别"
              f"{', 每类上限 ' + str(max_pc) if max_pc else ', 全量 ~1.8M'})")
        image_paths, labels, class_names, class_descriptions = \
            load_places365_data(max_per_class=max_pc)
        if args.prompt not in ("places", "scene"):
            args.prompt = "places"  # places365 默认用 places 模板
    else:
        if not args.dataset_path:
            raise ValueError("CUB 需要 --dataset_path 参数，例如 "
                             "./dataset_all/CUB_200_2011")
        print(f"数据集: CUB_200_2011 (200类细粒度)")
        image_paths, labels, class_names, class_descriptions = load_cub_data(args.dataset_path)
    print(f"{'='*60}")

    # 2. 加载 CLIP
    print(f"\n加载 CLIP 模型: {args.model}")
    model, preprocess = clip.load(args.model, device=args.device)
    model.eval()
    print(f"  设备: {args.device}")

    # 3. 提示词
    prompt_template = PROMPT_TEMPLATES[args.prompt]
    print(f"  提示词模板: {args.prompt} ({len(prompt_template)} 条)")
    if len(class_names) <= 5:
        for cls_name in class_names:
            print(f"    [{cls_name}] {class_descriptions[cls_name][0]}")

    # 4. 零样本分类
    print(f"\n开始零样本分类 ({len(image_paths)} 张图片)...")
    predictions = zero_shot_classify(
        model, preprocess, image_paths, class_names, class_descriptions,
        prompt_template, args.device
    )

    # 5. 评估
    overall_acc, per_class = evaluate(predictions, labels, class_names)

    print(f"\n{'='*60}")
    print(f"零样本分类结果")
    print(f"{'='*60}")
    print(f"模型:  {args.model}")
    print(f"数据集: {args.dataset} ({len(class_names)} 类)")
    print(f"提示词: {args.prompt}")
    print(f"整体准确率: {overall_acc:.2f}%  ({sum(p == g for p, g in zip(predictions, labels))}/{len(labels)})")

    # 预测类别分布
    from collections import Counter
    pred_count = Counter(predictions)
    print(f"\n预测类别分布:")
    for cls in class_names:
        bar = "█" * min(pred_count[cls] // max(1, len(labels) // 40), 40)
        print(f"  {cls:<8s}: {pred_count[cls]:5d}  {bar}")

    if len(class_names) <= 40:
        print(f"\n每类准确率 (按样本数排序):")
        for cls, (acc, n) in sorted(per_class.items(), key=lambda x: -x[1][1]):
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {cls:<35s} {acc:5.1f}% ({n:3d}) {bar}")


if __name__ == "__main__":
    main()
