"""
CUB200-2011 数据集特征提取脚本 (严格复刻版)
严格按照原版 compute_embeddings_batched 的逻辑保存数据：
Value = [image_embedding_tensor, class_folder_name_string]
"""

import torch
import clip
import argparse
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path

def load_dataset_image_paths(dataset_name, dataset_path):
    """
    根据数据集名称加载图片路径列表和标签映射。
    返回: (image_paths: List[str], label_map: dict | None)
        - image_paths: 完整图片路径列表
        - label_map: {filename_lower: true_label_string} 或 None（使用文件夹名作为标签时）
    """
    import pandas as pd
    dataset_path = Path(dataset_path)
    image_paths = []
    label_map = None

    if dataset_name == "CUB_200_2011":
        # CUB 专用逻辑：读取 images.txt（文件夹名即类别标签，无需 label_map）
        images_txt_path = dataset_path / "images.txt"
        if not images_txt_path.exists():
            raise FileNotFoundError(f"找不到 {images_txt_path}")

        print(f"正在从 {images_txt_path} 读取 CUB 数据集索引...")
        with open(images_txt_path, "r") as f:
            for line in f.readlines():
                # 格式: id relative_path
                parts = line.strip().split(" ")
                if len(parts) == 2:
                    # parts[1] 是类似 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
                    relative_path = parts[1]
                    full_path = dataset_path / "images" / relative_path
                    if full_path.exists():
                        image_paths.append(str(full_path))
    elif dataset_name == "derm7pt":
        # derm7pt 数据集：图片按临床中心存放在 images/ 下，真实标签在 meta.csv 的 diagnosis 列
        images_dir = dataset_path / "images"
        meta_csv_path = dataset_path / "meta" / "meta.csv"
        if not images_dir.exists():
            raise FileNotFoundError(f"找不到图片目录: {images_dir}")
        if not meta_csv_path.exists():
            raise FileNotFoundError(f"找不到 meta.csv: {meta_csv_path}")

        # 1) 从 meta.csv 建立 文件名→二分类标签 映射
        print(f"正在从 {meta_csv_path} 读取诊断标签（二分类: benign / malignant）...")
        meta_df = pd.read_csv(meta_csv_path)

        def _to_binary(diagnosis):
            """将细粒度诊断映射为二分类标签。"""
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
        print(f"  建立了 {len(label_map)} 个文件名→标签的映射 "
              f"(benign: {sum(1 for v in label_map.values() if v == 'benign')}, "
              f"malignant: {sum(1 for v in label_map.values() if v == 'malignant')})")

        # 2) 扫描图片
        print(f"正在扫描 derm7pt 数据集图片 ({images_dir})...")
        for ext in ['*.jpg', '*.JPG', '*.JPEG', '*.jpeg']:
            image_paths.extend([str(p) for p in images_dir.rglob(ext)])

        print(f"共扫描到 {len(image_paths)} 张图片")

    return image_paths, label_map

def compute_embeddings_batched(image_paths, model, transform, output_file, device, batch_size=64, label_map=None):
    """
    原版逻辑复刻：
    1. Key 是文件名 (split('/')[-1])
    2. Value 是 [image_embedding, class_label_string]
    当 label_map 不为 None 时，用 {filename_lower: true_label} 查找真实标签；
    否则沿用原逻辑取父文件夹名作为标签。
    """
    # Key 提取逻辑：取路径最后一部分作为文件名
    image_keys = [i.split("/")[-1] for i in image_paths]

    embeddings = {}
    counter = 0

    # Process images in batches
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        pbar = tqdm(range(0, len(image_paths), batch_size), desc="Embedding Segments")

        for i in pbar:
            batch_images = []
            # 切片获取当前批次的 keys
            batch_keys = image_keys[i:i+batch_size]

            # 1. 读取并预处理图片
            for x in range(i, min(i + batch_size, len(image_paths))):
                image_path = image_paths[x]
                try:
                    image = Image.open(image_path)
                    # 处理透明通道
                    if image.mode in ('P', 'RGBA'):
                        image = image.convert("RGBA")
                    else:
                        image = image.convert("RGB")

                    # 预处理并增加 batch 维度
                    batch_images.append(transform(image).unsqueeze(0))
                except Exception as e:
                    print(f"\n跳过损坏图片 {image_path}: {e}")
                    # 如果是损坏图片，我们需要占位或者跳过，这里为了对齐索引，稍后处理

            if not batch_images:
                continue

            # 合并 Tensor
            batch_tensor = torch.cat(batch_images).to(device)

            # 2. 计算嵌入
            batch_embeddings = model.encode_image(batch_tensor).cpu()

            # 3. 存入字典
            # 注意：这里必须严格保证 batch_keys 和 batch_embeddings 的索引对应
            for idx, full_path in enumerate(image_paths[i:i + len(batch_keys)]):
                try:
                    # 确定类别标签：
                    # - 如果有 label_map（如 derm7pt 的诊断标签），用文件名查找真实标签
                    # - 否则沿用原逻辑：取父文件夹名（CUB 等数据集文件夹名即类别）
                    if label_map is not None:
                        filename_lower = batch_keys[idx].lower()
                        class_name = label_map.get(filename_lower,
                                                   os.path.basename(os.path.dirname(full_path)))
                    else:
                        class_name = os.path.basename(os.path.dirname(full_path))

                    # 严格复刻结构：[Tensor, String]
                    embeddings[batch_keys[idx]] = [batch_embeddings[idx], class_name]
                except Exception as e:
                    print(f"保存失败：Error saving {full_path}: {e}")
                    pass

            # 4. 分批保存逻辑 (防止内存溢出)
            if len(embeddings) > 400000:
                outfile = output_file.replace(".torch", f"_{counter}.torch")
                torch.save(embeddings, outfile)
                print(f"疑似内存溢出：Saved embeddings dictionary to {outfile}")
                embeddings = {}
                counter += 1

        # 保存剩余部分
        if embeddings:
            outfile = output_file if counter == 0 else output_file.replace(".torch", f"_{counter}.torch")
            torch.save(embeddings, outfile)
            print(f"成功保存：Saved embeddings dictionary to {outfile} ({len(embeddings)} embeddings)")

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for CUB dataset")
    parser.add_argument('--dataset_path', type=str, default="./CUB_200_2011", help='Path to the dataset root')
    parser.add_argument('--out_dir', type=str, default="./CUB_200_2011_split", help='Directory to save output torch files')
    parser.add_argument('--dataset', type=str, default="CUB_200_2011", help='Dataset name')
    parser.add_argument('--model_name', type=str, default="ViT-L/14", help='CLIP model name')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use')

    args = parser.parse_args()

    # 1. 准备路径
    os.makedirs(args.out_dir, exist_ok=True)

    # 构造输出文件名: images_{dataset}_all_{model_name}.torch
    # 处理 model_name 中的斜杠，例如 ViT-L/14 -> ViT-L-14
    safe_model_name = args.model_name.replace("/", "")
    output_filename = f"images_{args.dataset}_all_{safe_model_name}.torch"
    output_path = os.path.join(args.out_dir, output_filename)

    print(f"输出文件将保存至: {output_path}")

    # 2. 加载模型
    print(f"正在加载 CLIP 模型 {args.model_name} ...")
    model, preprocess = clip.load(args.model_name, device=args.device)
    model.eval()

    # 3. 获取图片路径与标签映射
    image_paths, label_map = load_dataset_image_paths(args.dataset, args.dataset_path)
    print(f"共找到 {len(image_paths)} 张图片.")

    if len(image_paths) == 0:
        print("未找到图片，请检查 dataset_path 是否正确。")
        return

    # 4. 执行计算与保存
    compute_embeddings_batched(
        image_paths=image_paths,
        model=model,
        transform=preprocess,
        output_file=output_path,
        device=args.device,
        batch_size=64,
        label_map=label_map
    )

if __name__ == "__main__":
    main()
    # derm7pt:python -m scripts.preprocessing.image_all_emb_to_clip --dataset derm7pt --dataset_path ./dataset_all/derm7pt --out_dir ./derm7pt_split --model_name "ViT-L/14" --device cuda:0
    # CUB:python -m scripts.preprocessing.image_all_emb_to_clip --dataset CUB_200_2011 --dataset_path ./dataset_all/CUB_200_2011 --out_dir ./CUB_200_2011_split --model_name "ViT-L/14" --device cuda:0