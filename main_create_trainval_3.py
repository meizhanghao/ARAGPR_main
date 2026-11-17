import os,shutil,random
from pathlib import Path

def create_class_balanced_subset(original_dir, subset_dir, val_record_file=None, ratio=0.1):
    """
    创建类平衡的验证集子集（非破坏性版本）
    从原始数据集中按比例抽取每个类别的样本，创建一个平衡的子集（通常用于验证集），同时从原始数据集中移除这些样本
    参数说明:
        original_dir: 原始数据集目录（只读，不修改）
        subset_dir: 验证集目录（新创建）
        新增：val_record_file: 验证集记录文件路径（.txt），如果为None则不生成
        ratio: 每个类别抽取的比例(default: 0.1).
    """
    original_dir = Path(original_dir)
    subset_dir = Path(subset_dir)
    if not original_dir.is_dir(): # 验证原始目录是否存在
        raise ValueError(f"The provided path '{original_dir}' is not a directory.")
    if subset_dir.exists(): # 如果目标目录已存在，不覆盖，按理说应该没有才对
        raise ValueError(f"The provided path '{subset_dir}' already exists.")
    # 创建目标目录
    subset_dir.mkdir(parents=True, exist_ok=True)
    val_filenames = [] # 存储所有验证集文件名的列表
    # 遍历原始目录下的每个类别子目录
    class_dirs = [d for d in original_dir.iterdir() if d.is_dir()]
    print(f"在original_dataset_path下发现 {len(class_dirs)} 个类别")
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n 处理类别: {class_name}")

        # 获取类别内所有.jpg文件
        files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg']
        if not files:
            print(f"   警告: 类别 {class_name} 没有.jpg文件，跳过")
            continue
        # 随机打乱文件顺序
        random.shuffle(files)
        # 计算子集大小（至少保留1个文件）
        subset_size = max(1, int(len(files) * ratio))
        print(f"   类别 {class_name}: 总文件数={len(files)}, 验证集大小={subset_size}")
        # 创建目标类别的子目录
        subset_class_dir = subset_dir / class_name
        subset_class_dir.mkdir(exist_ok=True)
        # 选择前subset_size个文件作为验证集
        selected_files = files[:subset_size]
        # 复制文件并记录文件名
        for idx, file_path in enumerate(selected_files):
            # 复制文件（不删除原始文件）
            shutil.copy(str(file_path), str(subset_class_dir))
            # 记录相对路径（相对于原始数据集根目录）
            relative_path = str(file_path.relative_to(original_dir))
            val_filenames.append(relative_path)
            if idx < 5 or idx == subset_size - 1:  # 只显示前5个和最后一个
                print(f"   复制: {file_path.name} → {subset_class_dir}")

        print(f"   完成类别 {class_name}: 复制了 {subset_size} 个文件到验证集")

        # 生成验证集记录文件
    if val_record_file and val_filenames:
        val_record_file = Path(val_record_file)
        val_record_file.parent.mkdir(parents=True, exist_ok=True)
        with open(val_record_file, 'w') as f:
            for filename in val_filenames:
                f.write(f"{filename}\n")
        print(f"\n  生成验证集记录文件: {val_record_file}")
        print(f"   包含 {len(val_filenames)} 个验证集文件")
    print(f"\n验证集创建完成!")
    print(f"   验证集目录: {subset_dir}")
    print(f"   总共复制了 {len(val_filenames)} 个文件")
    if val_record_file:
        print(f"   验证集记录: {val_record_file}")


if __name__ == "__main__":
    # Paths to original dataset and subset directory
    original_dataset_path = "./dataset_all/CUB_200_2011/images"  # e.g. ../cifar100/train，上传原图的路径
    subset_dataset_path = "CUB_200_2011_val/val"  # e.g. ../cifar100/val,创建在本目录下
    val_record_file = "CUB_200_2011_val/val_imgs.txt"  # 验证集记录文件
    # 调用非破坏性版本
    create_class_balanced_subset(
        original_dataset_path,
        subset_dataset_path,
        val_record_file,  # 启用记录文件生成
        ratio=0.1  # 10%作为验证集
    )
