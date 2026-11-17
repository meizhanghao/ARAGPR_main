import os, shutil, random
from pathlib import Path
def create_class_balanced_subsets(original_dir, train_dir, test_dir, val_dir,
                                  train_record_file=None, test_record_file=None, val_record_file=None,
                                  test_ratio=0.2, val_ratio=0.2):
    """
    创建类平衡的训练集、测试集和验证集子集
    从原始数据集中按比例抽取每个类别的样本，创建平衡的子集
    参数说明:
        original_dir: 原始数据集目录
        train_dir: 训练集目录
        test_dir: 测试集目录
        val_dir: 验证集目录
        train_record_file: 训练集记录文件路径（.txt），如果为None则不生成
        test_record_file: 测试集记录文件路径（.txt），如果为None则不生成
        val_record_file: 验证集记录文件路径（.txt），如果为None则不生成
        test_ratio: 测试集比例(default: 0.2)
        val_ratio: 验证集比例(default: 0.2)
    """
    original_dir = Path(original_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    val_dir = Path(val_dir)

    if not original_dir.is_dir():
        raise ValueError(f"The provided path '{original_dir}' is not a directory.")

    # 检查目标目录是否已存在
    for dir_path, name in [(train_dir, 'train'), (test_dir, 'test'), (val_dir, 'val')]:
        if dir_path.exists():
            raise ValueError(f"The provided path '{dir_path}' already exists.")

    # 创建目标目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 存储各集合的文件名列表
    train_filenames = []
    test_filenames = []
    val_filenames = []

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

        # 计算各个集合的大小
        total_files = len(files)
        test_size = max(1, int(total_files * test_ratio))
        val_size = max(1, int(total_files * val_ratio))

        # 确保test_size + val_size不超过总文件数
        if test_size + val_size >= total_files:
            # 如果比例过大，调整为最多使用总数的一半用于test+val
            test_size = max(1, min(test_size, total_files // 3))
            val_size = max(1, min(val_size, total_files // 3))

        # 分配文件到不同集合
        test_files = files[:test_size]
        val_files = files[test_size:test_size + val_size]
        train_files = files[test_size + val_size:]

        print(f"   类别 {class_name}: 总文件数={total_files}, "
              f"训练集大小={len(train_files)}, 测试集大小={test_size}, 验证集大小={val_size}")

        # 创建目标类别的子目录
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        val_class_dir = val_dir / class_name

        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)

        # 复制测试集文件
        for idx, file_path in enumerate(test_files):
            shutil.copy(str(file_path), str(test_class_dir))
            relative_path = str(file_path.relative_to(original_dir))
            test_filenames.append(relative_path)
            if idx < 5 or idx == len(test_files) - 1:
                print(f"   复制到测试集: {file_path.name} → {test_class_dir}")

        # 复制验证集文件
        for idx, file_path in enumerate(val_files):
            shutil.copy(str(file_path), str(val_class_dir))
            relative_path = str(file_path.relative_to(original_dir))
            val_filenames.append(relative_path)
            if idx < 5 or idx == len(val_files) - 1:
                print(f"   复制到验证集: {file_path.name} → {val_class_dir}")

        # 复制训练集文件
        for idx, file_path in enumerate(train_files):
            shutil.copy(str(file_path), str(train_class_dir))
            relative_path = str(file_path.relative_to(original_dir))
            train_filenames.append(relative_path)
            if idx < 5 or idx == len(train_files) - 1:
                print(f"   复制到训练集: {file_path.name} → {train_class_dir}")

        print(f"   完成类别 {class_name}: "
              f"训练集={len(train_files)}, 测试集={test_size}, 验证集={val_size}")

    # 生成各集合记录文件
    def write_record_file(record_file, filenames, set_name):
        if record_file and filenames:
            record_file = Path(record_file)
            record_file.parent.mkdir(parents=True, exist_ok=True)
            with open(record_file, 'w') as f:
                for filename in filenames:
                    f.write(f"{filename}\n")
            print(f"\n  生成{set_name}记录文件: {record_file}")
            print(f"   包含 {len(filenames)} 个{set_name}文件")

    write_record_file(train_record_file, train_filenames, "训练集")
    write_record_file(test_record_file, test_filenames, "测试集")
    write_record_file(val_record_file, val_filenames, "验证集")

    print(f"\n数据集划分完成!")
    print(f"   训练集目录: {train_dir}")
    print(f"   测试集目录: {test_dir}")
    print(f"   验证集目录: {val_dir}")
    print(f"   总计: 训练集 {len(train_filenames)} 个文件, "
          f"测试集 {len(test_filenames)} 个文件, "
          f"验证集 {len(val_filenames)} 个文件")


if __name__ == "__main__":
    # 原始数据集路径
    # original_dataset_path = "./dataset_all/CUB_200_2011/images" # CUB
    original_dataset_path = "./dataset_all/derm7pt/images" # derm7pt
    # 新建的数据集目录结构
    # base_output_dir = "CUB_200_2011_split"  # CUB 输出目录
    base_output_dir = "derm7pt_split"

    # 设置各数据集目录
    train_dataset_path = f"{base_output_dir}/train"
    test_dataset_path = f"{base_output_dir}/test"
    val_dataset_path = f"{base_output_dir}/val"

    # 设置各记录文件路径
    train_record_file = f"{base_output_dir}/train_imgs.txt"
    test_record_file = f"{base_output_dir}/test_imgs.txt"
    val_record_file = f"{base_output_dir}/val_imgs.txt"

    # 调用函数，按6:2:2比例划分
    create_class_balanced_subsets(
        original_dataset_path,
        train_dataset_path,
        test_dataset_path,
        val_dataset_path,
        train_record_file,
        test_record_file,
        val_record_file,
        test_ratio=0.2,  # 20%作为测试集
        val_ratio=0.2  # 20%作为验证集，剩余60%为训练集
    )