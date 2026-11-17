import glob,os
import torch

def extract_subset(all_train_paths, dataset_name):
    # open subset file
    with open(f"./data/Embeddings/subsets/{dataset_name}/{dataset_name}_rand_50.txt") as f:
        train_subset = f.readlines()
    # create dict from all training images
    train_dict = {i.split("/")[-1]: i for i in all_train_paths}
    all_train_paths = None

    # create training subset based on random images
    train_paths = [train_dict[i.strip()] for i in train_subset]
    train_dict = None
    train_subset = None
    return train_paths


def load_CUB_bb():
    # load bounding boxes for CUB (based on numeric image id)
    # with open("./data/Datasets/CUB_200_2011/bounding_boxes.txt") as f: # 自己电脑+
    with open("./dataset_all/CUB_200_2011/bounding_boxes.txt") as f: # 服务器
        bb_gt = f.readlines() # id path
    # create dict mapping image id (numeric) to bounding box
    # with open("./data/Datasets/CUB_200_2011/images.txt") as f: # 自己电脑
    with open("./dataset_all/CUB_200_2011/images.txt") as f: # 服务器
        images = f.readlines() # id path

    # create dict mapping image id to image file name
    img_dict = {i.split(" ")[0]: i.split(" ")[1].strip() for i in images}

    bb_gt = {img_dict[i.split(" ")[0]].split("/")[-1].split(".")[0]: [int(j.split(".")[0]) for j in i.split(" ")[1:]] for i in bb_gt}
    return bb_gt


def load_data(dataset_name, model, prompt_specs, check_existing):
    if dataset_name == "CUB":
        dataset_name = "CUB_200_2011"
        # load file containing all training images
        # with open("./data/Datasets/CUB_200_2011/train_imgs.txt") as f:  原本是这个
        # with open("./data/Datasets/CUB_200_2011/images.txt") as f: # 自己电脑
        with open("./dataset_all/CUB_200_2011/images.txt") as f: # 服务器
            train_imgs = f.readlines()
        # general image path
        img_dir = "./dataset_all/CUB_200_2011/images/"
        # create list of image paths for training
        # train_paths = [img_dir+i.strip() for i in train_imgs]原代码
        train_paths = [
            os.path.normpath(
                os.path.join(
                    img_dir,
                    line.strip().split(' ', 1)[-1]  # 提取序号后的路径部分
                )
            )
            for line in train_imgs
            if line.strip() and len(line.strip().split(' ', 1)) > 1
        ]
        train_paths = [path.replace('\\', '/') for path in train_paths]

    if dataset_name == "ImageNette":
        # create dict with key: img id and value: path
        all_train_paths = glob.glob("./data/Datasets/imagenette2/train/*/*.JPEG")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "ImageWoof":
        all_train_paths = glob.glob("./data/Datasets/imagewoof2/train/*/*.JPEG")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "ImageNet":
        all_train_paths = glob.glob("./data/Datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/*/*.JPEG")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "Places365":
        all_train_paths = glob.glob("./data/Datasets/Places365/Data/train/*/*.jpg")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "cifar10":
        all_train_paths = glob.glob("./data/Datasets/cifar10/train/*/*.png")
        train_paths = extract_subset(all_train_paths, dataset_name)
    
    if dataset_name == "cifar100":
        all_train_paths = glob.glob("./data/Datasets/cifar100/train/*/*.png")
        train_paths = extract_subset(all_train_paths, dataset_name)
    
    if dataset_name == "ClimateTV":
        train_paths = glob.glob("./data/Datasets/ClimateTV/train/*/*.jpg")

    if dataset_name == "MiT-States":
        with open("./data/Datasets/MiT-States/train_imgs.txt") as f:
            train_imgs = f.readlines()
        img_dir = "./data/Datasets/MiT-States/release_dataset/images/"
        train_paths = [img_dir+i.strip() for i in train_imgs]

    if check_existing:
        # check if some images have already been processed
        if prompt_specs is None:
            ## successfully processed images
            complete = glob.glob(f"{dataset_name}_{model}/*/*/*")
            ## list of images without segments
            try:
                stats = torch.load(f"Seg_embs/{dataset_name}_{model}_STATS.torch")
            except:
                stats = {"total_img_wo_masks": []}
        else: 
            complete = glob.glob(f"{dataset_name}_{model}_{prompt_specs}/*/*/*")
            ## list of images without segments
            try:
                stats = torch.load(f"Seg_embs/{dataset_name}_{model}_{prompt_specs}_STATS.torch")
            except:
                stats = {"total_img_wo_masks": []}

        complete = [i.split("/")[-1] for i in complete]
        complete = [i.split("_OUT_")[0] for i in complete]

        no_seg_list = stats["total_img_wo_masks"]
        complete = complete + no_seg_list

        # keep only training images that are not yet processed
        train_paths_ver = []
        for path in train_paths:
            name = path.split("/")[-1].split(".")[0]
            if name not in complete:
                train_paths_ver.append(path)
    else: train_paths_ver = train_paths

    print("Creating segments for ", len(train_paths_ver), "images")
    if len(train_paths_ver) == 0:
        print("No images to segment")
        exit()
    
    return train_paths_ver, dataset_name


def load_segments(dataset_name, model, prompt_specs,seg_paths_file=None):
    # seg_paths_file 用于如果主函数用此函数，相对路径不匹配问题，即俩文件不在同一文件夹下的情况
    if seg_paths_file is None:
        if prompt_specs == "None":  # 是否有prompt_specs，提示词
            seg_paths = glob.glob(f"./data/Segments/Seg_embs/{dataset_name}_{model}/*")
        elif prompt_specs is not None:
            seg_paths = glob.glob(f"./data/Segments/Seg_embs/{dataset_name}_{model}_{prompt_specs}/*")
    else:
        # 使用绝对路径前缀
        if prompt_specs == "None":
            seg_paths = glob.glob(f"./data/Segments/Seg_embs/{seg_paths_file}/{dataset_name}_{model}/*")
        elif prompt_specs is not None:
            seg_paths = glob.glob(f"./data/Segments/Seg_embs/{seg_paths_file}/{dataset_name}_{model}_{prompt_specs}/*")
    # 判断seg_paths是否为空列表
    if not seg_paths:
        raise FileNotFoundError(f"找不到对应的 '{dataset_name}_{model}' 文件或目录，请修改data_loading文件第149行")
    print(seg_paths)
    return seg_paths
