# load packages
import torch
from PIL import Image, ImageFile
import os
import clip
import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to compute and save embeddings
def compute_embeddings(img_dir, model, transform, output_file, device):
    embeddings = {}
    for image_name in tqdm.tqdm(img_dir):
        image_path = img_dir[image_name]

        # load image
        image = Image.open(image_path)

        # Handle images with transparency
        if image.mode in ('P', 'RGBA'): image = image.convert("RGBA")
        else: image = image.convert("RGB")

        # use clip required preprocessing
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model.encode_image(image.to(device)).squeeze()#.numpy()  # Remove batch dimension and convert to numpy
        class_name = image_name.split('_')[0]
        embeddings[image_name] = [embedding, class_name]
    
    torch.save(embeddings, output_file)


def fix_embeddings(image_dict, model, transform, device):
    embeddings = {}
    corrupted = []

    for i in tqdm.tqdm(image_dict):
         # load image (skip corrupted files instead of crashing)
        try:
            image = Image.open(image_dict[i])
            image.verify()
            image = Image.open(image_dict[i])
        except Exception as e:
            print(f"  ⚠ 跳过损坏图片 [{image_dict[i]}]: {e}")
            corrupted.append(image_dict[i])
            continue

        # Handle images with transparency
        if image.mode in ('P', 'RGBA'): image = image.convert("RGBA")
        else: image = image.convert("RGB")

        # use clip required preprocessing
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model.encode_image(image.to(device)).squeeze()#.numpy()  # Remove batch dimension and convert to numpy
        class_name = os.path.basename(os.path.dirname(image_dict[i]))
        # 修复多级类名: 如果路径中包含 "/crops/", 取 crops/ 之后的所有部分作为类名
        dir_path = os.path.dirname(image_dict[i]).replace('\\', '/')
        if '/crops/' in dir_path:
            class_name = dir_path.split('/crops/', 1)[1]
        else:
            class_name = os.path.basename(dir_path)
        embeddings[i] = [embedding, class_name]

    if corrupted:
        print(f"\n⚠ fix_embeddings: 共跳过 {len(corrupted)} 张损坏的图片")

    return embeddings



# Function to compute and save embeddings as a dictionary
def compute_embeddings_batched(image_paths, model, transform, output_file, device, batch_size=64):
    # Collect all image paths and initialize the embeddings dictionary
    # 使用 crops/{class_path}/{basename} 作为 key，避免跨类 basename 冲突
    def _make_seg_key(p):
        p = p.replace('\\', '/')
        if '/crops/' in p:
            return p.split('/crops/', 1)[1]   # "airfield/0001_OUT_CROP_0_X_.jpg"
        return p.split('/')[-1]
    image_keys = [_make_seg_key(p) for p in image_paths]

    embeddings = {}
    corrupted = []  # 记录无法读取的损坏图片路径

    # Process images in batches
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Embedding Segments"):
            batch_images = []
            batch_keys = []
            batch_paths = []  # 记录每个成功加载图片的完整路径 (用于提取类名)

            for x in range(i, min(i + batch_size, len(image_paths))):
                image_path = image_paths[x]

                # Load image (skip corrupted files instead of crashing)
                try:
                    image = Image.open(image_path)
                    image.verify()  # 验证图片完整性
                    image = Image.open(image_path)  # verify() 后需要重新打开
                except Exception as e:
                    print(f"  ⚠ 跳过损坏图片 [{image_path}]: {e}")
                    corrupted.append(image_path)
                    continue

                # Handle images with transparency
                if image.mode in ('P', 'RGBA'):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")

                # Preprocess image for the model
                batch_images.append(transform(image).unsqueeze(0))  # Add batch dimension
                batch_keys.append(image_keys[x])
                batch_paths.append(image_path)

            # 如果整个 batch 都是损坏图片则跳过
            if len(batch_images) == 0:
                continue

            # Combine batch images into a tensor and move to device
            batch_images = torch.cat(batch_images).to(device)

            # Compute embeddings for the batch
            batch_embeddings = model.encode_image(batch_images).cpu()  # Move back to CPU

            # Store each embedding in the dictionary with the corresponding image key
            for idx, image_path in enumerate(batch_paths):
                try:
                    # 从完整路径中提取类名: crops/{class_path}/file.jpg → class_path
                    # 单级类: crops/airfield/0001.jpg → "airfield"
                    # 多级类: crops/apartment_building/outdoor/0001_OUT.jpg → "apartment_building/outdoor"
                    dir_path = os.path.dirname(image_path)
                    # 如果路径中包含 "/crops/", 取 crops/ 之后的所有部分作为类名
                    if '/crops/' in dir_path.replace('\\', '/'):
                        class_name = dir_path.replace('\\', '/').split('/crops/', 1)[1]
                    else:
                        class_name = os.path.basename(dir_path)
                    embeddings[batch_keys[idx]] = [batch_embeddings[idx], class_name]
                except Exception as e:
                    print(f"Error saving {image_path}: {e}")
                    pass

        # Save all embeddings to a single file (removed 400K sharding — downstream loaders expect one file)
        torch.save(embeddings, output_file)
        print(f"Saved embeddings dictionary to {output_file} ({len(embeddings)} embeddings)")

    # 报告损坏图片
    if corrupted:
        print(f"\n⚠ 共跳过 {len(corrupted)} 张损坏的图片:")
        for p in corrupted:
            print(f"  - {p}")
        # 保存损坏图片列表供后续排查
        corrupted_log = output_file.replace('.torch', '_corrupted.txt')
        with open(corrupted_log, 'w') as f:
            f.write('\n'.join(corrupted))
        print(f"损坏图片列表已保存至: {corrupted_log}")

# Function to compute and save embeddings as a dictionary
def compute_dinov2_batched(image_paths, model, transform, output_file, device, batch_size=64):
    # Collect all image paths and initialize the embeddings dictionary
    # 使用 crops/{class_path}/{basename} 作为 key，避免跨类 basename 冲突
    def _make_seg_key(p):
        p = p.replace('\\', '/')
        if '/crops/' in p:
            return p.split('/crops/', 1)[1]
        return p.split('/')[-1]
    image_keys = [_make_seg_key(p) for p in image_paths]
    
    embeddings = {}
    
    # Process images in batches
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Embedding Segments"):
            batch_images = []
            batch_keys = image_keys[i:i+batch_size]

            for x in range(i, min(i + batch_size, len(image_paths))):
                image_path = image_paths[x]

                # Load image
                image = Image.open(image_path)

                # Handle images with transparency
                if image.mode in ('P', 'RGBA'):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")

                # Preprocess image for the model
                batch_images.append(processor(images=image, return_tensors="pt").unsqueeze(0))  # Add batch dimension

            # Combine batch images into a tensor and move to device
            batch_images = torch.cat(batch_images).to(device)
            
            with torch.no_grad():
                # Compute embeddings for the batch
                batch_embeddings = model(batch_images).sqeeze().cpu()  # Move back to CPU

            # Store each embedding in the dictionary with the corresponding image key
            for idx, image_name in enumerate(batch_keys):
                try:
                    class_name = image_name.split('_')[0]
                    embeddings[image_name] = [batch_embeddings[idx], class_name]
                except Exception as e:
                    print(f"Error saving {image_name}: {e}")
                    pass

        # Save all embeddings to a single file (removed 400K sharding — downstream loaders expect one file)
        torch.save(embeddings, output_file)
        print(f"Saved embeddings dictionary to {output_file} ({len(embeddings)} embeddings)")
 
# Function to process a batch of images and compute embeddings
def process_batch(image_paths, model, transform, device):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
 
        # Handle images with transparency
        if image.mode in ('P', 'RGBA'):
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
 
        # Apply the CLIP required preprocessing
        images.append(transform(image).unsqueeze(0))  # Add batch dimension
    
    # Stack images into a single batch tensor
    images = torch.cat(images).to(device)
    
    # Perform inference
    with torch.no_grad():
        embeddings = model.encode_image(images)
 
    return embeddings.cpu()
 
# Function to compute and save embeddings for a folder using batch processing
def compute_folder_embeddings_batch(folder_name, model, transform, device, output_local_file, batch_size=100):
    embeddings_local = {}
    output_local = os.path.join(output_local_file, folder_name.split('/')[-1] + '.torch')
    
    if not os.path.exists(output_local):
        print("Creating embeddings for:", folder_name)
        image_paths = [os.path.join(folder_name, image_name) for image_name in os.listdir(folder_name)]
        
        # Process images in batches
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size)):
            batch_image_paths = image_paths[i:i+batch_size]
            embeddings = process_batch(batch_image_paths, model, transform, device)
 
            # Store embeddings with corresponding image names
            for j, image_path in enumerate(batch_image_paths):
                image_name = folder_name.split('/')[-1] + '_' + os.path.basename(image_path)
                embeddings_local[image_name] = embeddings[j]
        
        torch.save(embeddings_local, output_local)
        print("Embeddings saved for:", folder_name)
    else:
        print("Embedding already exists for:", folder_name)
 
# Function to parallelize folder embeddings with batch processing
def compute_embeddings_parallel(img_dirs, model, transform, output_file, device, batch_size=100):
    output_local_file = "../Embeddings/"
    if not os.path.exists(output_local_file):
        os.makedirs(output_local_file)
    
    # Parallelize embedding computation using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_folder_embeddings_batch, folder_name, model, transform, device, output_local_file, batch_size)
                   for folder_name in img_dirs]
 
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Ensure any exceptions are raised
 
    # Combine the embeddings into a single file
    output_file_ = "../Embeddings/images_incomplete.torch"
    if os.path.exists(output_file_):
        combined_embeddings = torch.load(output_file_)
        used_keys = torch.load("../Embeddings/used_keys.torch")
    else:
        used_keys = []
        combined_embeddings = {}
    for file in glob.glob(output_local_file + "*"):
        #embeddings_local = torch.load(file)
        #combined_embeddings.update(embeddings_local)
        try:
            if not file in used_keys:
                print(file)
                embeddings_local = torch.load(file)
                combined_embeddings.update(embeddings_local)
                used_keys.append(file)
        except Exception as e:
            print(f"Error saving {file}: {e}")
            if os.path.exists(file):
                os.remove(file)
            torch.save(combined_embeddings, output_file_)
            torch.save(used_keys, "../Embeddings/used_keys.torch")
            raise
            pass
    torch.save(combined_embeddings, output_file)
    # delete all entries in output_local_file
    for file in glob.glob(output_local_file + "*"):
        os.remove(file)
    print(f"Total embeddings: {len(combined_embeddings.keys())}")
    return combined_embeddings
 
 
# Function to compute and save embeddings
def compute_embeddings_pt(img_dir, model, transform, output_file, device):
    embeddings = {}
    for image_name in tqdm.tqdm(img_dir):
        image_path = img_dir[image_name]

        # load image
        image = Image.open(image_path)
        if image.mode == 'RGBA' or image.mode == 'LA':  # LA for grayscale + alpha
            # Discard the alpha channel and convert to RGB
            image = image.convert("RGBA")
        elif image.mode == 'P':  # 'P' mode means palette-based image, convert to RGB
            image = image.convert("RGBA")
        else:
            # For all other modes, convert to RGB directly (handles grayscale or other modes)
            image = image.convert("RGB")
            # Handle images with transparency
            if image.mode in ('P', 'RGBA'): 
                image = image.convert("RGBA")
            else: image = image.convert("RGB")

        # use clip required preprocessing
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model(image.to(device)).squeeze()#.numpy()  # Remove batch dimension and convert to numpy

        embeddings[image_name] = embedding
    
    torch.save(embeddings, output_file)
