import cv2
import os
import numpy as np
from .segment import create_statistics, create_crops_masks, filter_valid_masks, asset_sha256

def resize_image(image, target_size=1024):
    """Resize image while maintaining aspect ratio."""
    h, w, _ = image.shape
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def generate_mask(sam2, img_path, out_path, create_stats, create_segments, dataset_name, ifolder=None, mask_only=False):

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")
    source_metadata = {"path": os.path.abspath(img_path),
                       "size_wh": [image.shape[1], image.shape[0]],
                       "sha256": asset_sha256(img_path)}
    if ifolder is None:
        ifolder = os.path.basename(os.path.dirname(img_path))
    ipath = os.path.splitext(os.path.basename(img_path))[0]
    # create folder if not exists
    os.makedirs(f"{out_path}/crops/{ifolder}", exist_ok=True)
    os.makedirs(f"{out_path}/masks/{ifolder}", exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, target_size=1024)
    # Generate the masks
    # Initialize the automatic mask generator with detailed parameters
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=64,              # Increase to sample more points across the image # original: 32
        pred_iou_thresh=0.88,             # Increase for higher-quality masks # original 0.88
        stability_score_thresh=0.95,      # Higher stability threshold for more stable masks # ogirinal 0.9
        box_nms_thresh=0.5,              # Adjust Non Maximum Suppression to prevent overlapping masks # original 0.75
        min_mask_region_area=500         # Ignore small masks, set a lower value for more details # original 0

    )
    masks = filter_valid_masks(mask_generator.generate(image), image.shape)
    boxes = [m["bbox"] for m in masks]
    image_key = f"{ifolder}/{ipath}" if ifolder else ipath
    no_masks_count = int(len(masks) == 0)
    img_wo_masks = [image_key] if no_masks_count else []

    # ── Restore BGR before saving crops ──
    # The model processes RGB; segment.py writes with cv2.imwrite
    # which expects BGR.  Convert here so segment.py can save directly
    # without guessing the channel order.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if create_segments:
        ipath, _, _, _, _ = create_crops_masks(
            masks, image, out_path, ifolder, ipath, mask_only=mask_only,
            source_metadata=source_metadata)
    else:
        print("create_segments参数为false")

    if create_stats:

        s = create_statistics(boxes, img_path, dataset_name)

        return s, ipath, no_masks_count, img_wo_masks
    else:
        # Completion tracking always needs the expected output count.
        s = {"num_segments": len(boxes), "segments": boxes}
    return s, ipath, no_masks_count, img_wo_masks
