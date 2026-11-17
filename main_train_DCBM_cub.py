"""DCBM training — supports CUB_200_2011, derm7pt, BUS_UCLM, ISIC-2019, places365_part.

Usage:
    python main_train_DCBM_cub.py --dataset CUB_200_2011
    python main_train_DCBM_cub.py --dataset derm7pt
    python main_train_DCBM_cub.py --dataset BUS_UCLM
    python main_train_DCBM_cub.py --dataset ISIC-2019
    python main_train_DCBM_cub.py --dataset places365_part
"""
import os, sys, argparse, numpy as np, pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from utils.dcbm import *

# ── Common ──────────────────────────────────────────────────────────
model_name           = "CLIP-ViT-L14"
segment_model = "SAM2"
concept_name         = None
selected_image_concepts = None
# device is now set from CLI (default: cuda:0 if available else cpu)
cluster_method       = "kmeans"
centroid_method      = "median"
one_hot              = False
crop                 = False
use_wandb            = False
segment_path         = "./data/Segments/Seg_embs/"

# ── Dataset configs ─────────────────────────────────────────────────

def setup_dataset(args):
    """Return (embed_path, dataset, class_labels_path, hyperparams_dict)"""
    ds = args.dataset

    if ds == "CUB_200_2011":
        embed_path = "./CUB_200_2011_split/"
        class_labels_path = "./data/classes/cub_classes.txt"
        hp = dict(clusters=2048, concept_per_class=None, epochs=200,
                  lambda_1=1e-4, lr=1e-4, batch_size=32,
                  project="DCBM_SAM2_CUB_200_2011")

    elif ds == "derm7pt":
        embed_path = "./derm7pt_split/"
        class_labels_path = "./data/classes/derm7pt_classes.txt"
        # Ensure derm7pt binary class file exists
        os.makedirs(os.path.dirname(class_labels_path), exist_ok=True)
        with open(class_labels_path, 'w') as f:
            f.write("benign\nmalignant\n")
        hp = dict(clusters=100, concept_per_class=None, epochs=200,
                  lambda_1=1e-3, lr=5e-4, batch_size=16,
                  project="DCBM_SAM2_derm7pt")

    elif ds == "BUS_UCLM":
        embed_path = "./BUS_UCLM_split/"
        class_labels_path = "./data/classes/BUS_UCLM_classes.txt"
        # Generate class labels file from CSV
        csv_path = "./dataset_all/BUS-UCLM/BUS-UCLM_labels.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Run BUS_UCLM_csv.py first. Missing: {csv_path}")
        df = pd.read_csv(csv_path)
        classes = sorted(df["label"].unique())
        os.makedirs(os.path.dirname(class_labels_path), exist_ok=True)
        with open(class_labels_path, 'w') as f:
            for c in classes:
                f.write(f"{c}\n")
        print(f"[BUS_UCLM] {len(classes)} classes: {classes}")
        # BUS-UCLM: small dataset (~200 imgs), strong regularization
        hp = dict(clusters=50, concept_per_class=None, epochs=300,
                  lambda_1=1e-3, lr=5e-4, batch_size=8,
                  project="DCBM_SAM2_BUS_UCLM")

    elif ds == "ISIC-2019":
        embed_path = "./ISIC-2019_split/"
        class_labels_path = "./data/classes/ISIC-2019_classes.txt"
        # ISIC-2019: 9-class skin lesion (~25K train imgs), medium scale
        hp = dict(clusters=512, concept_per_class=None, epochs=200,
                  lambda_1=1e-4, lr=1e-4, batch_size=64,
                  project="DCBM_SAM2_ISIC_2019")

    elif ds == "places365_part":
        embed_path = "./places365_part_split/"
        class_labels_path = "./data/classes/places365_part_classes.txt"
        # places365_part: 365-class scenes, ~93K train, ~5M segments
        # 类名含 / 的特殊路径 (如 "apartment_building/outdoor"), 保留原样
        # concept_per_class=500 → 365×500=182.5K segments (avoid OOM from 5M)
        hp = dict(clusters=1024, concept_per_class=500, epochs=200,
                  lambda_1=1e-4, lr=1e-4, batch_size=64,
                  project="DCBM_SAM2_places365_part")

    else:
        raise ValueError(f"Unknown dataset: {ds}")

    return embed_path, ds, class_labels_path, hp


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DCBM Training")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CUB_200_2011', 'derm7pt', 'BUS_UCLM', 'ISIC-2019',
                                 'places365_part'],
                        help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0' if __import__('torch').cuda.is_available() else 'cpu',
                        help='Device (default: cuda:0 if available else cpu)')
    args = parser.parse_args()

    embed_path, dataset, class_labels_path, hp = setup_dataset(args)
    device = args.device

    print(f"\n{'='*60}")
    print(f"  DCBM Training — {dataset}")
    print(f"  K={hp['clusters']}, λ={hp['lambda_1']}, lr={hp['lr']}, "
          f"epochs={hp['epochs']}, batch={hp['batch_size']}")
    print(f"  embed_path: {embed_path}")
    print(f"  class_labels: {class_labels_path}")
    print(f"{'='*60}\n")

    cbm = CBM(embed_path, dataset, model_name, class_labels_path, device=device)

    cbm.load_concepts(
        segment_path, segment_model, concept_name,
        selected_image_concepts, hp['concept_per_class'], crop=crop)

    if hp['clusters'] is not None:
        cbm.cluster_image_concepts(cluster_method, hp['clusters'], pca=True)
    else:
        cbm.clustered_concepts = cbm.image_segments

    cbm.centroid_concepts(centroid_method)
    cbm.preprocess_data(type_="standard", label_type=one_hot)
    cbm.train(
        num_epochs=hp['epochs'], lambda_1=hp['lambda_1'], lr=hp['lr'],
        device=device, batch_size=hp['batch_size'], project=hp['project'],
        to_print=True, early_stopping_patience=None, one_hot=one_hot,
        use_wandb=use_wandb)


if __name__ == "__main__":
    main()
