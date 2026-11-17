"""
DCBM with multi-branch linear residuals (Shape / Cooccurrence).

Architecture:
    z_new = z_sem + eta_s*z_shape + eta_c*z_co
Residual heads are affine and L1-regularized; interpretation requires separate correspondence checks.
Semantic branch frozen after Stage 1.
Shape branch: CLIP-decoded at inference, SAM teacher during training only.

Two activation modes:
    global : original DCBM  (full-image CLIP embedding -> concepts)
    local  : DCBM-Local     (per-segment max-pool -> concepts)
"""

import numpy as np, os, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

# ---- Reproducibility ----

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # default seed for reproducibility


_IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def _build_unique_basename_index(image_keys):
    """Index basename aliases only when they identify exactly one image."""
    candidates = defaultdict(set)
    for image_key in image_keys:
        basename = image_key.replace("\\", "/").split("/")[-1]
        candidates[basename].add(image_key)
        candidates[os.path.splitext(basename)[0]].add(image_key)
    return {
        alias: next(iter(keys))
        for alias, keys in candidates.items()
        if len(keys) == 1
    }


def _resolve_image_key(source, image_keys, unique_basename_index):
    """Resolve a relative source path without guessing ambiguous basenames."""
    source = str(source).replace("\\", "/")
    while source.startswith("./"):
        source = source[2:]
    source = source.lstrip("/")
    has_extension = os.path.splitext(source)[1].lower() in {
        ".jpg", ".jpeg", ".png"
    }
    candidates = [source]
    if not has_extension:
        candidates.extend(source + ext for ext in _IMAGE_EXTENSIONS)

    for candidate in candidates:
        if candidate in image_keys:
            return candidate

    for candidate in candidates:
        basename = candidate.split("/")[-1]
        if basename in unique_basename_index:
            return unique_basename_index[basename]
    return None


# ---- GPU-accelerated PCA + KMeans ----

def _pca_kmeans_gpu(X, K, pca_dim=100, device='cuda:0', n_iter=30, batch_size=4096, seed=None):
    """GPU PCA + batched KMeans.  pca_dim=0 skips PCA."""
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    N, D = X_t.shape
    rng = np.random.RandomState(seed if seed is not None else 42)

    if pca_dim > 0 and pca_dim < D:
        # torch.pca_lowrank: returns (U, S, V)
        print(f"    torch.pca_lowrank ({N}x{D} -> {pca_dim})...")
        _, _, V = torch.pca_lowrank(X_t, q=pca_dim)
        X_work = (X_t @ V)  # [N, pca_dim], stays on GPU
    else:
        X_work = X_t
        pca_dim = D

    # KMeans on GPU (batched cdist to avoid OOM)
    print(f"    GPU KMeans K={K} on {N}x{pca_dim} ({n_iter} iters)...")
    # Init: kmeans++ style (pick K random points, then refine)
    idx = torch.from_numpy(rng.choice(N, K, replace=False)).to(device)
    centers = X_work[idx].clone()

    for it in range(n_iter):
        # E-step: assign labels in batches
        labels = torch.empty(N, dtype=torch.long, device=device)
        for b in range(0, N, batch_size):
            be = min(b + batch_size, N)
            dists = torch.cdist(X_work[b:be], centers)  # [B, K]
            labels[b:be] = dists.argmin(dim=1)

        # M-step: update centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=device)
        for k in range(K):
            mask = (labels == k)
            cnt = mask.sum()
            if cnt > 0:
                new_centers[k] = X_work[mask].mean(dim=0)
                counts[k] = cnt
            else:
                # Re-init empty cluster with random point
                ri = torch.randint(0, N, (1,), device=device)[0]
                new_centers[k] = X_work[ri]
                counts[k] = 1

        shift = ((centers - new_centers) ** 2).sum().item()
        centers = new_centers
        if (it + 1) % 10 == 0:
            print(f"      iter {it+1}/{n_iter} shift={shift:.4f}")

    # Final labels
    labels_final = torch.empty(N, dtype=torch.long, device=device)
    for b in range(0, N, batch_size):
        be = min(b + batch_size, N)
        labels_final[b:be] = torch.cdist(X_work[b:be], centers).argmin(dim=1)

    if pca_dim < D:
        return X_work.cpu().numpy(), centers.cpu().numpy(), labels_final.cpu().numpy()
    else:
        return None, centers.cpu().numpy(), labels_final.cpu().numpy()

# ---- residual modules ----
class ShapeBranch(nn.Module):
    def __init__(self, Ks, nc):
        super().__init__()
        self.linear = nn.Linear(Ks, nc, bias=True)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)

class CooccurBranch(nn.Module):
    def __init__(self, K, nc):
        super().__init__()
        self.linear = nn.Linear(K, nc, bias=True)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)

# ---- CLIP Shape Decoder (inference-only, SAM teacher during training) ----
class CLIPShapeDecoder(nn.Module):
    """Linear decoder: frozen CLIP image embedding -> SAM-derived shape activation.

    Training:  h_CLIP -> s_SAM  (MSE / SmoothL1, SAM provides teacher signal)
    Inference: h_CLIP -> s_hat_CLIP  (no SAM, no mask, no contour extraction)
    """
    def __init__(self, d_clip, K_shape):
        super().__init__()
        self.linear = nn.Linear(d_clip, K_shape, bias=True)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h):
        return self.linear(h)

# ---- Base ----
class BaseCBM:
    def __init__(self, epath, ds, mn, clp, device="cuda:0"):
        self.device = device; self.dataset = ds; self.model_name = mn
        self.embed_path = epath; self.class_labels = None; self.model = None
        self._cp(); self._ld(); self._validate_split_keys(); self._edl(); self._ll(clp)
    def _cp(self):
        mn = self.model_name; bp = self.embed_path; ds = self.dataset
        for s in ["train", "val", "test"]:
            p = os.path.join(bp, f"images_{ds}_{s}_{mn}.torch")
            setattr(self, f"{s}_path", p if os.path.exists(p) else None)
    def _ld(self):
        for s in ["train", "val", "test"]:
            p = getattr(self, f"{s}_path", None)
            if p: setattr(self, f"data_{s}_raw", torch.load(p, weights_only=False, map_location="cpu")); print(f"{s} loaded")
            else: setattr(self, f"data_{s}_raw", None)
    def _validate_split_keys(self):
        """Reject image-key overlap across train/validation/test splits."""
        split_keys = {
            split: set(getattr(self, f"data_{split}_raw", {}) or {})
            for split in ["train", "val", "test"]
        }
        overlaps = []
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            shared = split_keys[left] & split_keys[right]
            if shared:
                overlaps.append(
                    f"{left}/{right}: {len(shared)} shared keys, "
                    f"examples={sorted(shared)[:3]}"
                )
        if overlaps:
            raise ValueError(
                "Dataset split leakage detected. Image keys must be disjoint "
                "before fitting DCBM or the shape teacher:\n  "
                + "\n  ".join(overlaps)
            )
    def _edl(self):
        def sp(d, split_name=""):
            X, L, keys = [], [], []
            for k, v in d.items():
                X.append(np.squeeze(v[0].cpu().numpy()))
                L.append(v[1])
                keys.append(k)
            X_arr = np.array(X)
            L_arr = np.array(L)
            # NaN 过滤: CLIP 对损坏图片可能输出 NaN, 直接丢弃
            nan_mask = np.isnan(X_arr).any(axis=1)
            n_nan = nan_mask.sum()
            if n_nan > 0:
                print(f"  ⚠ {split_name}: {n_nan} 条 NaN 嵌入 (损坏图片), 已跳过")
                X_arr = X_arr[~nan_mask]
                L_arr = L_arr[~nan_mask]
                keys = [k for i, k in enumerate(keys) if not nan_mask[i]]
            return X_arr, L_arr, keys
        for s in ["train", "val", "test"]:
            raw = getattr(self, f"data_{s}_raw", None)
            if raw is not None:
                X, L, kept_keys = sp(raw, s)
                setattr(self, f"X_{s}", X); setattr(self, f"labels_{s}", L)
                # 同步过滤 raw dict
                if len(kept_keys) != len(raw):
                    setattr(self, f"data_{s}_raw", {k: raw[k] for k in kept_keys})
        enc = OneHotEncoder(sparse_output=False).fit(self.labels_train.reshape(-1, 1))
        for s in ["train", "val", "test"]:
            L = getattr(self, f"labels_{s}", None)
            if L is not None:
                # 过滤掉训练集中未出现的类别 (如 ISIC-2019 的 UNK 可能全在 val/test)
                known_mask = np.isin(L, enc.categories_[0])
                if not known_mask.all():
                    unknown_labels = set(L[~known_mask])
                    print(f"  ⚠ {s}: {len(L) - known_mask.sum()} 个样本含训练集未知标签 "
                          f"{unknown_labels}，已跳过")
                    L = L[known_mask]
                    Xs = getattr(self, f"X_{s}")
                    setattr(self, f"X_{s}", Xs[known_mask])
                    setattr(self, f"labels_{s}", L)
                    # 同步过滤 data_{s}_raw, 否则 shape teacher 等下游会用未过滤的 dict
                    raw = getattr(self, f"data_{s}_raw", None)
                    if raw is not None:
                        keep_keys = [k for k, v in raw.items()
                                     if v[1] in enc.categories_[0]]
                        filtered_raw = {k: raw[k] for k in keep_keys}
                        setattr(self, f"data_{s}_raw", filtered_raw)
                if len(L) == 0:
                    setattr(self, f"y_{s}_oh", np.empty((0, len(enc.categories_[0])), dtype=np.float32))
                    setattr(self, f"y_{s}_int", np.empty((0,), dtype=np.int64))
                    continue
                oh = enc.transform(L.reshape(-1, 1)).astype(np.float32)
                setattr(self, f"y_{s}_oh", oh)
                # Handle string labels (CUB format: "180.Wilson_Warbler")
                L_flat = L.flatten()
                if L_flat.dtype.kind in ('U', 'S', 'O'):
                    setattr(self, f"y_{s}_int", np.argmax(oh, axis=1).astype(np.int64))
                else:
                    setattr(self, f"y_{s}_int", L_flat.astype(np.int64))
        print(f"  classes: {self.y_train_oh.shape[1]}")
    def _ll(self, path):
        with open(path) as f: d = {i: l.strip() for i, l in enumerate(f)}
        if self.dataset == "CUB_200_2011": self.class_labels = [d[i].split(".")[1] for i in sorted(d)]
    def load_seg_mapping(self, sp, st):
        """Load segcrop file and build seg→image mapping ONLY (no clustering).

        Use this when concepts are restored from a checkpoint and only the
        segment-to-image-key index is needed (e.g. ResidualTrainer).
        """
        fn = f"segcrop_{self.dataset}_{st}_{self.model_name}.torch"
        cd = torch.load(os.path.join(sp, fn), weights_only=False, map_location="cpu")
        all_keys = list(cd.keys())

        all_img_keys = set()
        for raw in [self.data_train_raw, self.data_val_raw, self.data_test_raw]:
            if raw:
                all_img_keys.update(raw.keys())
        basename_to_ak = _build_unique_basename_index(all_img_keys)

        seg_to_img_all = []
        for sk in all_keys:
            found = None
            for m in ["_OUT_CROP_", "_OUT_SHAPE_", "_OUT_MASK_"]:
                if m in sk:
                    bn = sk.split(m)[0]
                    found = _resolve_image_key(bn, all_img_keys, basename_to_ak)
                    break
            seg_to_img_all.append(found)

        self.seg_keys = all_keys
        self.seg_to_img = seg_to_img_all
        del cd
        n_mapped = sum(1 for x in seg_to_img_all if x is not None)
        print(f"  seg→img mapping: {n_mapped}/{len(all_keys)} segments matched")

    def load_concepts(self, sp, st, K=None, use_pca=True, max_segments_per_class=500):
        """Load segcrop, cluster into K prototypes (GPU-accelerated).

        DATA LEAKAGE GUARD:
        - seg→image mapping built FIRST, before any sampling/clustering.
        - Per-class reservoir sampling and KMeans clustering use TRAIN segments ONLY.
        - PCA (if enabled) is fit on train segments only.
        - All segments (train+val+test) are stored for downstream per-image activation
          computation (LocalCBM max-pooling, shape teacher, etc.).

        use_pca=True (LocalCBM): PCA→100d before clustering
        use_pca=False (GlobalCBM): cluster on full 768d
        max_segments_per_class: per-class reservoir sampling (default 500).
            Classes with fewer segments are kept in full.
            Set to 0 for unlimited (may OOM on large datasets).
        """
        fn = f"segcrop_{self.dataset}_{st}_{self.model_name}.torch"
        cd = torch.load(os.path.join(sp, fn), weights_only=False, map_location="cpu")
        all_keys = list(cd.keys())
        M_total = len(all_keys)
        D = np.squeeze(cd[all_keys[0]][0]).shape[0]  # infer embedding dim from first entry

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Build seg→image mapping FIRST (before any sampling)
        # ═══════════════════════════════════════════════════════════════
        # Supports two seg key formats:
        #   a) bare basename: "00000001_OUT_CROP_0_X_.jpg" (legacy CUB, derm7pt, ...)
        #   b) class_path/basename: "airfield/00000001_OUT_CROP_0_X_.jpg"
        #      (places365_part with cross-class basename collisions)
        all_img_keys = set()
        for raw in [self.data_train_raw, self.data_val_raw, self.data_test_raw]:
            if raw: all_img_keys.update(raw.keys())
        # Legacy bare-basename keys are accepted only when globally unique.
        basename_to_ak = _build_unique_basename_index(all_img_keys)

        seg_to_img_all = []  # length M_total, aligned with all_keys
        for sk in all_keys:
            found = None
            for m in ["_OUT_CROP_", "_OUT_SHAPE_", "_OUT_MASK_"]:
                if m in sk:
                    bn = sk.split(m)[0]       # "airfield/00003658" or bare "00003658"
                    found = _resolve_image_key(
                        bn, all_img_keys, basename_to_ak
                    )
                    break
            seg_to_img_all.append(found)

        # ═══════════════════════════════════════════════════════════════
        # Step 2: Split segment keys into train vs non-train
        # ═══════════════════════════════════════════════════════════════
        train_img_set = set(self.data_train_raw.keys()) if self.data_train_raw else set()
        train_seg_indices = []      # indices into all_keys
        non_train_seg_indices = []
        for i, sk in enumerate(all_keys):
            img = seg_to_img_all[i]
            if img is not None and img in train_img_set:
                train_seg_indices.append(i)
            else:
                non_train_seg_indices.append(i)

        n_train_total = len(train_seg_indices)
        n_val_test = len(non_train_seg_indices)
        print(f"  {M_total} total segments: {n_train_total} train, "
              f"{n_val_test} val/test (excluded from KMeans)")

        # ═══════════════════════════════════════════════════════════════
        # Step 3: Per-class reservoir sampling on TRAIN segments ONLY
        # ═══════════════════════════════════════════════════════════════
        if max_segments_per_class and max_segments_per_class > 0:
            rng = random.Random(42)
            class_samples = defaultdict(list)   # label -> list of up to N keys
            class_counts = defaultdict(int)      # label -> total seen (train only)
            for idx in train_seg_indices:
                k = all_keys[idx]
                label = cd[k][1]
                class_counts[label] += 1
                n = len(class_samples[label])
                if n < max_segments_per_class:
                    class_samples[label].append(k)
                else:
                    # Reservoir sampling: replace with prob N / count
                    j = rng.randint(0, class_counts[label] - 1)
                    if j < max_segments_per_class:
                        class_samples[label][j] = k
            sampled_train_keys = []
            for keys in class_samples.values():
                sampled_train_keys.extend(keys)
            M = len(sampled_train_keys)
            n_classes = len(class_counts)
            print(f"  per-class ≤{max_segments_per_class} on TRAIN only → "
                  f"sampled {M} train segments across {n_classes} classes, dim={D}")
        else:
            sampled_train_keys = [all_keys[i] for i in train_seg_indices]
            M = len(sampled_train_keys)
            print(f"  unlimited sampling → {M} train segments, dim={D}")

        # ═══════════════════════════════════════════════════════════════
        # Step 4: Extract embeddings; cluster on TRAIN-ONLY
        # ═══════════════════════════════════════════════════════════════
        # MEMORY: For GlobalCBM (use_pca=False), ONLY extract train-sampled embeddings.
        # For LocalCBM (which always uses_pca), also extract ALL embeddings for per-segment max-pool.
        # Places365 has 5M+ segments → all_embs ≈ 14 GiB float32, only needed for LocalCBM.
        # NOTE: isinstance check, NOT use_pca flag — GlobalCBM may also pass use_pca=True
        # for PCA-during-clustering, but it does NOT need per-segment embeddings.

        # Extract embeddings for sampled train keys (always needed for KMeans)
        sampled_train_set = set(sampled_train_keys)
        train_embs = np.array([np.squeeze(cd[k][0].cpu().numpy()) for k in sampled_train_keys],
                              dtype=np.float32)

        if isinstance(self, LocalCBM):
            # LocalCBM: extract ALL embeddings (needed for per-segment activations)
            print(f"  Extracting ALL {M_total} segment embeddings for LocalCBM "
                  f"(≈ {M_total * D * 4 / (1024**3):.1f} GiB)...")
            all_embs = np.array([np.squeeze(cd[k][0].cpu().numpy()) for k in all_keys],
                                dtype=np.float32)
        else:
            # GlobalCBM: only keep train-sampled embeddings
            all_embs = train_embs

        del cd  # free the full dict immediately

        self.seg_keys = all_keys          # ALL keys (for downstream: LocalCBM, shape teacher)
        self.seg_to_img = seg_to_img_all  # ALL mappings

        if K is None: K = min(1024, M)
        if K > M:
            print(f"  ⚠ K={K} > train segments={M}, capping K to {M}")
            K = M

        train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
        semantic_proto_representative = {}

        if use_pca:
            # ── PCA path: fit on train, transform ALL ──
            pca_dim = min(100, M, D)
            print(f"  sklearn PCA {D}->{pca_dim} (fit on {M} train segments)...")
            pca_model = PCA(n_components=pca_dim, random_state=42)
            train_embs_pca = pca_model.fit_transform(train_norm).astype(np.float32)

            print(f"  sklearn KMeans K={K} on {M}x{pca_dim} (k-means++, 10 init)...")
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            train_labels = km.fit_predict(train_embs_pca)

            # Transform ALL segments through train-fitted PCA, predict labels for all
            all_norm = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
            all_embs_pca = pca_model.transform(all_norm).astype(np.float32)
            all_labels = km.predict(all_embs_pca)

            # Median centroid from ORIGINAL (un-normalized) train embeddings per cluster
            # — matches dcbm.py behaviour (clustering on L2-normed, centroids from original)
            concepts_768 = np.zeros((K, D), dtype=np.float32)
            for k in range(K):
                mask = train_labels == k
                cnt = mask.sum()
                if cnt > 0:
                    cluster_embs = train_embs[mask]  # ← original, NOT L2-normalized
                    median_vec = np.median(cluster_embs, axis=0)
                    distances = np.sum(np.abs(cluster_embs - median_vec), axis=1)
                    cluster_indices = np.flatnonzero(mask)
                    rep_idx = int(cluster_indices[np.argmin(distances)])
                    concepts_768[k] = train_embs[rep_idx]
                    semantic_proto_representative[k] = sampled_train_keys[rep_idx]
            self.K = K
            self.concepts = concepts_768           # [K, 768] — GlobalCBM
            self.concepts_pca = km.cluster_centers_.astype(np.float32)  # [K, pca_dim] — LocalCBM
            self.seg_pca = all_embs_pca            # [M_total, pca_dim] — LocalCBM
            self.seg_labels = all_labels            # [M_total] — LocalCBM
            self.seg_embs = all_embs                # [M_total, 768] — LocalCBM
            print(f"  semantic K={K} (PCA-clustered on train, centroids in 768d)")
        else:
            # ── Direct KMeans on full 768d (L2-normalized), train-only (GlobalCBM) ──
            print(f"  sklearn KMeans K={K} on {M}x{D} (k-means++, 10 init)...")
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            train_labels = km.fit_predict(train_norm)

            # Median centroid from ORIGINAL (un-normalized) train embeddings
            # — matches dcbm.py behaviour
            concepts_768 = np.zeros((K, D), dtype=np.float32)
            for k in range(K):
                mask = train_labels == k
                cnt = mask.sum()
                if cnt > 0:
                    cluster_embs = train_embs[mask]  # ← original, NOT L2-normalized
                    median_vec = np.median(cluster_embs, axis=0)
                    distances = np.sum(np.abs(cluster_embs - median_vec), axis=1)
                    cluster_indices = np.flatnonzero(mask)
                    rep_idx = int(cluster_indices[np.argmin(distances)])
                    concepts_768[k] = train_embs[rep_idx]
                    semantic_proto_representative[k] = sampled_train_keys[rep_idx]
            self.K = K
            self.concepts = concepts_768          # [K, D] — median centroids (original space)
            self.concepts_pca = None
            self.seg_pca = None
            self.seg_labels = train_labels         # [M] — train-only (unused by GlobalCBM)
            self.seg_embs = None                   # free memory (GlobalCBM doesn't need seg_embs)
            print(f"  semantic K={K} (train-only clustering, GlobalCBM)")

        # Each semantic coordinate is represented by one real training segment.
        # Persisting its key makes post-hoc explanations path-stable and avoids
        # basename ambiguity in datasets such as Places365.
        self.semantic_proto_representative = semantic_proto_representative

        # ── Data leakage audit ──
        n_train_mapped = sum(1 for img in seg_to_img_all if img is not None and img in train_img_set)
        n_val_mapped   = sum(1 for img in seg_to_img_all
                             if img is not None
                             and hasattr(self, 'data_val_raw') and self.data_val_raw
                             and img in self.data_val_raw)
        n_test_mapped  = sum(1 for img in seg_to_img_all
                             if img is not None
                             and hasattr(self, 'data_test_raw') and self.data_test_raw
                             and img in self.data_test_raw)
        print(f"  Data leakage guard: KMeans on {M} train segments only")
        print(f"    seg→img mapping: {n_train_mapped} train / {n_val_mapped} val / "
              f"{n_test_mapped} test (val+test stored but NOT used for clustering)")

# ---- Global DCBM ----
class GlobalCBM(BaseCBM):
    def preprocess(self):
        C = torch.tensor(self.concepts, dtype=torch.float32, device=self.device)
        n2 = torch.norm(C, dim=1, keepdim=True).pow(2)
        def ac(X):
            Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = Xt @ C.T / n2.T
            # 防御: 若上游未过滤干净, NaN logits 会导致整行 softmax 输出 NaN
            if torch.isnan(logits).any():
                n_nan = torch.isnan(logits).any(dim=1).sum().item()
                print(f"  ⚠ preprocess: {n_nan} 行含 NaN logits, 替换为 -inf")
                logits = torch.nan_to_num(logits, nan=float('-inf'))
            return F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        for s in ["train", "val", "test"]:
            X = getattr(self, f"X_{s}", None)
            if X is not None: setattr(self, f"A_{s}", ac(X))
        sc = StandardScaler().fit(self.A_train)
        for s in ["train", "val", "test"]:
            A = getattr(self, f"A_{s}", None)
            if A is not None: setattr(self, f"A_{s}_scaled", sc.transform(A).astype(np.float32))
        self.scaler = sc
        print(f"  A_train={self.A_train_scaled.shape}")

    def train_semantic(self, lr=1e-4, l1=1e-4, epochs=200, bs=32, es=30, tp=True):
        nc = self.y_train_oh.shape[1]
        model = nn.Linear(self.K, nc, bias=True).to(self.device)
        opt = optim.Adam(model.parameters(), lr=lr)
        Xt = torch.tensor(self.A_train_scaled, dtype=torch.float32)
        Xv = torch.tensor(self.A_val_scaled, dtype=torch.float32)
        Xte = torch.tensor(self.A_test_scaled, dtype=torch.float32)
        yt = torch.tensor(self.y_train_oh, dtype=torch.float32)
        yv = torch.tensor(self.y_val_oh, dtype=torch.float32)
        yte = torch.tensor(self.y_test_oh, dtype=torch.float32)
        def ceo(i, t): return -(t * F.log_softmax(i, dim=1)).sum(dim=1).mean()
        bv = -1; bw = None; pat = 0
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(Xt)); tl = 0.0; corr = 0
            for b in range(0, len(Xt), bs):
                idx = perm[b:b+bs]; bx, by = Xt[idx].to(self.device), yt[idx].to(self.device)
                opt.zero_grad(); out = model(bx)
                loss = ceo(out, by) + l1 * torch.norm(model.weight, p=1)
                loss.backward(); opt.step()
                tl += loss.item() * len(idx); corr += (out.argmax(1) == by.argmax(1)).sum().item()
            model.eval()
            with torch.no_grad():
                vl = ceo(model(Xv.to(self.device)), yv.to(self.device)).item()
                va = (model(Xv.to(self.device)).argmax(1) == yv.to(self.device).argmax(1)).float().mean().item() * 100
                ta = (model(Xte.to(self.device)).argmax(1) == yte.to(self.device).argmax(1)).float().mean().item() * 100
            if va > bv: bv = va; bw = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
            else: pat += 1
            if tp: print(f"  Ep {ep+1}/{epochs} Loss {tl/len(Xt):.2f} Train {corr/len(Xt)*100:.2f}% Val {va:.2f}% Test {ta:.2f}%")
            if es and pat >= es: break
        model.load_state_dict(bw); self.model = model
        print(f"  Semantic done. best_val={bv:.2f}")

    def freeze_semantic(self):
        for p in self.model.parameters(): p.requires_grad = False
        self.model.eval()

# ---- Local DCBM ----
class LocalCBM(GlobalCBM):
    def preprocess(self):
        # LocalCBM 在 PCA 空间中计算 segment-level activations
        # concepts_pca: [K, pca_dim], seg_pca: [M, pca_dim]
        C_pca = self.concepts_pca if self.concepts_pca is not None else self.concepts
        C = torch.tensor(C_pca, dtype=torch.float32, device=self.device)
        n2 = torch.norm(C, dim=1, keepdim=True).pow(2)
        st = torch.tensor(self.seg_pca if self.seg_pca is not None else self.seg_embs,
                          dtype=torch.float32, device=self.device)
        raw = st @ C.T / n2.T
        self.seg_act = F.softmax(raw, dim=1).cpu().numpy().astype(np.float32)
        print(f"  seg_act: {self.seg_act.shape}")
        i2s = {}
        for si, ik in enumerate(self.seg_to_img):
            if ik is not None: i2s.setdefault(ik, []).append(si)
        def po(raw_dict):
            if raw_dict is None: return None
            keys = list(raw_dict.keys()); N = len(keys)
            act = np.zeros((N, self.K), dtype=np.float32)
            for i, k in enumerate(keys):
                idxs = i2s.get(k, [])
                if idxs: act[i] = self.seg_act[idxs].max(axis=0)
            return act
        for s in ["train", "val", "test"]:
            raw = getattr(self, f"data_{s}_raw", None)
            if raw is not None: setattr(self, f"A_{s}", po(raw))
        sc = StandardScaler().fit(self.A_train)
        for s in ["train", "val", "test"]:
            A = getattr(self, f"A_{s}", None)
            if A is not None: setattr(self, f"A_{s}_scaled", sc.transform(A).astype(np.float32))
        self.scaler = sc; self._tw(i2s)
        print(f"  A_train={self.A_train_scaled.shape}")

    def _tw(self, i2s):
        K = self.K
        def w(raw_dict):
            if raw_dict is None: return None
            keys = list(raw_dict.keys()); N = len(keys)
            widx = np.full((N, K), -1, dtype=np.int32)
            for i, k in enumerate(keys):
                idxs = i2s.get(k, [])
                if idxs: widx[i] = [idxs[self.seg_act[idxs][:, j].argmax()] for j in range(K)]
            return widx
        for s in ["train", "val", "test"]:
            raw = getattr(self, f"data_{s}_raw", None)
            if raw is not None: setattr(self, f"winners_{s}", w(raw))

# ---- Residual Trainer ----
class ResidualTrainer:
    def __init__(self, cbm, device="cuda:0"):
        self.cbm = cbm; self.device = device
        self.K = cbm.K; self.nc = cbm.y_train_oh.shape[1]
        self._i2s = {}
        for si, ik in enumerate(cbm.seg_to_img):
            if ik is not None: self._i2s.setdefault(ik, []).append(si)
        # Detect available branches

    def setup_shape_teacher(self, fd_csv, K_shape=256, fd_dim=32, shape_mode="decoded",
                             feat_cols=None,
                             shape_standardize=True,
                             shape_clustering="kmeans",
                             shape_prototype="median_nearest"):
        """
        Build SAM-derived shape activations (oracle or teacher signal).

        - oracle:   SAM shape activations used directly at both train & test (upper bound).
        - decoded:  SAM shape activations used as teacher to train CLIP→Shape decoder;
                    at inference, shape activations are decoded from CLIP (no masks).
        - decoded_rel: same as decoded, with reliability-weighted features.

        Ablation controls (Section 4.3):
        - shape_standardize: Z-score descriptor standardization before clustering (default True).
        - shape_clustering:  "kmeans" | "minibatch_kmeans".
        - shape_prototype:   "median_nearest" | "mean_center" | "mean_nearest"
                             (deprecated alias: "medoid" → "mean_nearest").

        DATA LEAKAGE GUARD:
        - Shape prototypes (KMeans) fitted on TRAIN masks only.
        - Feature scaler fitted on TRAIN segments only.
        - Val/test masks used ONLY for activation computation (oracle) or
          teacher signal + decodability evaluation (decoded).
        """
        is_oracle = (shape_mode == "oracle")
        tag = "Shape Oracle" if is_oracle else "Shape Teacher"
        print(f"[{tag}] FD: {fd_csv} (dim={fd_dim})")

        # ── Build image-key lookup (independent of semantic seg_keys) ──
        # Shape teacher loads ALL CSV rows and maps them to CBM image keys directly.
        # It does NOT depend on cbm.seg_keys — shape features are a fully independent modality.
        import csv as _csv2

        all_img_keys = set()
        for raw in [self.cbm.data_train_raw, self.cbm.data_val_raw, self.cbm.data_test_raw]:
            if raw: all_img_keys.update(raw.keys())
        basename_to_ak = _build_unique_basename_index(all_img_keys)

        # Infer feature column names from CSV header
        if feat_cols is None:
            with open(fd_csv, 'r', newline='') as fh_tmp:
                reader_tmp = _csv2.DictReader(fh_tmp)
                feat_cols = [c for c in reader_tmp.fieldnames
                             if c not in ('source_image', 'region_id', 'mask_relpath',
                                          'crop_relpath', 'mask_sha256', 'crop_sha256')]
        fdc = feat_cols

        # ── Build reverse index: CBM image key → row index per split ──
        def _build_rev_idx(raw_dict):
            if raw_dict is None: return {}
            return {k: i for i, k in enumerate(raw_dict.keys())}
        rev_train = _build_rev_idx(self.cbm.data_train_raw)
        rev_val   = _build_rev_idx(self.cbm.data_val_raw)
        rev_test  = _build_rev_idx(self.cbm.data_test_raw)
        train_img_set = set(rev_train.keys())
        n_imgs_train = len(rev_train)
        n_imgs_val   = len(rev_val)
        n_imgs_test  = len(rev_test)

        # Helper: map CSV source_image → CBM image key
        def _csv_src_to_img(src):
            return _resolve_image_key(src, all_img_keys, basename_to_ak)

        # ═══════════════════════════════════════════════════════════
        # PASS 1: Streaming reservoir sampling + partial_fit StandardScaler + KMeans
        # ═══════════════════════════════════════════════════════════
        # StandardScaler uses partial_fit on ALL training segments (statistically correct).
        # KMeans uses reservoir-sampled subset (up to MAX_TRAIN_SAMPLE) for efficiency.
        # This single-pass design avoids OOM while producing correct normalization stats.
        MAX_TRAIN_SAMPLE = 200000
        PARTIAL_FIT_BATCH = 50000
        N_INIT_KMEANS = 10       # sklearn default, stable clustering

        rng = random.Random(42)
        train_samples = []          # reservoir-sampled features for KMeans
        train_sample_keys = []      # seg reference keys
        train_sample_sources = []   # exact paired-asset provenance, if provided
        pf_batch = []               # partial_fit batch accumulator
        n_train_seen = 0
        n_total_matched = 0

        # Initialize incremental scaler (fit on ALL training data via partial_fit)
        if shape_standardize:
            self.shape_feat_scaler = StandardScaler()
        else:
            self.shape_feat_scaler = None

        with open(fd_csv, 'r', newline='') as fh:
            for row in _csv2.DictReader(fh):
                src = row['source_image']
                img_key = _csv_src_to_img(src)
                if img_key is None:
                    continue
                n_total_matched += 1
                feats = [float(row[c]) for c in fdc]

                if img_key in train_img_set:
                    n_train_seen += 1

                    # Accumulate for partial_fit (ALL training segments)
                    if self.shape_feat_scaler is not None:
                        pf_batch.append(feats)
                        if len(pf_batch) >= PARTIAL_FIT_BATCH:
                            self.shape_feat_scaler.partial_fit(
                                np.array(pf_batch, dtype=np.float32))
                            pf_batch.clear()

                    # Reservoir sample for KMeans
                    region_id = row['region_id'].strip()
                    if not region_id.isascii() or not region_id.isdigit():
                        raise ValueError(f"Invalid region_id for {src}: {region_id!r}")
                    # Region IDs are filename tokens; preserve any leading zeros.
                    sample_key = f"{src}_OUT_CROP_{region_id}_X_.jpg"
                    source = {c: row[c] for c in ('mask_relpath', 'crop_relpath',
                              'mask_sha256', 'crop_sha256') if row.get(c)}
                    if len(train_samples) < MAX_TRAIN_SAMPLE:
                        train_samples.append(feats)
                        train_sample_keys.append(sample_key)
                        train_sample_sources.append(source)
                    else:
                        j = rng.randint(0, n_train_seen - 1)
                        if j < MAX_TRAIN_SAMPLE:
                            train_samples[j] = feats
                            train_sample_keys[j] = sample_key
                            train_sample_sources[j] = source

        # Final partial_fit batch
        if self.shape_feat_scaler is not None and pf_batch:
            self.shape_feat_scaler.partial_fit(
                np.array(pf_batch, dtype=np.float32))
            pf_batch.clear()

        if n_total_matched == 0:
            raise ValueError("No CSV rows matched any CBM image keys.")
        n_train_sampled = len(train_samples)
        print(f"  Shape CSV: {n_total_matched} rows matched ({n_train_seen} train, "
              f"reservoir sampled {n_train_sampled} for KMeans)")
        if self.shape_feat_scaler is not None:
            print(f"  StandardScaler partial_fit on {n_train_seen} training segments "
                  f"({n_train_seen // PARTIAL_FIT_BATCH + 1} batches of ≤{PARTIAL_FIT_BATCH})")

        assert n_train_sampled >= K_shape, \
            f"Train shape segments ({n_train_sampled}) < K_shape ({K_shape}). Cannot cluster."

        # ── Transform + L2-normalize + Cluster (on reservoir-sampled train only) ──
        f_train = np.array(train_samples, dtype=np.float32)
        del train_samples  # free list of Python floats

        if shape_standardize:
            fs_train = self.shape_feat_scaler.transform(f_train).astype(np.float32)
        else:
            fs_train = f_train.astype(np.float32)

        # Ordinary KMeans on L2-normalized descriptors (not spherical KMeans).
        fs_train_norm = fs_train / (np.linalg.norm(fs_train, axis=1, keepdims=True) + 1e-8)

        if shape_clustering == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans
            print(f"    MiniBatchKMeans K={K_shape} on {n_train_sampled} L2-normed train features "
                  f"(k-means++, n_init={N_INIT_KMEANS}, batch=4096)...")
            km_shape = MiniBatchKMeans(n_clusters=K_shape, random_state=42,
                                       n_init=N_INIT_KMEANS, batch_size=4096)
        else:
            print(f"    KMeans K={K_shape} on {n_train_sampled} L2-normed train features "
                  f"(k-means++, n_init={N_INIT_KMEANS})...")
            km_shape = KMeans(n_clusters=K_shape, random_state=42, n_init=N_INIT_KMEANS)
        labels_shape = km_shape.fit_predict(fs_train_norm)
        print(f"    clustering done, computing prototypes ({shape_prototype})...")

        fd_dim_local = fs_train_norm.shape[1]
        representative_indices = {}
        if shape_prototype == "mean_center":
            # Cluster center in L2-normed space → re-normalize
            sp_norm = km_shape.cluster_centers_.astype(np.float32)
            sp_norm = sp_norm / (np.linalg.norm(sp_norm, axis=1, keepdims=True) + 1e-8)
        elif shape_prototype == "median_nearest":
            sp_norm = np.zeros((K_shape, fd_dim_local), dtype=np.float32)
            for k in range(K_shape):
                mask = labels_shape == k
                if mask.sum() > 0:
                    cluster_feats = fs_train_norm[mask]          # already L2-normed
                    median_vec = np.median(cluster_feats, axis=0)
                    # Cosine nearest to median: max dot product
                    cos_to_median = cluster_feats @ median_vec
                    selected_local = int(np.argmax(cos_to_median))
                    sp_norm[k] = cluster_feats[selected_local]
                    representative_indices[k] = int(np.flatnonzero(mask)[selected_local])
            # Re-normalize prototypes
            sp_norm = sp_norm / (np.linalg.norm(sp_norm, axis=1, keepdims=True) + 1e-8)
        elif shape_prototype in ("medoid", "mean_nearest"):
            if shape_prototype == "medoid":
                print(f"    ⚠ 'medoid' is deprecated, using 'mean_nearest' instead.")
            sp_norm = np.zeros((K_shape, fd_dim_local), dtype=np.float32)
            for k in range(K_shape):
                mask = labels_shape == k
                if mask.sum() > 0:
                    cluster_feats = fs_train_norm[mask]          # already L2-normed
                    center = cluster_feats.mean(axis=0)
                    # Cosine nearest to mean center
                    cos_to_center = cluster_feats @ center
                    selected_local = int(np.argmax(cos_to_center))
                    sp_norm[k] = cluster_feats[selected_local]
                    representative_indices[k] = int(np.flatnonzero(mask)[selected_local])
            # Re-normalize prototypes
            sp_norm = sp_norm / (np.linalg.norm(sp_norm, axis=1, keepdims=True) + 1e-8)
        else:
            raise ValueError(f"Unknown shape_prototype: {shape_prototype}")

        print(f"    prototypes done ({K_shape} prototypes in L2-normalized cosine space)")
        self.sp = sp_norm      # unit-norm prototypes — consistent with cosine activation
        self.Ks = K_shape

        # Retain the exact selected training member, including descriptor ties.
        # A mean center has only a display exemplar, not an exact real mask.
        if shape_prototype == "mean_center":
            for k in range(K_shape):
                members = np.flatnonzero(labels_shape == k)
                if len(members):
                    local = int(np.argmax(fs_train_norm[members] @ sp_norm[k]))
                    representative_indices[k] = int(members[local])
        self.shape_proto_representative = {
            k: train_sample_keys[i] for k, i in representative_indices.items()}
        self.shape_proto_metadata = {
            k: {
                "representative_key": train_sample_keys[i],
                "representative_role": ("nearest_cluster_member_display_only"
                    if shape_prototype == "mean_center" else "exact_selected_member"),
                "prototype_method": shape_prototype,
                "source_split": "train",
                "feature_columns": list(fdc),
                "raw_descriptor": f_train[i].tolist(),
                **train_sample_sources[i],
            } for k, i in representative_indices.items()}

        # ── Streaming top-5 nearest train segments per prototype (cosine similarity) ──
        import heapq
        proto_nearest_heaps = [[] for _ in range(K_shape)]
        # sp_norm and fs_train_norm are both L2-normalized → dot product = cosine similarity
        for local_i in range(len(fs_train_norm)):
            fv = fs_train_norm[local_i]
            cos_sims = fv @ sp_norm.T                             # [K_shape]
            for k in range(K_shape):
                heapq.heappush(proto_nearest_heaps[k], (cos_sims[k].item(), train_sample_keys[local_i]))
                if len(proto_nearest_heaps[k]) > 5:
                    heapq.heappop(proto_nearest_heaps[k])
        self.shape_proto_nearest = {}
        for k in range(K_shape):
            proto_nearest_heaps[k].sort(key=lambda x: -x[0])
            self.shape_proto_nearest[k] = [sk for _, sk in proto_nearest_heaps[k][:5]]
        print(f"  Saved nearest training masks for {K_shape} shape prototypes.")

        # Free train-only data
        del fs_train, fs_train_norm, f_train, labels_shape

        # ═══════════════════════════════════════════════════════════
        # PASS 2: Batched per-image max-cos (no full ssa / fa / shape_i2s)
        # ═══════════════════════════════════════════════════════════
        # Batch processing avoids Python-loop + single-row sklearn overhead
        # on large datasets (Places365 has 5M+ mask rows).
        # self.sp is already L2-normalized (unit-norm prototypes from cosine KMeans)
        sn = self.sp  # [K_shape, d] — unit norm, cos-sim = dot product
        PASS2_BATCH = 20000

        Ash_train = np.full((n_imgs_train, K_shape), -1e9, dtype=np.float32) if n_imgs_train > 0 else None
        Ash_val   = np.full((n_imgs_val,   K_shape), -1e9, dtype=np.float32) if n_imgs_val   > 0 else None
        Ash_test  = np.full((n_imgs_test,  K_shape), -1e9, dtype=np.float32) if n_imgs_test  > 0 else None
        seen_train = np.zeros(n_imgs_train, dtype=bool) if n_imgs_train > 0 else None
        seen_val   = np.zeros(n_imgs_val,   dtype=bool) if n_imgs_val   > 0 else None
        seen_test  = np.zeros(n_imgs_test,  dtype=bool) if n_imgs_test  > 0 else None

        def _process_batch2(rows_batch):
            """Batch-transform features and update per-image max activations."""
            if not rows_batch:
                return
            feats_batch = np.array([r[0] for r in rows_batch], dtype=np.float32)
            # Batch transform (much faster than row-by-row)
            if shape_standardize:
                feats_s = self.shape_feat_scaler.transform(feats_batch)
            else:
                feats_s = feats_batch
            fn_batch = feats_s / (np.linalg.norm(feats_s, axis=1, keepdims=True) + 1e-8)
            sims_batch = (fn_batch @ sn.T).astype(np.float32)  # [B, K_shape]

            for i in range(len(rows_batch)):
                img_key = rows_batch[i][1]
                sims = sims_batch[i]
                for rev_idx, Ash_split, seen_split in [
                    (rev_train, Ash_train, seen_train),
                    (rev_val,   Ash_val,   seen_val),
                    (rev_test,  Ash_test,  seen_test)]:
                    row_idx = rev_idx.get(img_key)
                    if row_idx is not None and Ash_split is not None:
                        Ash_split[row_idx] = np.maximum(Ash_split[row_idx], sims)
                        if seen_split is not None:
                            seen_split[row_idx] = True

        p2_batch = []
        with open(fd_csv, 'r', newline='') as fh:
            for row in _csv2.DictReader(fh):
                src = row['source_image']
                img_key = _csv_src_to_img(src)
                if img_key is None:
                    continue
                feats = [float(row[c]) for c in fdc]
                p2_batch.append((feats, img_key))
                if len(p2_batch) >= PASS2_BATCH:
                    _process_batch2(p2_batch)
                    p2_batch.clear()
        _process_batch2(p2_batch)  # final batch

        # Finalize: -1e9 → 0.0 for images with no shape segments
        for Ash_split in [Ash_train, Ash_val, Ash_test]:
            if Ash_split is not None:
                Ash_split[Ash_split < -1e8] = 0.0

        miss_train = (~seen_train).sum() if seen_train is not None else 0
        miss_val   = (~seen_val).sum()   if seen_val   is not None else 0
        miss_test  = (~seen_test).sum()  if seen_test  is not None else 0
        for s, miss, n in [("train", int(miss_train), n_imgs_train),
                            ("val",   int(miss_val),   n_imgs_val),
                            ("test",  int(miss_test),  n_imgs_test)]:
            if n > 0:
                print(f"    [{tag}] {miss}/{n} {s} images have no valid shape segment (set to 0)")

        self.Ash_train = Ash_train; self.Ash_val = Ash_val; self.Ash_test = Ash_test
        self.has_shape_train = seen_train
        self.has_shape_val   = seen_val
        self.has_shape_test  = seen_test

        # ── StandardScaler on train Ash: fit ONLY on images with valid shape ──
        # (Fixes: missing-teacher images DON'T contaminate scaler statistics)
        train_has_shape = self.has_shape_train
        if train_has_shape.sum() == 0:
            raise RuntimeError("No training images have valid shape — cannot fit shape_scaler.")
        self.shape_scaler = StandardScaler().fit(self.Ash_train[train_has_shape])
        for s in ["train", "val", "test"]:
            A = getattr(self, f"Ash_{s}", None)
            has_shape = getattr(self, f"has_shape_{s}", None)
            if A is not None:
                A_s = np.zeros_like(A, dtype=np.float32)
                if has_shape is not None and has_shape.any():
                    A_s[has_shape] = self.shape_scaler.transform(A[has_shape]).astype(np.float32)
                # Images without valid shape stay at zero → gated by has_shape mask in residual training
                setattr(self, f"Ash_{s}_s", A_s)

        # ── Leakage audit ──
        n_train_imgs = len(self.Ash_train) if hasattr(self, 'Ash_train') and self.Ash_train is not None else 0
        n_val_imgs   = len(self.Ash_val)   if hasattr(self, 'Ash_val')   and self.Ash_val   is not None else 0
        n_test_imgs  = len(self.Ash_test)  if hasattr(self, 'Ash_test')  and self.Ash_test  is not None else 0
        if is_oracle:
            inference_note = "True (oracle upper bound — SAM masks required at inference)"
        else:
            inference_note = "False (SDR-CBM uses CLIP-decoded Sclip — no masks at inference)"
        print(f"[{tag} Leakage Audit]")
        print(f"  shape standardize       : {shape_standardize}")
        print(f"  shape clustering        : {shape_clustering}")
        print(f"  shape prototype         : {shape_prototype}")
        print(f"  KMeans fit split        : train only "
              f"({n_train_sampled} sampled from {n_train_seen} segments)")
        if shape_standardize:
            print(f"  Feature scaler fit split: train only")
        else:
            print(f"  Feature scaler           : disabled")
        print(f"  Ash scaler fit split    : train only")
        print(f"  train image activations : {n_train_imgs}")
        print(f"  val   image activations : {n_val_imgs}")
        print(f"  test  image activations : {n_test_imgs}")
        print(f"  inference uses masks    : {inference_note}")
        print(f"  Ash_train_s={self.Ash_train_s.shape}")

    @staticmethod
    def _pearson_per_dim(Y, P):
        """Per-dimension Pearson correlation. Y, P: (N, K). Returns (mean_corr, corr_per_dim)."""
        Y = Y.astype(np.float32); P = P.astype(np.float32)
        Y0 = Y - Y.mean(axis=0, keepdims=True)
        P0 = P - P.mean(axis=0, keepdims=True)
        num = (Y0 * P0).sum(axis=0)
        den = np.sqrt((Y0 ** 2).sum(axis=0)) * np.sqrt((P0 ** 2).sum(axis=0)) + 1e-8
        corr = num / den
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.mean(corr)), corr

    def train_clip_shape_decoder(self, lr=1e-3, wd=1e-4, epochs=300, bs=64, es=30):
        """Train linear decoder: frozen CLIP image embedding -> SAM shape teacher.

        Uses SmoothL1 loss. After training, freeze decoder and precompute Sclip_* features.
        These Sclip_* replace Ash_*_s as the shape branch input at inference.

        DATA LEAKAGE / METRIC GUARD:
        - Decoder trained ONLY on training images that have valid shape teacher.
        - Val loss computed ONLY on val images with valid shape teacher.
        - Decodability metrics (MSE, R², Pearson) computed ONLY on shape-covered subsets.
        - Coverage stats reported per split.
        - Zero-predictor baseline reported for reference.
        """
        print(f"[CLIP-Shape Decoder] d_clip={self.cbm.X_train.shape[1]}, K_shape={self.Ks}")
        # Standardize CLIP embeddings for stable training
        Htr_raw = self.cbm.X_train.astype(np.float32)
        self.h_scaler = StandardScaler().fit(Htr_raw)
        Htr_all = self.h_scaler.transform(Htr_raw)
        Hva_all = self.h_scaler.transform(self.cbm.X_val.astype(np.float32))
        Hte_all = self.h_scaler.transform(self.cbm.X_test.astype(np.float32))

        # Teacher: SAM-derived shape activation (Ash_*_s, already standardized)
        Ytr_all = self.Ash_train_s.astype(np.float32)
        Yva_all = self.Ash_val_s.astype(np.float32)
        Yte_all = self.Ash_test_s.astype(np.float32)

        # ── Filter to images with valid shape teacher ──
        train_has = self.has_shape_train
        val_has   = self.has_shape_val   if hasattr(self, 'has_shape_val')   and self.has_shape_val   is not None else np.ones(len(Yva_all), dtype=bool)
        test_has  = self.has_shape_test  if hasattr(self, 'has_shape_test')  and self.has_shape_test  is not None else np.ones(len(Yte_all), dtype=bool)

        n_train_cov, n_val_cov, n_test_cov = int(train_has.sum()), int(val_has.sum()), int(test_has.sum())
        print(f"  Shape coverage: train {n_train_cov}/{len(Htr_all)} ({100*n_train_cov/max(len(Htr_all),1):.1f}%), "
              f"val {n_val_cov}/{len(Hva_all)} ({100*n_val_cov/max(len(Hva_all),1):.1f}%), "
              f"test {n_test_cov}/{len(Hte_all)} ({100*n_test_cov/max(len(Hte_all),1):.1f}%)")

        if n_train_cov == 0:
            raise RuntimeError("No training images have valid shape — cannot train CLIPShapeDecoder.")
        val_unavailable = False
        if n_val_cov == 0:
            print(f"  ⚠ No val images have valid shape — using train subset for validation "
                  f"(val metrics will be marked unavailable)")
            val_unavailable = True

        Htr = Htr_all[train_has]
        Ytr = Ytr_all[train_has]
        if val_unavailable:
            # Use training data for val — keeps training loop working;
            # val metrics marked as unavailable in decodability report.
            Hva = Htr
            Yva = Ytr
        else:
            Hva = Hva_all[val_has]
            Yva = Yva_all[val_has]

        decoder = CLIPShapeDecoder(Htr.shape[1], self.Ks).to(self.device)
        opt = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=wd)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.SmoothL1Loss()

        Xt_t = torch.tensor(Htr, dtype=torch.float32)
        Xv_t = torch.tensor(Hva, dtype=torch.float32)
        Xte_t = torch.tensor(Hte_all, dtype=torch.float32)  # keep all for precomputation
        yt_t = torch.tensor(Ytr, dtype=torch.float32)
        yv_t = torch.tensor(Yva, dtype=torch.float32)
        yte_t = torch.tensor(Yte_all, dtype=torch.float32)  # keep all for precomputation

        bv = float('inf'); bst = None; pat = 0
        for ep in range(epochs):
            decoder.train(); tl = 0.0
            perm = torch.randperm(len(Xt_t))
            for b in range(0, len(Xt_t), bs):
                idx = perm[b:b+bs]
                pred = decoder(Xt_t[idx].to(self.device))
                loss = loss_fn(pred, yt_t[idx].to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * len(idx)
            sch.step()
            decoder.eval()
            with torch.no_grad():
                vl = loss_fn(decoder(Xv_t.to(self.device)), yv_t.to(self.device)).item()
            if vl < bv: bv = vl; bst = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}; pat = 0
            else: pat += 1
            if (ep+1) % 50 == 0:
                print(f"  Ep {ep+1:3d}/{epochs} TrainL={tl/len(Xt_t):.4f} ValL={vl:.4f}  (best={bv:.4f})")
            if pat >= es: print(f"  Early stop @ {ep+1}"); break

        decoder.load_state_dict(bst)
        decoder.eval()
        self.shape_decoder = decoder
        print(f"  Best ValL={bv:.4f}")

        # Precompute CLIP-decoded shape features for ALL images
        # (decoder can run inference on any image; only training used covered subset)
        with torch.no_grad():
            Xt_all_t = torch.tensor(Htr_all, dtype=torch.float32)
            Xv_all_t = torch.tensor(Hva_all, dtype=torch.float32)
            Xte_all_t = torch.tensor(Hte_all, dtype=torch.float32)
            self.Sclip_train = decoder(Xt_all_t.to(self.device)).cpu().numpy().astype(np.float32)
            self.Sclip_val   = decoder(Xv_all_t.to(self.device)).cpu().numpy().astype(np.float32)
            self.Sclip_test  = decoder(Xte_all_t.to(self.device)).cpu().numpy().astype(np.float32)

        # ── Decodability report (covered subsets only) ──
        from sklearn.metrics import r2_score

        # Sclip_* now has ALL images; filter to covered for metrics
        Ytr_cov = Ytr_all[train_has]
        Yte_cov = Yte_all[test_has]
        Sc_tr_cov = self.Sclip_train[train_has]
        Sc_te_cov = self.Sclip_test[test_has]

        if val_unavailable:
            r2_val = float('nan')
            pearson_val = float('nan')
            pearson_each_val = np.full(self.Ks, float('nan'), dtype=np.float32)
            Yva_cov = None; Sc_va_cov = None
        else:
            Yva_cov = Yva_all[val_has] if hasattr(self, 'has_shape_val') and self.has_shape_val is not None else Yva_all
            Sc_va_cov = self.Sclip_val[val_has] if hasattr(self, 'has_shape_val') and self.has_shape_val is not None else self.Sclip_val
            r2_val   = r2_score(Yva_cov, Sc_va_cov)
            pearson_val, pearson_each_val = self._pearson_per_dim(Yva_cov, Sc_va_cov)

        r2_train = r2_score(Ytr_cov, Sc_tr_cov)
        r2_test  = r2_score(Yte_cov, Sc_te_cov)

        pearson_train, pearson_each_train = self._pearson_per_dim(Ytr_cov, Sc_tr_cov)
        pearson_test,  pearson_each_test  = self._pearson_per_dim(Yte_cov, Sc_te_cov)
        mse_test = ((Yte_cov - Sc_te_cov) ** 2).mean()

        # Zero-predictor baseline (predicting the mean of zero after scaling = all zeros)
        zero_pred_mse = float((Yte_cov ** 2).mean())

        # Also compute all-image metrics for comparison
        r2_test_all = r2_score(Yte_all, self.Sclip_test)
        mse_test_all = ((Yte_all - self.Sclip_test) ** 2).mean()

        self.shape_decodability = {
            "r2_train": r2_train, "r2_val": r2_val, "r2_test": r2_test,
            "pearson_train": pearson_train, "pearson_val": pearson_val,
            "pearson_test": pearson_test, "mse_test": mse_test,
            "pearson_each_train": pearson_each_train,
            "pearson_each_val": pearson_each_val,
            "pearson_each_test": pearson_each_test,
            "zero_baseline_mse": zero_pred_mse,
            "r2_test_all_images": r2_test_all, "mse_test_all_images": mse_test_all,
            "n_train_covered": n_train_cov, "n_train_total": len(Htr_all),
            "n_val_covered": n_val_cov, "n_val_total": len(Hva_all),
            "n_test_covered": n_test_cov, "n_test_total": len(Hte_all),
        }
        top_k = 5
        top_idx = np.argsort(pearson_each_test)[::-1][:top_k]
        low_idx = np.argsort(pearson_each_test)[:top_k]
        print(f"  Decodability (covered test, {n_test_cov}/{len(Hte_all)} images):")
        print(f"    R2={r2_test:.4f}  Pearson={pearson_test:.4f}  MSE={mse_test:.4f}")
        print(f"    Zero-predictor MSE: {zero_pred_mse:.4f}  (decoder/zero = {mse_test/max(zero_pred_mse,1e-8):.4f})")
        print(f"    ALL-images (incl. missing): R2={r2_test_all:.4f}  MSE={mse_test_all:.4f}")
        print(f"    Top-{top_k} decodable: {list(top_idx)} Pearson={[f'{pearson_each_test[i]:.3f}' for i in top_idx]}")
        print(f"    Low-{top_k} decodable: {list(low_idx)} Pearson={[f'{pearson_each_test[i]:.3f}' for i in low_idx]}")

        # Build validation-reliability weighted shape features
        self.build_shape_reliability_features(threshold=0.0, normalize=True, max_weight=2.0)

    def build_shape_reliability_features(self, threshold=0.0, normalize=True, max_weight=2.0):
        """
        Build validation-reliability weighted CLIP-decoded shape features.

        Reliability r_k = max(corr_val_k, 0), where corr_val_k is the per-prototype
        Pearson correlation between CLIP-decoded shape activation and SAM teacher
        on the VALIDATION set (covered subset only). Low-reliability prototypes are suppressed.

        IMPORTANT: uses VALIDATION (not test) decodability for weighting.
        Uses only images with valid shape teacher for correlation computation.
        """
        if not hasattr(self, "Sclip_train"):
            raise RuntimeError("build_shape_reliability_features() requires Sclip_*. "
                               "Call train_clip_shape_decoder() first.")
        if not hasattr(self, "Ash_val_s"):
            raise RuntimeError("build_shape_reliability_features() requires Ash_val_s. "
                               "Call setup_shape_teacher() first.")

        # Use val covered subset for reliability computation
        val_has = self.has_shape_val if hasattr(self, 'has_shape_val') and self.has_shape_val is not None else None
        if val_has is not None and val_has.any():
            Ash_val_cov = self.Ash_val_s[val_has]
            Sclip_val_cov = self.Sclip_val[val_has]  # Sclip is full-size now, filter to covered
        else:
            Ash_val_cov = self.Ash_val_s
            Sclip_val_cov = self.Sclip_val

        pearson_val, pearson_each_val = self._pearson_per_dim(Ash_val_cov, Sclip_val_cov)

        r = np.maximum(pearson_each_val, 0.0)
        r[r < threshold] = 0.0

        if normalize and r.mean() > 1e-8:
            r = r / (r.mean() + 1e-8)
        if max_weight is not None:
            r = np.clip(r, 0.0, max_weight)
        r = r.astype(np.float32)

        self.shape_reliability = r
        self.shape_reliability_val_pearson = pearson_each_val.astype(np.float32)

        self.Sclip_train_rel = (self.Sclip_train * r[None, :]).astype(np.float32)
        self.Sclip_val_rel   = (self.Sclip_val   * r[None, :]).astype(np.float32)
        self.Sclip_test_rel  = (self.Sclip_test  * r[None, :]).astype(np.float32)

        n_active = int((r > 0).sum())
        top = np.argsort(r)[::-1][:5]; low = np.argsort(r)[:5]
        print(f"[Shape Reliability]")
        print(f"  validation Pearson mean: {pearson_val:.4f}")
        print(f"  active reliable prototypes: {n_active}/{len(r)}")
        print(f"  reliability weight: min={r.min():.4f} mean={r.mean():.4f} max={r.max():.4f}")
        print(f"  Top reliable: {list(top)} weights={[f'{r[i]:.3f}' for i in top]}")
        print(f"  Low reliable: {list(low)} weights={[f'{r[i]:.3f}' for i in low]}")

    def explain_instance(self, split="test", idx=0, class_idx=None, topk=5,
                         shape_mode="decoded"):
        """
        Return concept-level decomposition for one sample.

        Contributions:  contribution = weight[class, k] * activation[k]

        Returns dict with:
          z_final, z_sem, z_ctx, z_shape
          top_sem_pos/neg, top_ctx_pos/neg, top_shape_pos/neg
          sem_contrib, ctx_contrib, shape_contrib
          ctx_neighbors (for top contextual concepts, which neighbors drove support)
          shape_reliability (if reliability weighting is active)
        """
        A_sem  = getattr(self.cbm, f"A_{split}_scaled")[idx]
        A_raw  = getattr(self.cbm, f"A_{split}")[idx]
        A_ctx  = getattr(self, f"Actx_{split}", None)
        A_shape = None

        if shape_mode == "decoded":
            A_shape = getattr(self, f"Sclip_{split}", None)
        elif shape_mode == "decoded_rel":
            A_shape = getattr(self, f"Sclip_{split}_rel", None)
        elif shape_mode == "oracle":
            A_shape = getattr(self, f"Ash_{split}_s", None)
        else:
            raise ValueError(f"Unknown shape_mode: {shape_mode}")

        if A_ctx is not None:
            A_ctx = A_ctx[idx]
        if A_shape is not None:
            A_shape = A_shape[idx]

        W_sem = self.cbm.model.weight.detach().cpu().numpy()
        b_sem = self.cbm.model.bias.detach().cpu().numpy()
        z_sem = W_sem @ A_sem + b_sem
        z = z_sem.copy()

        z_ctx = None
        if hasattr(self, "cb") and self.cb is not None and A_ctx is not None:
            W_ctx = self.cb.linear.weight.detach().cpu().numpy()
            b_ctx = self.cb.linear.bias.detach().cpu().numpy()
            z_ctx = W_ctx @ A_ctx + b_ctx
            z = z + self.eta_c * z_ctx

        z_shape = None
        if hasattr(self, "sb") and self.sb is not None and A_shape is not None:
            # Gate oracle shape contribution: if this image has no valid shape,
            # the ENTIRE residual should be zero (not W@0+b = b).
            oracle_missing = False
            if shape_mode == "oracle":
                has_shape_mask = getattr(self, f"has_shape_{split}", None)
                if has_shape_mask is not None and not has_shape_mask[idx]:
                    oracle_missing = True
            if oracle_missing:
                z_shape = np.zeros_like(z_sem)
            else:
                W_shape = self.sb.linear.weight.detach().cpu().numpy()
                b_shape = self.sb.linear.bias.detach().cpu().numpy()
                z_shape = W_shape @ A_shape + b_shape
            z = z + self.eta_s * z_shape

        if class_idx is None:
            class_idx = int(np.argmax(z))

        sem_contrib = W_sem[class_idx] * A_sem

        out = {
            "split": split, "idx": idx, "class_idx": class_idx,
            "z_final": z, "z_sem": z_sem,
            "top_sem_pos": np.argsort(sem_contrib)[::-1][:topk].tolist(),
            "top_sem_neg": np.argsort(sem_contrib)[:topk].tolist(),
            "sem_contrib": sem_contrib,
        }

        if z_ctx is not None:
            ctx_contrib = self.eta_c * W_ctx[class_idx] * A_ctx
            out["z_ctx"] = z_ctx
            out["top_ctx_pos"] = np.argsort(ctx_contrib)[::-1][:topk].tolist()
            out["top_ctx_neg"] = np.argsort(ctx_contrib)[:topk].tolist()
            out["ctx_contrib"] = ctx_contrib

            C = self.Co.detach().cpu().numpy()
            neighbor_info = {}
            for j in out["top_ctx_pos"]:
                support = A_raw * C[:, j]
                nbrs = np.argsort(support)[::-1][:topk]
                neighbor_info[int(j)] = [
                    {"neighbor": int(i), "a_i": float(A_raw[i]),
                     "C_ij": float(C[i, j]), "support": float(support[i])}
                    for i in nbrs
                ]
            out["ctx_neighbors"] = neighbor_info

        if z_shape is not None:
            shape_contrib = self.eta_s * W_shape[class_idx] * A_shape
            out["z_shape"] = z_shape
            out["top_shape_pos"] = np.argsort(shape_contrib)[::-1][:topk].tolist()
            out["top_shape_neg"] = np.argsort(shape_contrib)[:topk].tolist()
            out["shape_contrib"] = shape_contrib
            if hasattr(self, "shape_reliability"):
                out["shape_reliability"] = self.shape_reliability.tolist()

        return out

    def setup_contextual_cooccur(self, topk=20, topm_neighbors=None):
        """构造基于共现图的上下文特征 a_ctx = a ⊙ (a @ C).

        与旧版 setup_cooccur() 的四个关键区别:
          (1) 用 raw activation A_train (非 scaled) 判断概念是否活跃—— scaled 含负值,
              不适合做 "概念是否出现" 的阈值判断.
          (2) 每张图取 top-k 活跃概念, 而非全局 percentile—— 避免图像间激活分布差异
              导致某些图活跃概念过多或过少.
          (3) C[i,j] = P(j|i): 行归一化后, 支持向量为 a @ C (非 a @ C^T).
          (4) a_ctx = a ⊙ (a @ C): 逐元素乘法—— 概念自己的激活 × 邻居的上下文支持,
              使共现分支成为 "放大器" 而非 "发现器".
        """
        print(f"[Contextual Cooccur] topk={topk}, topm_neighbors={topm_neighbors}")
        K = self.K
        # ① 用 raw activation (非 scaled) —— softmax 输出, 值在 [0,1], 语义明确
        A_train_raw = self.cbm.A_train.astype(np.float32)
        co = np.zeros((K, K), dtype=np.float32)

        # ② 每张图取 top-k active concepts, 而非全局 percentile
        k_eff = min(topk, K)  # guard: topk 不能超过 K
        for row in A_train_raw:
            idx = np.argpartition(row, -k_eff)[-k_eff:]
            idx = idx[row[idx] > 0]  # softmax 下几乎不过滤, 保留以备未来改用非 softmax 激活
            for i in idx:
                for j in idx:
                    if i != j:
                        co[i, j] += 1.0  # binary counting (unweighted), 可解释性最强

        np.fill_diagonal(co, 0.0)

        # ③ Row normalization: C[i, j] = P(j | i)
        C = co / (co.sum(axis=1, keepdims=True) + 1e-8)

        # ④ 可选: 每个 concept 只保留 top-m neighbors, 增强稀疏性和可解释性
        if topm_neighbors is not None:
            m_eff = min(topm_neighbors, K - 1)  # guard: 邻居数不超过 K-1
            mask = np.zeros_like(C)
            for i in range(K):
                if C[i].sum() <= 0:  # 零行: 该概念从未与其他概念共现
                    continue
                nbrs = np.argpartition(C[i], -m_eff)[-m_eff:]
                mask[i, nbrs] = 1.0
            C = C * mask
            C = C / (C.sum(axis=1, keepdims=True) + 1e-8)

        self.Co = torch.tensor(C, dtype=torch.float32, device=self.device)

        # ⑤ 构造 contextual feature: a_ctx = a ⊙ (a @ C)
        def build_ctx(A_raw):
            support = A_raw @ C                               # (N, K): 邻居支持
            ctx = A_raw * support                             # 逐元素乘: 自己 × 邻居
            return ctx.astype(np.float32)

        self.Actx_train_raw = build_ctx(self.cbm.A_train.astype(np.float32))
        self.Actx_val_raw   = build_ctx(self.cbm.A_val.astype(np.float32))
        self.Actx_test_raw  = build_ctx(self.cbm.A_test.astype(np.float32))

        # ⑥ 对 contextual feature 做 StandardScaler (只 fit train)
        sc = StandardScaler().fit(self.Actx_train_raw)
        self.Actx_train = sc.transform(self.Actx_train_raw).astype(np.float32)
        self.Actx_val   = sc.transform(self.Actx_val_raw).astype(np.float32)
        self.Actx_test  = sc.transform(self.Actx_test_raw).astype(np.float32)
        self.ctx_scaler = sc
        # Free raw (unscaled) activations — no longer needed after cooccurrence setup
        del self.Actx_train_raw, self.Actx_val_raw, self.Actx_test_raw
        for s in ["train", "val", "test"]:
            # Free unscaled A matrices; only scaled versions needed for training
            a_raw = getattr(self.cbm, f"A_{s}", None)
            if a_raw is not None:
                setattr(self.cbm, f"A_{s}", None)

        print(f"  Actx_train={self.Actx_train.shape}")

    @torch.no_grad()
    def _eval_residual_behavior(self, A_sem, A_shape, y_oh, eta_s,
                                 shape_head=None, shape_mode="decoded",
                                 has_shape_mask=None, bs=64):
        """Compute per-instance residual metrics on a full split (batched).

        Args:
            shape_head: the CURRENTLY-TRAINING ShapeBranch (local ``sb`` variable).
                        Must be passed explicitly — do NOT rely on ``self.sb``
                        which is only assigned after training completes.

        Returns dict:
          res_mag, sem_mag, ratio, consistency, correction, harm,
          n_corrected, n_harmed, n_base_wrong, n_base_correct, accuracy
        """
        N = A_sem.shape[0]
        C = y_oh.shape[1]
        do_s = shape_head is not None
        gate_shape = (shape_mode == "oracle") and has_shape_mask is not None

        total_res_abs = 0.0
        total_sem_abs = 0.0
        n_consistent = 0
        n_corrected = 0
        n_harmed = 0
        n_base_correct = 0
        n_base_wrong = 0
        n_final_correct = 0

        for b in range(0, N, bs):
            be = min(b + bs, N)
            idx = np.arange(b, be)
            bx_sem = torch.tensor(A_sem[idx], dtype=torch.float32, device=self.device)
            by = torch.tensor(y_oh[idx], dtype=torch.float32, device=self.device)
            true_labels = by.argmax(dim=1)

            z_sem = self.cbm.model(bx_sem)
            if do_s and A_shape is not None:
                bx_shape = torch.tensor(A_shape[idx], dtype=torch.float32, device=self.device)
                z_shape = eta_s * shape_head(bx_shape)
                if gate_shape:
                    sm = torch.tensor(has_shape_mask[idx].astype(np.float32),
                                      device=self.device).unsqueeze(1)
                    z_shape = z_shape * sm
            else:
                z_shape = torch.zeros_like(z_sem)

            z_final = z_sem + z_shape

            base_pred = z_sem.argmax(dim=1)
            final_pred = z_final.argmax(dim=1)

            total_res_abs += z_shape.abs().sum().item()
            total_sem_abs += z_sem.abs().sum().item()
            n_consistent += (final_pred == base_pred).sum().item()

            base_wrong = (base_pred != true_labels)
            base_right = (base_pred == true_labels)
            n_base_wrong += base_wrong.sum().item()
            n_base_correct += base_right.sum().item()
            n_corrected += (base_wrong & (final_pred == true_labels)).sum().item()
            n_harmed += (base_right & (final_pred != true_labels)).sum().item()
            n_final_correct += (final_pred == true_labels).sum().item()

        res_mag = total_res_abs / (N * C) if C > 0 else 0.0
        sem_mag = total_sem_abs / (N * C) if C > 0 else 0.0
        ratio = res_mag / (sem_mag + 1e-8)
        consistency = n_consistent / N if N > 0 else 0.0
        correction = n_corrected / max(n_base_wrong, 1)
        harm = n_harmed / max(n_base_correct, 1)
        accuracy = n_final_correct / N * 100.0 if N > 0 else 0.0

        return {
            "res_mag": res_mag, "sem_mag": sem_mag, "ratio": ratio,
            "consistency": consistency, "correction": correction, "harm": harm,
            "n_corrected": n_corrected, "n_harmed": n_harmed,
            "n_base_wrong": n_base_wrong, "n_base_correct": n_base_correct,
            "accuracy": accuracy,
        }

    def train_residuals(self, eta_s=0.05, eta_c=0.05,
                        lr=1e-3, epochs=100, bs=32, es=30, tp=True,
                        use_shape=True, use_co=True,
                        shape_mode="decoded", pinggu=False):
        """
        Args:
            shape_mode: "decoded"     = SDR-CBM main method, CLIP-decoded shape.
                        "decoded_rel" = validation-reliability weighted CLIP-decoded shape.
                        "oracle"      = SAM-shape upper bound (test-time masks).
            pinggu: If True, record per-epoch val residual metrics and compute
                    final test correction/harm after best-ckpt restore.
        """
        self.cbm.freeze_semantic(); nc = self.nc; K = self.K
        do_s = use_shape and hasattr(self, "Ks")
        do_c = use_co and hasattr(self, "Co") and self.Co is not None

        # ── Shape has_valid mask (for gating residual on images without shape) ──
        if do_s:
            has_shape_train = getattr(self, 'has_shape_train', None)
            has_shape_val   = getattr(self, 'has_shape_val', None)
            has_shape_test  = getattr(self, 'has_shape_test', None)
            # Only gate in oracle mode (decoded mode uses CLIP decoder for all images)
            gate_shape = (shape_mode == "oracle") and has_shape_train is not None
            if gate_shape:
                hsm_tr = torch.tensor(has_shape_train.astype(np.float32)).unsqueeze(1)
                hsm_va = torch.tensor(has_shape_val.astype(np.float32)).unsqueeze(1) if has_shape_val is not None else None
                hsm_te = torch.tensor(has_shape_test.astype(np.float32)).unsqueeze(1) if has_shape_test is not None else None
            else:
                hsm_tr = hsm_va = hsm_te = None
        else:
            gate_shape = False
            hsm_tr = hsm_va = hsm_te = None

        # ── Shape input selection ──
        if do_s:
            if shape_mode == "decoded":
                if not hasattr(self, "Sclip_train"):
                    raise RuntimeError(
                        "shape_mode='decoded' requires CLIP-decoded shape features. "
                        "Call setup_shape_teacher(...) then train_clip_shape_decoder(...) first, "
                        "or use shape_mode='oracle' for upper-bound experiments.")
                Ast_raw = self.Sclip_train
                Asv_raw = self.Sclip_val
                Aste_raw = self.Sclip_test
            elif shape_mode == "decoded_rel":
                if not hasattr(self, "Sclip_train_rel"):
                    raise RuntimeError(
                        "shape_mode='decoded_rel' requires reliability-weighted shape features. "
                        "Call train_clip_shape_decoder() first (which auto-calls "
                        "build_shape_reliability_features()).")
                Ast_raw = self.Sclip_train_rel
                Asv_raw = self.Sclip_val_rel
                Aste_raw = self.Sclip_test_rel
            elif shape_mode == "oracle":
                Ast_raw = self.Ash_train_s
                Asv_raw = self.Ash_val_s
                Aste_raw = self.Ash_test_s
            else:
                raise ValueError(f"Unknown shape_mode: {shape_mode}")
        else:
            Ast_raw = Asv_raw = Aste_raw = None

        sb = ShapeBranch(self.Ks, nc).to(self.device) if do_s else None
        cb = CooccurBranch(K, nc).to(self.device) if do_c else None
        params = []
        if do_s: params += list(sb.parameters())
        if do_c: params += list(cb.parameters())
        if len(params) == 0:
            raise RuntimeError(
                "train_residuals() called with no active residual branch. "
                "Set use_shape=True and/or use_co=True.")
        opt = optim.Adam(params, lr=lr); sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        At = torch.tensor(self.cbm.A_train_scaled, dtype=torch.float32)
        Av = torch.tensor(self.cbm.A_val_scaled, dtype=torch.float32)
        Ate = torch.tensor(self.cbm.A_test_scaled, dtype=torch.float32)
        Ast = torch.tensor(Ast_raw, dtype=torch.float32) if do_s else None
        Asv = torch.tensor(Asv_raw, dtype=torch.float32) if do_s else None
        Aste = torch.tensor(Aste_raw, dtype=torch.float32) if do_s else None
        Act = torch.tensor(self.Actx_train, dtype=torch.float32) if do_c else None
        Acv = torch.tensor(self.Actx_val, dtype=torch.float32) if do_c else None
        Acte = torch.tensor(self.Actx_test, dtype=torch.float32) if do_c else None
        yt = torch.tensor(self.cbm.y_train_oh, dtype=torch.float32)
        yv = torch.tensor(self.cbm.y_val_oh, dtype=torch.float32)
        yte = torch.tensor(self.cbm.y_test_oh, dtype=torch.float32)
        def ceo(i, t): return -(t * F.log_softmax(i, dim=1)).sum(dim=1).mean()
        def ev(As, Ash, Aco, yt_, shape_mask=None):
            N = As.shape[0]; tlo = 0.0; co = 0; to_ = 0
            for b in range(0, N, bs):
                be = min(b + bs, N); idx = np.arange(b, be)
                bx = As[idx].to(self.device)
                bxs = Ash[idx].to(self.device) if Ash is not None else None
                bxc = Aco[idx].to(self.device) if Aco is not None else None
                by = yt_[idx].to(self.device)
                with torch.no_grad():
                    zn = self.cbm.model(bx)
                    if do_s and bxs is not None:
                        s_contrib = eta_s * sb(bxs)
                        if gate_shape and shape_mask is not None:
                            s_contrib = s_contrib * shape_mask[idx].to(self.device)
                        zn = zn + s_contrib
                    if do_c and bxc is not None: zn = zn + eta_c * cb(bxc)
                    l_ = ceo(zn, by); pr = zn.argmax(1); tl_l = by.argmax(1)
                    to_ += (be - b); co += (pr == tl_l).sum().item(); tlo += l_.item() * (be - b)
            return tlo / to_, 100.0 * co / to_
        bv = -1; bst = None; pat = 0; lam = 1e-4
        pinggu_history = [] if pinggu else None
        label = "+".join(filter(None, ["S" if do_s else "", "C" if do_c else ""]))
        print(f"[Residual] {label} eta_s={eta_s} eta_c={eta_c}")
        for ep in range(epochs):
            if do_s: sb.train()
            if do_c: cb.train()
            perm = torch.randperm(len(At)); tl = 0.0; corr = 0
            for b in range(0, len(At), bs):
                idx = perm[b:b+bs].numpy()
                bx = At[idx].to(self.device)
                bxs = Ast[idx].to(self.device) if Ast is not None else None
                bxc = Act[idx].to(self.device) if Act is not None else None
                by = yt[idx].to(self.device)
                opt.zero_grad()
                with torch.no_grad(): zn = self.cbm.model(bx)
                if do_s and bxs is not None:
                    s_contrib = eta_s * sb(bxs)
                    if gate_shape:
                        s_contrib = s_contrib * hsm_tr[idx].to(self.device)
                    zn = zn + s_contrib
                if do_c and bxc is not None: zn = zn + eta_c * cb(bxc)
                ce = ceo(zn, by); l1 = torch.tensor(0.0, device=self.device)
                if do_s: l1 = l1 + lam * torch.norm(sb.linear.weight, p=1)
                if do_c: l1 = l1 + lam * torch.norm(cb.linear.weight, p=1)
                loss = ce + l1
                loss.backward(); opt.step()
                tl += loss.item() * len(idx)
                with torch.no_grad(): corr += (zn.argmax(1) == by.argmax(1)).sum().item()
            tra = corr / len(At) * 100

            # ── Per-epoch evaluation ──
            if do_s: sb.eval()
            if do_c: cb.eval()
            vl, va = ev(Av, Asv, Acv, yv, hsm_va)

            if pinggu:
                # Full val-set residual magnitude (replaces At[:500] diagnostic)
                val_metrics = self._eval_residual_behavior(
                    Av.numpy(), Asv.numpy() if do_s and Asv is not None else None,
                    yv.numpy(), eta_s,
                    shape_head=sb if do_s else None,
                    shape_mode=shape_mode,
                    has_shape_mask=has_shape_val if gate_shape else None, bs=bs)
                pinggu_history.append({
                    "epoch": ep + 1, "eta_s": eta_s,
                    "val_acc": va, "val_loss": vl,
                    "res_mag": val_metrics["res_mag"],
                    "sem_mag": val_metrics["sem_mag"],
                    "ratio": val_metrics["ratio"],
                })
                if tp:
                    print(f"  Ep {ep+1:3d}/{epochs} Loss {tl/len(At):.2f} "
                          f"Train {tra:.2f}% Val {va:.2f}% "
                          f"|res|/|sem|={val_metrics['ratio']:.4f}")
            else:
                # Diagnostic: mean abs residual on first 500 train samples
                with torch.no_grad():
                    sample_n = min(500, len(At))
                    zs_ref = self.cbm.model(At[:sample_n].to(self.device))
                    zn_ref = zs_ref.clone()
                    if do_s:
                        s_diag = eta_s * sb(Ast[:sample_n].to(self.device))
                        if gate_shape:
                            s_diag = s_diag * hsm_tr[:sample_n].to(self.device)
                        zn_ref = zn_ref + s_diag
                    if do_c: zn_ref = zn_ref + eta_c * cb(Act[:sample_n].to(self.device))
                    res_mag = (zn_ref - zs_ref).abs().mean().item()
                tel, tea = ev(Ate, Aste, Acte, yte, hsm_te)
                if tp: print(f"  Ep {ep+1}/{epochs} Loss {tl/len(At):.2f} Train {tra:.2f}% Val {va:.2f}% Test {tea:.2f}% |res|={res_mag:.4f}")

            if va > bv:
                bv = va; bst = {}
                if do_s: bst["sb"] = {k: v.clone() for k, v in sb.state_dict().items()}
                if do_c: bst["cb"] = {k: v.clone() for k, v in cb.state_dict().items()}
                pat = 0
            else: pat += 1
            sch.step()
            if es and pat >= es: break

        # ── Restore best checkpoint ──
        if bst:
            if do_s and "sb" in bst: sb.load_state_dict(bst["sb"])
            if do_c and "cb" in bst: cb.load_state_dict(bst["cb"])
        self.sb = sb; self.cb = cb
        self.eta_s = eta_s; self.eta_c = eta_c

        # ── Final evaluation ──
        vl, va = ev(Av, Asv, Acv, yv, hsm_va)
        tel, tea = ev(Ate, Aste, Acte, yte, hsm_te)
        print(f"  Residual done. Val {va:.2f}% Test {tea:.2f}%")

        if pinggu:
            # Full test-set correction/harm metrics after best-ckpt restore
            test_metrics = self._eval_residual_behavior(
                Ate.numpy(),
                Aste.numpy() if do_s and Aste is not None else None,
                yte.numpy(), eta_s,
                shape_head=sb if do_s else None,
                shape_mode=shape_mode,
                has_shape_mask=has_shape_test if gate_shape else None, bs=bs)
            self._pinggu_final = test_metrics
            self._pinggu_history = pinggu_history
            print(f"  [pinggu] test consistency={test_metrics['consistency']:.4f} "
                  f"correction={test_metrics['correction']:.4f} "
                  f"harm={test_metrics['harm']:.4f} "
                  f"ratio={test_metrics['ratio']:.4f}")

        return va, tea
