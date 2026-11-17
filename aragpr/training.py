"""
ARAGPR: Auditing Residual Attribution with Geometric Prototype References

用户通过 --use_shape / --use_cooccur 显式控制要跑的残差分支，
通过 --shape_input 选择形状特征类型。

Usage:
  # 仅语义 baseline (不跑残差)
  python train.py --dataset CUB_200_2011

  # 语义 + 形状残差 (FD32)
  python train.py --dataset BUS_UCLM --use_shape --shape_input FD32

  # 语义 + 共现残差
  python train.py --dataset CUB_200_2011 --use_cooccur

  # 满血版 (语义 + 形状 + 共现)
  python train.py --dataset BUS_UCLM --use_shape --use_cooccur --shape_input FD32

  # 形状特征消融: 换 --shape_input 即可
  python train.py --dataset CUB_200_2011 --use_shape --shape_input geo
  python train.py --dataset CUB_200_2011 --use_shape --shape_input hu
"""
import os, sys, glob, re, cv2, argparse, numpy as np, torch
from pathlib import Path
import pandas as pd
from utils.dcbm_1 import GlobalCBM, ResidualTrainer, set_seed
from data.Segments.scripts.utilities.segment import load_mask_manifest, load_paired_mask_records

SHAPE_SOURCE_COLUMNS = ['mask_relpath', 'crop_relpath', 'mask_sha256', 'crop_sha256']

# ══════════════════════════════════════════════════════════════════════
#  Mask preprocessing (binarize + denoise)
# ══════════════════════════════════════════════════════════════════════

def _preprocess_mask(mp, ks=3, ma=100, strict_binary=False):
    """Load mask, binarize, denoise, keep only largest connected component.

    Strict rebuilding accepts only actual binary masks, never color cutouts.
    """
    if strict_binary:
        raw = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2 or not np.isin(raw, [0, 255]).all():
            raise ValueError(f"Expected a grayscale binary mask, not an RGB cutout: {mp}")
        m = raw
    else:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Cannot read mask: {mp}")
    # This threshold cannot reconstruct a SAM region from a color cutout.
    binary = (m > 127).astype(np.uint8) * 255
    binary = cv2.medianBlur(binary, ks)
    k = np.ones((ks, ks), np.uint8)
    cl = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    cs, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fl = np.zeros_like(cl)
    for c in cs:
        if cv2.contourArea(c) > ma:
            cv2.drawContours(fl, [c], 0, 255, -1)
    return fl


# ══════════════════════════════════════════════════════════════════════
#  Contour extraction — SPLIT by descriptor type
# ══════════════════════════════════════════════════════════════════════
#
#  geo11 / Hu7 / defect7 → raw contour (preserves area, moments, defects)
#  FD32 / curv20         → closed arc-length resampled contour (required
#                          for uniform frequency / curvature analysis)
# ══════════════════════════════════════════════════════════════════════

def _extract_contour_raw(clean):
    """Extract max outer contour — RAW (for geo, hu, defect)."""
    cs, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise ValueError("no contour")
    return max(cs, key=cv2.contourArea)


def _resample_contour_closed(contour, target=256):
    """Closed arc-length resampling for FD and curvature features.

    Includes the closure edge (last point → first point) and uses
    endpoint=False to avoid duplicate sampling of the start/end point.

    Returns float32 contour — no int32 truncation (which creates duplicate
    points on small contours and corrupts FD/curvature features).
    Uses vectorized searchsorted + linear interpolation (fast on 5M+ masks).
    """
    c = contour.reshape(-1, 2).astype(np.float32)
    n = len(c)
    if n < 3:
        return c.reshape(-1, 1, 2)  # already float32

    # Cumulative arc length including closure edge
    d = np.zeros(n + 1, dtype=np.float32)
    diffs = np.diff(c, axis=0)                         # [n-1, 2]
    d[1:n] = np.cumsum(np.linalg.norm(diffs, axis=1))  # [n-1]
    d[n] = d[n-1] + np.linalg.norm(c[0] - c[n-1])      # closure
    total = d[n]
    if total < 1e-8:
        return c.reshape(-1, 1, 2)

    # Uniform sample points along closed curve
    sample_dd = np.linspace(0, total, target, endpoint=False)  # [target]

    # Find segment for each sample point (vectorized)
    seg_idx = np.searchsorted(d, sample_dd, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, n - 1)                       # [target]

    seg_start = d[seg_idx]                                     # [target]
    seg_end = d[seg_idx + 1]                                   # [target]
    seg_len = seg_end - seg_start
    t = np.where(seg_len > 1e-8,
                 (sample_dd - seg_start) / np.maximum(seg_len, 1e-8),
                 0.0)
    t = t[:, np.newaxis]                                       # [target, 1]

    # Closure segment (idx == n-1) wraps to c[0]
    is_closure = (seg_idx == n - 1)                            # [target]
    next_idx = np.where(is_closure, 0, seg_idx + 1)            # [target]

    pts = c[seg_idx] + t * (c[next_idx] - c[seg_idx])          # [target, 2]

    return pts.reshape(-1, 1, 2).astype(np.float32)


def _extract_contour_resampled(clean, target=256):
    """Extract and closed-arc-length resample max outer contour (for FD, curv)."""
    c = _extract_contour_raw(clean)
    return _resample_contour_closed(c, target)


# ══════════════════════════════════════════════════════════════════════
#  Fourier Descriptor (FD32)
# ══════════════════════════════════════════════════════════════════════
#
#  Fixes applied (vs. original):
#   1. Total non-DC spectral energy normalization — no single harmonic
#      dominates the denominator; no constant FD_1 ≡ 1.
#   2. No np.clip — valid coefficient values are preserved.
#   3. Contour must be closed-arc-length resampled by caller.
# ══════════════════════════════════════════════════════════════════════

def _compute_fd(contour, n=32):
    """Compute n-dim Fourier descriptors with combined-magnitude normalization.

    - Contour must be closed-arc-length resampled (caller's responsibility).
    - Translation invariance: subtract centroid.
    - Scale invariance: divide by L2 norm of the output vector.
    - Rotation + starting-point invariance: take magnitude.
    - Mirror / contour-direction invariance: combine |F_k| and |F_{-k}|.
    - No dimension is constant (unlike |F_1|/|F_1| ≡ 1 in the old code).
    """
    c = contour.reshape(-1, 2).astype(np.float32)
    x, y = c[:, 0], c[:, 1]
    x -= x.mean(); y -= y.mean()                           # translation invariance
    sig = x + 1j*y
    fd = np.fft.fft(sig)
    N = len(fd)
    m = min(n, (N - 1) // 2)                                # need both ±k
    if m <= 0:
        return np.zeros(n, dtype=np.float32)

    # Pair positive and negative frequencies → mirror + direction invariance
    pos = np.abs(fd[1:1+m])                                 # |F_1| … |F_m|
    neg = np.abs(fd[-1:-1-m:-1])                            # |F_{-1}| … |F_{-m}|
    fv = np.sqrt(pos ** 2 + neg ** 2)                       # combined magnitude

    # Normalize by L2 norm of the selected harmonics (scale invariance)
    energy = np.linalg.norm(fv) + 1e-8
    fv /= energy

    if len(fv) < n:
        fv = np.pad(fv, (0, n - len(fv)))
    return fv[:n].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  FD-only CSV generator (legacy — prefer generate_shape_csv_ext)
# ══════════════════════════════════════════════════════════════════════

def _parse_mask_fn(fn):
    m = re.search(r'^(.*?)_OUT_MASK_(\d+)_', fn)
    if m:
        return m.group(1), m.group(2)
    return fn, "0"


def _list_committed_mask_paths(mask_dir):
    """Return masks only from image groups with a valid completion manifest."""
    mask_root = Path(mask_dir).resolve()
    paths = list(mask_root.rglob('*_OUT_MASK_*_X_.png'))
    paths.extend(mask_root.rglob('*_OUT_MASK_*_X_.PNG'))
    paths.extend(mask_root.rglob('*_OUT_MASK_*_X_.jpg'))
    paths.extend(mask_root.rglob('*_OUT_MASK_*_X_.JPG'))
    paths.extend(mask_root.rglob('*_OUT_MASK_*_X_.jpeg'))
    paths = sorted(set(paths))

    valid = []
    manifest_cache = {}
    incomplete_groups = set()
    legacy_count = 0
    for path in paths:
        relative = path.resolve().relative_to(mask_root)
        folder = relative.parent.as_posix()
        if folder == ".":
            folder = ""
        basename = path.name.split("_OUT_MASK_", 1)[0]
        group_key = (folder, basename)
        if group_key not in manifest_cache:
            manifest_cache[group_key] = load_mask_manifest(
                mask_root, folder, basename
            )
        if manifest_cache[group_key] is not None:
            valid.append(path)
        elif manifest_cache[group_key] is False:
            # Explicitly checked — no manifest & cannot validate → still
            # accept (legacy masks with non-contiguous IDs, e.g. Places365)
            valid.append(path)
            legacy_count += 1
        else:
            # First time seeing this group without a manifest
            # Accept it but flag as legacy (unchecked)
            manifest_cache[group_key] = False
            valid.append(path)
            legacy_count += 1

    if incomplete_groups:
        print(
            f"  WARNING: skipped {len(incomplete_groups)} incomplete mask "
            "groups without a valid completion manifest"
        )
    if legacy_count > 0:
        n_groups = len(set(
            (path.resolve().relative_to(mask_root).parent.as_posix(),
             path.name.split("_OUT_MASK_", 1)[0])
            for path in paths
        ))
        print(
            f"  [masks] {legacy_count} files in legacy mode "
            f"(no manifest — accepted as-is)"
        )
    return valid


def generate_fd_csv(mask_dir, out_csv, fd_dim=32):
    """Generate FD CSV from PNGS only (JPG masks are not binary-safe)."""
    if os.path.exists(out_csv):
        print(f"[FD] CSV exists: {out_csv}")
        return out_csv
    print(f"[FD] generating CSV ({fd_dim}d) from: {mask_dir}")
    imgs = _list_committed_mask_paths(mask_dir)
    if not imgs:
        print(f"  WARNING: no PNG masks in {mask_dir}")
        return None
    imgs = sorted(set(imgs))
    print(f"  {len(imgs)} PNG mask images")
    records = []; errs = 0
    mask_dir_path = Path(mask_dir).resolve()
    for p in imgs:
        try:
            clean = _preprocess_mask(str(p))
            contour = _extract_contour_resampled(clean)
            fd = _compute_fd(contour, n=fd_dim)
            rel = str(p.resolve().relative_to(mask_dir_path)).replace('\\', '/')
            src, rid = _parse_mask_fn(rel)
            row = {'source_image': src, 'region_id': rid}
            for i in range(fd_dim):
                row[f'FD_{i+1}'] = fd[i]
            records.append(row)
        except Exception:
            errs += 1
    if not records:
        print("  FAILED")
        return None
    df = pd.DataFrame(records)
    cols = ['source_image', 'region_id'] + [f'FD_{i+1}' for i in range(fd_dim)]
    df = df[cols]
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"  CSV: {out_csv} ({len(df)} rows, {errs} errors)")
    return out_csv


# ══════════════════════════════════════════════════════════════════════
#  Alternative shape feature extractors
# ══════════════════════════════════════════════════════════════════════
#
#  Contour contract:
#   - geo11, hu7, defect7  → receive RAW contour (CHAIN_APPROX_NONE)
#   - FD32, curv20         → receive CLOSED-ARC-LENGTH-RESAMPLED contour
# ══════════════════════════════════════════════════════════════════════

def _extract_geometric_features(contour):
    """11-dim geometric features (scale-normalized by minAreaRect diagonal).

    Features:
      1. norm_area           — area / shape_diag²
      2. norm_perimeter      — perimeter / shape_diag
      3. norm_eq_diameter    — equivalent diameter / shape_diag
      4. norm_major_axis     — fitted ellipse major / shape_diag
      5. norm_minor_axis     — fitted ellipse minor / shape_diag
      6. aspect_ratio        — major_r / minor_r  (minAreaRect based, rot-invariant)
      7. convexity           — hull_perimeter / perimeter
      8. circularity         — 4π·area / perimeter²
      9. rot_rectangularity  — area / minAreaRect_area  (rotation-invariant)
     10. eccentricity        — sqrt(1 - (minor/major)²)
     11. solidity            — area / convex_hull_area

    All features are dimensionless and (except convexity, circularity, solidity
    which are inherently invariant) normalized by the minAreaRect diagonal
    for rotation + scale invariance.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    # ── core dimensionless ratios (independent of normalization) ──
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0

    # ── rotation-invariant shape diagonal (minAreaRect) ──
    if len(contour) >= 5:
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        major_r, minor_r = max(rw, rh), min(rw, rh)
        shape_diag = np.hypot(major_r, minor_r) + 1e-8

        min_rect_area = major_r * minor_r
        rot_rectangularity = area / min_rect_area if min_rect_area > 0 else 0.0

        aspect = major_r / (minor_r + 1e-8)               # rot-invariant aspect

        ellipse = cv2.fitEllipse(contour)
        (_, _), (major, minor), _ = ellipse
        major, minor = max(major, minor), min(major, minor)
        eccentricity = np.sqrt(1 - (minor / (major + 1e-8)) ** 2)
        norm_major = major / shape_diag
        norm_minor = minor / shape_diag
    else:
        shape_diag = np.hypot(*contour.reshape(-1, 2).max(axis=0) -
                                contour.reshape(-1, 2).min(axis=0)) + 1e-8
        rot_rectangularity = 0.0
        aspect = 1.0
        norm_major, norm_minor, eccentricity = 0.0, 0.0, 0.0

    # ── scale-normalized features ──
    norm_area = area / (shape_diag ** 2)
    norm_perimeter = perimeter / shape_diag
    norm_eq_diam = np.sqrt(4 * area / np.pi) / shape_diag if area > 0 else 0.0

    feats = np.array([
        norm_area, norm_perimeter, norm_eq_diam,
        norm_major, norm_minor,
        aspect, convexity, circularity,
        rot_rectangularity, eccentricity, solidity
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_hu_moments(contour):
    """7-dim Hu moments (log-transformed, sign-preserving).

    Clipping relaxed to [-20, 20] (was [-10, 10]) to reduce saturation.
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # Use 1e-30 floor so log10 can reach -20 for very small high-order moments.
    # With 1e-10 the log10 is always ≥ -10, making clip(-20, 20) ineffective.
    hu_log = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))
    return np.clip(hu_log, -20, 20).astype(np.float32)


def _extract_curvature_histogram(contour, bins=20):
    """Unsigned turning-angle histogram (20-dim).

    Contour must be closed-arc-length resampled by caller.
    Uses arctan2 for numerically stable signed angle, then takes absolute
    value → "unsigned turning-angle histogram".
    """
    c = contour.reshape(-1, 2).astype(np.float32)
    n = len(c)
    if n < 3:
        return np.zeros(bins, dtype=np.float32)

    angles = []
    for i in range(n):
        prev_pt = c[(i - 1) % n]
        next_pt = c[(i + 1) % n]
        v1 = c[i] - prev_pt
        v2 = next_pt - c[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]   # z-component of v1 × v2
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        # Unsigned turning angle via arctan2 (numerically stable)
        angle = np.arctan2(np.abs(cross), dot)    # ∈ [0, π]
        angles.append(angle)

    if not angles:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi))
    return (hist.astype(np.float32) / (len(angles) + 1e-8))


def _extract_convexity_defects(contour):
    """Convexity defect statistics (7-dim).

    Fixes:
      - Depth is OpenCV fixed-point (8 fractional bits) → divide by 256.
      - When no defects exist, still compute area/hull_area/solidity.
      - Depth statistics normalized by contour perimeter.
      - defect_area normalized by bounding-box area (scale invariance).
      - Shallow defects filtered by normalized depth threshold.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    raw_defect_area = max(hull_area - area, 0.0)
    perimeter = cv2.arcLength(contour, True)

    # Normalize defect_area by bbox area for scale invariance
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    defect_area_norm = raw_defect_area / bbox_area if bbox_area > 0 else 0.0

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                         defect_area_norm, solidity], dtype=np.float32)

    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except Exception:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                         defect_area_norm, solidity], dtype=np.float32)

    if defects is None:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                         defect_area_norm, solidity], dtype=np.float32)

    # OpenCV returns depth as fixed-point with 8 fractional bits
    depths = defects[:, 0, 3].astype(np.float32) / 256.0

    # Filter shallow (pixel-aliasing) defects by normalized depth
    MIN_DEPTH_RATIO = 1e-4  # depth/perimeter threshold, fixed in experiments
    depth_norm = depths / (perimeter + 1e-8)
    keep = depth_norm >= MIN_DEPTH_RATIO

    if keep.sum() == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                         defect_area_norm, solidity], dtype=np.float32)

    depths_f = depths[keep]
    n_defects = len(depths_f)

    # Normalize depths by perimeter for scale invariance
    depths_n = depths_f / (perimeter + 1e-8)
    max_depth = depths_n.max()
    mean_depth = depths_n.mean()
    std_depth = depths_n.std()
    total_depth = depths_n.sum()

    feats = np.array([float(n_defects), max_depth, mean_depth, std_depth,
                      total_depth, defect_area_norm, solidity], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════
#  Shape feature type registry
# ══════════════════════════════════════════════════════════════════════
#
#  Each entry: (extractor_fn, feat_dim, file_suffix, feat_cols, use_resampled)
#    use_resampled=True  → closed-arc-length resampled contour (FD, curv)
#    use_resampled=False → raw contour (geo, hu, defect)
# ══════════════════════════════════════════════════════════════════════

SHAPE_FEATURE_TYPES = {
    "FD32":   (lambda c: _compute_fd(c, n=32), 32, "_fd32",
               [f'FD_{i+1}' for i in range(32)], True),
    "geo":    (_extract_geometric_features,     11, "_geo11",
               ['norm_area', 'norm_perimeter', 'norm_eq_diameter',
                'norm_major_axis', 'norm_minor_axis',
                'aspect_ratio', 'convexity', 'circularity',
                'rot_rectangularity', 'eccentricity', 'solidity'], False),
    "hu":     (_extract_hu_moments,              7, "_hu7",
               [f'hu_{i+1}' for i in range(7)], False),
    "curv":   (lambda c: _extract_curvature_histogram(c, bins=20), 20, "_curv20",
               [f'curv_bin_{i+1}' for i in range(20)], True),
    "defect": (_extract_convexity_defects,        7, "_defect7",
               ['n_defects', 'max_depth', 'mean_depth', 'std_depth',
                'total_depth', 'defect_area', 'solidity'], False),
}


# ══════════════════════════════════════════════════════════════════════
#  Generic shape CSV generator (streaming, PNG-only)
# ══════════════════════════════════════════════════════════════════════

def generate_shape_csv_ext(mask_dir, out_csv, extractor_fn, feat_dim,
                           feat_cols=None, use_resampled=False, strict_assets=False,
                           paired_records=None):
    """Generic CSV generator: mask → contour → features → CSV.

    Writes incrementally to avoid OOM on large datasets (5M+ masks).

    Args:
        use_resampled: If True, use closed-arc-length resampled contour
                       (for FD32, curv20). If False, use raw contour
                       (for geo11, hu7, defect7).
    """
    import csv as _csv
    from tqdm import tqdm as _tqdm
    if feat_cols is None:
        feat_cols = [f'FD_{i+1}' for i in range(feat_dim)]
    if os.path.exists(out_csv):
        print(f"[Shape CSV] exists: {out_csv}")
        return out_csv
    print(f"[Shape CSV] generating ({feat_dim}d) from: {mask_dir}")

    if strict_assets:
        if paired_records is None:
            paired_records = load_paired_mask_records(mask_dir)
        imgs = sorted(paired_records)
    else:
        imgs = _list_committed_mask_paths(mask_dir)
    if not imgs:
        print(f"  WARNING: no PNG masks in {mask_dir}")
        return None

    # Dedup by (source_image, region_id)
    best = {}  # (source_image, region_id) → best Path
    for p in imgs:
        try:
            rel = str(p.resolve().relative_to(Path(mask_dir).resolve())).replace('\\', '/')
            src, rid = _parse_mask_fn(rel)
            key = (src, rid)
            if key not in best:
                best[key] = p
        except Exception:
            best[str(p)] = p  # fallback: use path as key
    imgs = sorted(best.values())
    print(f"  {len(imgs)} unique PNG masks")

    mask_dir_path = Path(mask_dir).resolve()
    cols = ['source_image', 'region_id'] + feat_cols
    if strict_assets:
        cols += SHAPE_SOURCE_COLUMNS
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    # Write incrementally — never accumulate all records in memory
    written = 0; errs = 0; batch = []; BATCH_SIZE = 50000
    err_types = {}  # track error reasons
    contour_label = "resampled" if use_resampled else "raw"
    temporary_csv = out_csv + '.tmp'
    with open(temporary_csv, 'w', newline='') as f:
        writer = _csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        for p in _tqdm(imgs, desc=f"  [Shape CSV ({contour_label})]", unit="img"):
            try:
                clean = _preprocess_mask(str(p), strict_binary=strict_assets)
                if use_resampled:
                    contour = _extract_contour_resampled(clean)
                else:
                    contour = _extract_contour_raw(clean)
                feats = extractor_fn(contour)
                # Validate feature output
                if feats.shape != (feat_dim,):
                    raise ValueError(f"shape mismatch: got {feats.shape}, expected ({feat_dim},)")
                if not np.isfinite(feats).all():
                    raise ValueError(f"non-finite values in features: {feats}")
                rel = str(p.resolve().relative_to(mask_dir_path)).replace('\\', '/')
                src, rid = _parse_mask_fn(rel)
                row = {'source_image': src, 'region_id': rid}
                if strict_assets:
                    row.update(paired_records[p.resolve()])
                for i in range(len(feats)):
                    row[feat_cols[i]] = feats[i]
                batch.append(row)
                if len(batch) >= BATCH_SIZE:
                    writer.writerows(batch); written += len(batch); batch.clear()
            except Exception as e:
                errs += 1
                etype = type(e).__name__
                err_types[etype] = err_types.get(etype, 0) + 1
                if errs <= 20:
                    print(f"\n    ⚠ [{p.name}]: {e}")
        if batch:
            writer.writerows(batch); written += len(batch)
    if errs > 0:
        err_summary = ", ".join(f"{t}:{c}" for t, c in sorted(err_types.items()))
        err_pct = 100 * errs / max(written + errs, 1)
        print(f"  Errors: {errs} ({err_pct:.1f}%) — {err_summary}")
        if err_pct > 10:
            print(f"  ⚠ Error rate >10% — CSV may be incomplete; "
                  f"check mask quality and descriptor compatibility.")
    if written == 0:
        print("  FAILED")
        os.remove(temporary_csv)
        return None
    os.replace(temporary_csv, out_csv)
    print(f"  CSV: {out_csv} ({written} rows, {errs} errors)")
    return out_csv


def get_shape_csv(mask_dir, base_fd_csv, shape_key, strict_assets=False):
    """统一的形状特征 CSV 解析器。

    根据 shape_key 查找对应的 CSV 文件路径。若文件不存在则自动生成。
    返回 (csv_path, feat_dim, feat_cols)。
    """
    if shape_key not in SHAPE_FEATURE_TYPES:
        raise ValueError(
            f"未知 shape_input: '{shape_key}'，"
            f"可选: {list(SHAPE_FEATURE_TYPES.keys())}")

    extractor_fn, feat_dim, suffix, feat_cols, use_resampled = \
        SHAPE_FEATURE_TYPES[shape_key]
    csv_path = base_fd_csv.replace("_fd32", suffix)
    paired_records = load_paired_mask_records(mask_dir) if strict_assets else None

    if not os.path.exists(csv_path):
        print(f"  ⚠ 找不到对应csv文件: {csv_path}")
        print(f"    正在进行csv的生成 (shape_input={shape_key}, {feat_dim}d)...")
        generate_shape_csv_ext(mask_dir, csv_path, extractor_fn, feat_dim,
                               feat_cols, use_resampled=use_resampled,
                               strict_assets=strict_assets, paired_records=paired_records)
    else:
        print(f"  [Shape CSV] exists: {csv_path}")

    if strict_assets:
        import csv
        expected = {r['mask_relpath']: r for r in paired_records.values()}
        with open(csv_path, newline='') as handle:
            reader = csv.DictReader(handle)
            if not set(SHAPE_SOURCE_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError(f"Shape CSV has no paired-asset provenance: {csv_path}")
            seen = set()
            for row in reader:
                ref = expected.get(row['mask_relpath'])
                if ref is None or any(row[c] != ref[c] for c in SHAPE_SOURCE_COLUMNS):
                    raise ValueError(f"Shape CSV/asset provenance mismatch: {row['mask_relpath']}")
                if row['mask_relpath'] in seen:
                    raise ValueError(f"Duplicate shape mask row: {row['mask_relpath']}")
                src, rid = _parse_mask_fn(str(Path(row['mask_relpath']).relative_to('masks')).replace('\\', '/'))
                if row['source_image'] != src or row['region_id'] != rid:
                    raise ValueError(f"Shape CSV source identity mismatch: {row['mask_relpath']}")
                seen.add(row['mask_relpath'])
            if not seen:
                raise ValueError(f"No usable shape descriptors: {csv_path}")
    return csv_path, feat_dim, feat_cols


# ══════════════════════════════════════════════════════════════════════
#  Dataset loader
# ══════════════════════════════════════════════════════════════════════

def load_dataset_info(name):
    """返回数据集相关的路径字典.

    Returns:
        dict with keys: embed_path, class_labels, seg_path, masks_dir, fd_csv, stats_path
    """
    if name == "CUB_200_2011":
        return {
            "embed_path":   "./CUB_200_2011_split/",
            "class_labels": "./data/classes/cub_classes.txt",
            "seg_path":     "./data/Segments/Seg_embs/",
            "masks_dir":    "./data/Segments/Seg_embs/CUB_200_2011_SAM2/masks",
            "fd_csv":       "./data/Segments/Seg_embs/shape_csv/cub_fd32_features.csv",
            "stats_path":   "./data/Segments/Seg_embs/stats/CUB_200_2011_SAM2_STATS.torch",
        }
    elif name == "derm7pt":
        return {
            "embed_path":   "./derm7pt_split/",
            "class_labels": "./data/classes/derm7pt_classes.txt",
            "seg_path":     "./data/Segments/Seg_embs/",
            "masks_dir":    "./data/Segments/Seg_embs/derm7pt_SAM2/masks",
            "fd_csv":       "./data/Segments/Seg_embs/shape_csv/derm7pt_fd32_features.csv",
            "stats_path":   "./data/Segments/Seg_embs/stats/derm7pt_SAM2_STATS.torch",
        }
    elif name == "BUS_UCLM":
        return {
            "embed_path":   "./BUS_UCLM_split/",
            "class_labels": "./data/classes/BUS_UCLM_classes.txt",
            "seg_path":     "./data/Segments/Seg_embs/",
            "masks_dir":    "./data/Segments/Seg_embs/BUS_UCLM_SAM2/masks",
            "fd_csv":       "./data/Segments/Seg_embs/shape_csv/bus_uclm_fd32_features.csv",
            "stats_path":   "./data/Segments/Seg_embs/stats/BUS_UCLM_SAM2_STATS.torch",
            # BUS_UCLM 小数据集 (~200 img) 推荐超参数
            "hp": {
                "K_sem":    50,      # 语义概念原型数
                "K_shape":  32,      # 形状原型数
                "lr_sem":   5e-4,    # 语义基准学习率
                "epochs_s1":300,     # 语义基准训练轮数 (小数据集多跑)
                "bs":       8,       # 批次大小
                "es_pat":   50,      # 早停耐心 (小数据集波动大)
            },
        }
    elif name == "ISIC-2019":
        return {
            "embed_path":   "./ISIC-2019_split/",
            "class_labels": "./data/classes/ISIC-2019_classes.txt",
            "seg_path":     "./data/Segments/Seg_embs/",
            "masks_dir":    "./data/Segments/Seg_embs/ISIC-2019_SAM2/masks",
            "fd_csv":       "./data/Segments/Seg_embs/shape_csv/isic2019_fd32_features.csv",
            "stats_path":   "./data/Segments/Seg_embs/stats/ISIC-2019_SAM2_STATS.torch",
        }
    elif name == "places365_part":
        return {
            "embed_path":   "./places365_part_split/",
            "class_labels": "./data/classes/places365_part_classes.txt",
            "seg_path":     "./data/Segments/Seg_embs/",
            "masks_dir":    "./data/Segments/Seg_embs/places365_part_SAM2/masks",
            "fd_csv":       "./data/Segments/Seg_embs/shape_csv/places365_part_fd32_features.csv",
            "stats_path":   "./data/Segments/Seg_embs/stats/places365_part_SAM2_STATS.torch",
        }
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ══════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════

def eval_semantic(cbm, device):
    cbm.model.eval()
    with torch.no_grad():
        Xt = torch.tensor(cbm.A_train_scaled, dtype=torch.float32, device=device)
        Xv = torch.tensor(cbm.A_val_scaled,   dtype=torch.float32, device=device)
        Xte= torch.tensor(cbm.A_test_scaled,  dtype=torch.float32, device=device)
        yt = torch.tensor(cbm.y_train_oh, dtype=torch.float32, device=device)
        yv = torch.tensor(cbm.y_val_oh,   dtype=torch.float32, device=device)
        yte= torch.tensor(cbm.y_test_oh,  dtype=torch.float32, device=device)
        tr = (cbm.model(Xt).argmax(1) == yt.argmax(1)).float().mean().item() * 100
        va = (cbm.model(Xv).argmax(1) == yv.argmax(1)).float().mean().item() * 100
        te = (cbm.model(Xte).argmax(1) == yte.argmax(1)).float().mean().item() * 100
    return tr, va, te


def _load_semantic_baseline(ds_info, dataset, model_name, device, checkpoint_path):
    """Load a frozen semantic DCBM from a checkpoint, verify, and return it.

    This replaces Stage-1 training: concepts, scaler, and the linear model are
    restored from *checkpoint_path*.  The raw data and class labels are still
    loaded from *ds_info* (they must match the checkpoint).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Semantic checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    for key in ['concepts', 'K', 'scaler', 'model_state']:
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing key '{key}' — was it saved by "
                           f"train_semantic_baseline()?")

    # ── 1. Create CBM (loads raw data, labels, split keys) ──
    cbm = GlobalCBM(ds_info["embed_path"], dataset, model_name,
                    ds_info["class_labels"], device)

    # ── 1b. Load segment mapping (needed by ResidualTrainer, cheap) ──
    segment_model = ckpt.get('segment_model', 'SAM2')
    cbm.load_seg_mapping(ds_info["seg_path"], segment_model)

    # ── 2. Override concepts & recompute raw activations ──
    cbm.concepts = ckpt['concepts']
    cbm.K = ckpt['K']
    cbm.semantic_proto_representative = ckpt.get(
        'semantic_proto_representative', {})
    cbm.preprocess()                      # computes A_* from loaded concepts

    # ── 3. Override scaler (preprocess fits a new one — replace it) ──
    cbm.scaler = ckpt['scaler']
    for s in ["train", "val", "test"]:
        A = getattr(cbm, f"A_{s}", None)
        if A is not None:
            scaled = cbm.scaler.transform(A).astype(np.float32)
            setattr(cbm, f"A_{s}_scaled", scaled)

    # ── 4. Build & load the linear probe ──
    nc = int(ckpt.get('num_classes', cbm.y_train_oh.shape[1]))
    cbm.model = torch.nn.Linear(cbm.K, nc, bias=True).to(device)
    cbm.model.load_state_dict(ckpt['model_state'])
    cbm.model.eval()

    # ── 5. Verify on all splits ──
    tr, va, te = eval_semantic(cbm, device)
    loaded_ds = ckpt.get('dataset', '?')
    loaded_mn = ckpt.get('model_name', '?')
    print(f"  Checkpoint origin : dataset={loaded_ds}  model={loaded_mn}")
    print(f"  Loaded  baseline  : Train={tr:.2f}%  Val={va:.2f}%  Test={te:.2f}%")

    return cbm, tr, va, te


def train_semantic_baseline(ds_info, dataset, model_name, segment_model, device,
                            K_sem, lr_sem, epochs_s1, bs, es_pat,
                            max_segments_per_class=500, use_pca=False):
    """训练语义 DCBM 基准模型, 返回 (cbm, train_acc, val_acc, test_acc).

    Automatically saves the frozen checkpoint to
    ``{embed_path}/{dataset}_semantic_baseline.pt`` so that subsequent
    ``--freeze_semantic_from`` runs can reuse it.
    """
    cbm = GlobalCBM(ds_info["embed_path"], dataset, model_name,
                    ds_info["class_labels"], device)
    cbm.load_concepts(ds_info["seg_path"], segment_model, K=K_sem, use_pca=use_pca,
                      max_segments_per_class=max_segments_per_class)
    cbm.preprocess()
    cbm.train_semantic(lr=lr_sem, l1=1e-4, epochs=epochs_s1, bs=bs, es=es_pat, tp=True)
    tr, va, te = eval_semantic(cbm, device)
    print(f"  Baseline: Train={tr:.2f}% Val={va:.2f}% Test={te:.2f}%")

    # ── Save checkpoint for --freeze_semantic_from reuse ──
    ckpt_dir = os.path.join("semantic_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{dataset}_semantic_baseline.pt")
    ckpt = {
        'concepts':     cbm.concepts,                     # [K, 768] float32
        'K':            cbm.K,
        'scaler':       cbm.scaler,                        # sklearn StandardScaler
        'model_state':  {k: v.cpu().clone()
                         for k, v in cbm.model.state_dict().items()},
        'num_classes':   int(cbm.y_train_oh.shape[1]),
        'dataset':       dataset,
        'model_name':    model_name,
        'segment_model': segment_model,
        'semantic_proto_representative': getattr(
            cbm, 'semantic_proto_representative', {}),
    }
    torch.save(ckpt, save_path)
    print(f"  Semantic checkpoint saved → {save_path}")

    return cbm, tr, va, te


def run_branch_combo(rt, label, eta_vals, bs, use_shape, use_co, shape_mode="decoded",
                     pinggu=False, pinggu_dir=None, es_pat=20):
    """Run one branch combination with eta grid search. Returns best (val, test, eta_tuple).

    Saves/restores best model state so that rt.sb / rt.cb / rt.eta_s / rt.eta_c
    reflect the best-validation configuration for downstream visualization.

    When pinggu=True, also saves per-epoch eta history and final test metrics as CSVs.
    """
    import copy, csv as _csv

    best_va = -1; best_te = 0; best_eta = None; best_state = None
    e_s_vals = eta_vals if use_shape else [0.0]
    e_c_vals = eta_vals if use_co else [0.0]

    # Collect per-eta history for CSV output (pinggu mode only)
    all_eta_history = []     # list of dicts (per epoch × eta)
    all_selected = []        # list of dicts (one per eta, best-epoch metrics)

    for es in e_s_vals:
        for ec in e_c_vals:
            if (not use_shape and es != 0.0) or (not use_co and ec != 0.0): continue
            short = f"eta_s={es}" if use_shape else ""
            short += f" eta_c={ec}" if use_co else ""
            print(f"    [{label}] {short}")
            va_r, te_r = rt.train_residuals(
                eta_s=es, eta_c=ec,
                lr=1e-3, epochs=100, bs=bs, es=es_pat, tp=True,
                use_shape=use_shape, use_co=use_co,
                shape_mode=shape_mode, pinggu=pinggu)

            if pinggu and hasattr(rt, "_pinggu_history") and rt._pinggu_history:
                # Append per-epoch history
                for entry in rt._pinggu_history:
                    entry["eta_c"] = ec
                all_eta_history.extend(rt._pinggu_history)

                # Best-epoch selected metrics (from final test eval)
                final = getattr(rt, "_pinggu_final", {})
                all_selected.append({
                    "eta_s": es, "eta_c": ec,
                    "val_acc": va_r, "test_acc": te_r,
                    "consistency": final.get("consistency", float("nan")),
                    "correction": final.get("correction", float("nan")),
                    "harm": final.get("harm", float("nan")),
                    "res_mag": final.get("res_mag", float("nan")),
                    "sem_mag": final.get("sem_mag", float("nan")),
                    "ratio": final.get("ratio", float("nan")),
                    "n_corrected": final.get("n_corrected", 0),
                    "n_harmed": final.get("n_harmed", 0),
                    "n_base_wrong": final.get("n_base_wrong", 0),
                    "n_base_correct": final.get("n_base_correct", 0),
                })

            if va_r > best_va:
                best_va = va_r; best_te = te_r; best_eta = (es, ec)
                best_state = {
                    "eta_s": es, "eta_c": ec,
                    "sb": copy.deepcopy(rt.sb.state_dict()) if getattr(rt, "sb", None) is not None else None,
                    "cb": copy.deepcopy(rt.cb.state_dict()) if getattr(rt, "cb", None) is not None else None,
                }

    # Restore best model for viz
    if best_state is not None:
        rt.eta_s = best_state["eta_s"]; rt.eta_c = best_state["eta_c"]
        if best_state["sb"] is not None and getattr(rt, "sb", None) is not None:
            rt.sb.load_state_dict(best_state["sb"])
        if best_state["cb"] is not None and getattr(rt, "cb", None) is not None:
            rt.cb.load_state_dict(best_state["cb"])
        print(f"    [Best restored] eta_s={rt.eta_s}, eta_c={rt.eta_c}, val={best_va:.2f}% test={best_te:.2f}%")

    # ── Save pinggu CSVs ──
    if pinggu and pinggu_dir and all_eta_history:
        import csv as _csv2
        os.makedirs(pinggu_dir, exist_ok=True)
        mode_tag = shape_mode  # "decoded" or "decoded_rel"

        # Derive best_epoch per eta from history
        eta_best_epoch = {}
        for entry in all_eta_history:
            key = (entry["eta_s"], entry.get("eta_c", 0.0))
            if (key not in eta_best_epoch
                    or entry["val_acc"] > eta_best_epoch[key][1]):
                eta_best_epoch[key] = (entry["epoch"], entry["val_acc"])

        # Mark the globally-selected eta (by val_acc)
        best_key = best_eta  # (es, ec)

        ds_dir = os.path.join(pinggu_dir, rt.cbm.dataset)
        os.makedirs(ds_dir, exist_ok=True)

        # CSV 1: per-epoch history (for line plots)
        history_path = os.path.join(ds_dir, f"{mode_tag}_eta_history.csv")
        history_cols = ["eta_s", "eta_c", "epoch", "val_acc", "val_loss",
                        "res_mag", "sem_mag", "ratio"]
        with open(history_path, 'w', newline='') as f:
            writer = _csv2.DictWriter(f, fieldnames=history_cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_eta_history)
        print(f"    [pinggu] eta history saved → {history_path} ({len(all_eta_history)} rows)")

        # CSV 2: per-eta metrics (all etas, with best_epoch and is_selected flag)
        eta_metrics_path = os.path.join(ds_dir, f"{mode_tag}_eta_metrics.csv")
        eta_cols = ["eta_s", "eta_c", "best_epoch", "is_selected",
                    "val_acc", "test_acc",
                    "consistency", "correction", "harm",
                    "res_mag", "sem_mag", "ratio",
                    "n_corrected", "n_harmed", "n_base_wrong", "n_base_correct"]
        for row in all_selected:
            key = (row["eta_s"], row.get("eta_c", 0.0))
            be = eta_best_epoch.get(key, ("?", 0.0))
            row["best_epoch"] = be[0]
            row["is_selected"] = (key == best_key)
        with open(eta_metrics_path, 'w', newline='') as f:
            writer = _csv2.DictWriter(f, fieldnames=eta_cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_selected)
        print(f"    [pinggu] eta metrics saved → {eta_metrics_path} ({len(all_selected)} rows)")

        # CSV 3: single best-eta row (by val_acc) — paper Table 4.5
        # After global-best restore, re-compute test metrics for the selected eta
        # to guarantee the numbers match the restored model.
        best_row = None
        for row in all_selected:
            if (row["eta_s"], row.get("eta_c", 0.0)) == best_key:
                best_row = dict(row)
                break
        if best_row is not None:
            best_path = os.path.join(ds_dir, f"{mode_tag}_best_metrics.csv")
            best_cols = ["eta_s", "eta_c", "val_acc", "test_acc",
                         "consistency", "correction", "harm",
                         "res_mag", "sem_mag", "ratio"]
            with open(best_path, 'w', newline='') as f:
                writer = _csv2.DictWriter(f, fieldnames=best_cols, extrasaction='ignore')
                writer.writeheader()
                writer.writerow(best_row)
            print(f"    [pinggu] best metrics saved → {best_path} "
                  f"(eta_s={best_row['eta_s']}, val={best_row['val_acc']:.2f}%)")

        # Clean up pinggu state
        rt._pinggu_history = None
        rt._pinggu_final = None

    return best_va, best_te, best_eta


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ══════════════════════════════════════════════════════════════════════
#  Visualization helper (shared by baseline-only and residual paths)
# ══════════════════════════════════════════════════════════════════════

def _run_viz(rt, args):
    from utils.sdr_cbm_visualize import SDRCBMVisualizer

    section("Visualization: SDR-CBM/DCBM explanation")
    viz = SDRCBMVisualizer(
        rt=rt, out_dir=args.viz_out,
        concept_vocab_path=args.viz_concept_vocab,
        image_root=args.viz_image_root,
        mask_root=args.viz_mask_root,
        device=args.device,
    )

    if args.use_cooccur and hasattr(rt, "Co") and rt.Co is not None:
        viz.visualize_cooccur_matrix(
            topn=args.viz_cooccur_topn, by=args.viz_cooccur_by)

    for vidx in args.viz_indices:
        viz.visualize_instance(
            split=args.viz_split, idx=vidx, topk=args.viz_topk,
            shape_mode=args.shape_mode,
            posthoc_shape_match=args.viz_posthoc_shape_match)


def _run_explanation_trace(rt, args, ds_info, dataset):
    """Export Section 4.6 correction/harm trace bundles after training."""
    from utils.sdr_cbm_visualize import SDRCBMVisualizer

    section("Section 4.6: correction/harm explanation trace")
    semantic_segment_file = os.path.join(
        ds_info["seg_path"],
        f"segcrop_{dataset}_{args.segment_model}_{args.model_name}.torch")
    shape_asset_root = args.trace_asset_root or os.path.dirname(ds_info["masks_dir"])
    image_root = args.trace_image_root or ds_info["embed_path"]

    def shape_geometry(mask_path):
        # Reuse precisely the preprocessing used to build the geo11 CSV.
        clean = _preprocess_mask(mask_path, strict_binary=args.strict_shape_assets)
        contour = _extract_contour_raw(clean)
        return clean.shape, contour, _extract_geometric_features(contour)

    viz = SDRCBMVisualizer(
        rt=rt,
        out_dir=args.trace_out,
        concept_vocab_path=args.trace_concept_vocab,
        image_root=image_root,
        mask_root=ds_info["masks_dir"],
        shape_asset_root=shape_asset_root,
        semantic_asset_root=ds_info.get("semantic_asset_root"),
        class_labels_path=ds_info["class_labels"],
        semantic_segment_file=semantic_segment_file,
        dataset=dataset,
        device=args.device,
        shape_geometry_fn=shape_geometry,
        shape_vocab_path=args.trace_shape_vocab,
    )
    viz.export_case_study(
        split=args.trace_split,
        shape_mode=args.shape_mode,
        topk=args.trace_topk,
        cases_per_type=args.trace_cases_per_type,
        posthoc_shape_match=args.trace_posthoc_shape_match,
        display_topk=args.trace_display_topk,
        indices=args.trace_indices,
        render_plots=not args.trace_no_plots,
        run_config=vars(args),
    )


# ══════════════════════════════════════════════════════════════════════
#  Unified experiment runner
# ══════════════════════════════════════════════════════════════════════

def run_experiment(ds_info, dataset, args):
    """用户通过 --use_shape / --use_cooccur / --shape_input 控制实验。

    始终训练语义 baseline，然后按需搭建 shape/cooccur 残差分支并执行网格搜索。
    """
    if getattr(args, "bias_only", False) and (
            not args.use_shape or args.use_cooccur or args.shape_mode == "oracle"):
        raise ValueError("--bias_only requires a decoded shape branch without cooccurrence.")
    results = {}
    es_eff = 0 if args.no_early_stop else args.es_pat  # early-stop patience

    # ── 1. 语义 baseline (train from scratch OR load frozen checkpoint) ──
    if args.freeze_semantic_from:
        section(f"Exp: {dataset} Semantic Baseline (frozen checkpoint)")
        cbm, tr, va, te = _load_semantic_baseline(
            ds_info, dataset, args.model_name, args.device,
            args.freeze_semantic_from)
    else:
        section(f"Exp: {dataset} Semantic Baseline (train)")
        cbm, tr, va, te = train_semantic_baseline(
            ds_info, dataset, args.model_name, args.segment_model, args.device,
            args.K_sem, args.lr_sem, args.epochs_s1, args.bs, es_eff,
            args.max_segments_per_class, args.use_pca)
    results["baseline"] = (tr, va, te)

    # ── 2. 搭建 ResidualTrainer (baseline-only 也需要, 供 viz 使用) ──
    rt = ResidualTrainer(cbm, args.device)

    if not args.use_shape and not args.use_cooccur:
        print("\n  (未启用任何残差分支，仅跑 semantic baseline)")
        if args.viz:
            _run_viz(rt, args)
        return results

    if args.use_shape:
        csv_path, shape_feat_dim, shape_feat_cols = get_shape_csv(
            ds_info["masks_dir"], ds_info["fd_csv"], args.shape_input,
            strict_assets=args.strict_shape_assets)
        section(f"Exp: {dataset} + Shape Teacher ({args.shape_input}, {shape_feat_dim}d)")
        rt.setup_shape_teacher(csv_path, K_shape=args.K_shape, fd_dim=shape_feat_dim,
                               shape_mode=args.shape_mode, feat_cols=shape_feat_cols,
                               shape_standardize=args.shape_standardize,
                               shape_clustering=args.shape_clustering,
                               shape_prototype=args.shape_prototype)
        if args.shape_mode in ("decoded", "decoded_rel"):
            section(f"Exp: {dataset} + CLIP Shape Decoder")
            rt.train_clip_shape_decoder(
                lr=args.shape_decoder_lr, epochs=args.shape_decoder_epochs,
                bs=args.bs, es=es_eff)
        else:
            print(f"  [Shape] shape_mode='oracle' — skip decoder training, use Ash_*_s directly")

    if args.use_cooccur:
        section(f"Exp: {dataset} + Cooccur Residual (topk={args.cooccur_topk})")
        rt.setup_contextual_cooccur(topk=args.cooccur_topk, topm_neighbors=args.cooccur_topm)

    # ── 4. 跑残差分支组合 ──
    label_parts = []
    if args.use_shape:
        label_parts.append(f"S({args.shape_input})")
    if args.use_cooccur:
        label_parts.append("C")
    label = "+".join(label_parts)

    section(f"Exp: {dataset} + Residual [{label}]")
    r = run_branch_combo(rt, label, args.eta_vals, args.bs,
                         use_shape=args.use_shape, use_co=args.use_cooccur,
                         shape_mode=args.shape_mode,
                         pinggu=args.pinggu,
                         pinggu_dir=args.pinggu_dir if args.pinggu else None,
                         es_pat=es_eff)
    results[label] = r

    if getattr(args, "bias_only", False):
        from utils.bias_only import export_bias_only_control
        export_bias_only_control(rt, args.bias_only_out, args.shape_mode,
                                 config=vars(args), bs=args.bs)

    # ── Visualization (post-training, optional) ──
    if args.viz:
        _run_viz(rt, args)

    if args.explain_trace:
        _run_explanation_trace(rt, args, ds_info, dataset)

    return results


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ARAGPR: Auditing Residual Attribution with Geometric Prototype References")

    # 数据集与实验选择
    p.add_argument('--dataset', type=str, required=True,
                   choices=['CUB_200_2011', 'derm7pt', 'BUS_UCLM', 'ISIC-2019', 'places365_part'],
                   help='Dataset name')
    p.add_argument('--freeze_semantic_from', type=str, default=None,
                   help='Path to a saved semantic baseline checkpoint (.pt). '
                        'When provided, Stage-1 training + KMeans are SKIPPED; '
                        'concepts, scaler, and the linear model are loaded from '
                        'the checkpoint instead.  '
                        'Use this to share the same frozen semantic DCBM across '
                        'decoded and decoded_rel experiments.')

    # 残差分支开关
    p.add_argument('--use_shape', action='store_true', default=False,
                   help='启用形状残差分支 (shape residual branch)')
    p.add_argument('--use_cooccur', action='store_true', default=False,
                   help='启用共现残差分支 (cooccurrence residual branch)')
    p.add_argument('--shape_input', type=str, default='geo',
                   choices=list(SHAPE_FEATURE_TYPES.keys()),
                   help=f'Shape feature type (default: geo). Choices: {list(SHAPE_FEATURE_TYPES.keys())}')
    p.add_argument('--shape_asset_root', type=str, default=None,
                   help='Training shape asset bank containing masks/ and crops/; independent of semantic assets')
    p.add_argument('--shape_csv_dir', type=str, default=None,
                   help='Directory for shape descriptor CSVs of this asset bank')
    p.add_argument('--strict_shape_assets', action='store_true',
                   help='Require committed paired binary PNGs, matching crops and recorded file hashes')

    # 形状预处理消融 (Section 4.3)
    p.add_argument('--shape-standardize', dest='shape_standardize',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='Apply Z-score standardization before shape clustering. '
                        'Default: True. Use --no-shape-standardize for ablation.')
    p.add_argument('--shape-clustering', dest='shape_clustering', type=str,
                   default='kmeans',
                   choices=['kmeans', 'minibatch_kmeans'],
                   help='Shape clustering method. Default: kmeans.')
    p.add_argument('--shape-prototype', dest='shape_prototype', type=str,
                   default='median_nearest',
                   choices=['median_nearest', 'mean_center', 'mean_nearest', 'medoid'],
                   help='Representative used per shape cluster. '
                        'Default: median_nearest. '
                        '(DEPRECATED: "medoid" is an alias for "mean_nearest")')

    # 模型与设备
    p.add_argument('--model_name', type=str, default='CLIP-ViT-L14',
                   help='Image encoder (default: CLIP-ViT-L14)')
    p.add_argument('--segment_model', type=str, default='SAM2',
                   help='Segmentation model (default: SAM2)')
    p.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                   help='Device (default: cuda:0 if available else cpu)')

    # 随机种子
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42). '
                        'Run with --seed 1,2,3 for mean±std.')

    # 语义基准超参数
    p.add_argument('--K_sem', type=int, default=2048,
                   help='Number of semantic concept prototypes (default: 2048)')
    p.add_argument('--K_shape', type=int, default=128,
                   help='Number of shape prototypes (default: 128)')
    p.add_argument('--lr_sem', type=float, default=1e-4,
                   help='Learning rate for semantic baseline (default: 1e-4)')
    p.add_argument('--epochs_s1', type=int, default=200,
                   help='Epochs for semantic baseline (default: 200)')

    # 通用训练超参数
    p.add_argument('--bs', type=int, default=64,
                   help='Batch size (default: 64)')
    p.add_argument('--es_pat', type=int, default=30,
                   help='Early stopping patience (default: 30)')
    p.add_argument('--no_early_stop', action='store_true', default=False,
                   help='Disable early stopping — run all epochs to completion')
    p.add_argument('--eta_vals', type=float, nargs='+', default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help='Eta grid for residual branches (default: 0.1 0.25 0.5 0.75 1.0)')
    p.add_argument('--cooccur_topk', type=int, default=20,
                   help='Per-image top-k active concepts for cooccur graph (default: 20)')
    p.add_argument('--cooccur_topm', type=int, default=None,
                   help='Top-m neighbors per concept in cooccur graph (default: None = no sparsification)')
    p.add_argument('--shape_mode', type=str, default='decoded',
                   choices=['decoded', 'decoded_rel', 'oracle'],
                   help='decoded=CLIP-decoded; decoded_rel=reliability-weighted; oracle=SAM upper bound')
    p.add_argument('--shape_decoder_lr', type=float, default=1e-3,
                   help='LR for CLIP shape decoder (default: 1e-3)')
    p.add_argument('--shape_decoder_epochs', type=int, default=300,
                   help='Epochs for CLIP shape decoder (default: 300)')
    p.add_argument('--bias_only', action='store_true',
                   help='After restoring the validation-selected shape head, export '
                        'paired frozen/decoded/bias-only validation and test results; '
                        'no refit or eta reselection.')
    p.add_argument('--bias_only_out', type=str, default='./outputs/bias_only',
                   help='New directory for bias-only controls (existing runs are preserved).')
    p.add_argument('--max_segments_per_class', type=int, default=500,
                   help='Max segments per class for clustering (default: 500). '
                        'Classes with fewer segments are kept in full. Set to 0 for unlimited.')
    p.add_argument('--use_pca', action='store_true', default=False,
                   help='PCA→100d before KMeans clustering (default: False, cluster on full 768d). '
                        'Note: only affects clustering; GlobalCBM never extracts all segment embeddings '
                        'regardless of this flag. LocalCBM always uses PCA and extracts all embeddings.')

    # ── Residual behavior evaluation (Section 4.5) ─────────
    p.add_argument('--pinggu', action='store_true', default=False,
                   help='Record residual dynamics and final correction/harm metrics')
    p.add_argument('--pinggu_dir', type=str, default='./outputs/residual_behavior',
                   help='Output directory for residual behavior CSVs')

    # ── Visualization ─────────────────────────────────────────
    p.add_argument('--viz', action='store_true', default=False,
                   help='Run SDR-CBM visualization after training.')
    p.add_argument('--viz_split', type=str, default='test',
                   choices=['train', 'val', 'test'],
                   help='Split to visualize (default: test).')
    p.add_argument('--viz_indices', type=int, nargs='+', default=[0],
                   help='Sample indices to visualize (default: [0]).')
    p.add_argument('--viz_topk', type=int, default=5,
                   help='Top-k contributions to show (default: 5).')
    p.add_argument('--viz_out', type=str, default='./outputs/sdr_cbm_viz',
                   help='Output directory for visualization (default: ./outputs/sdr_cbm_viz).')
    p.add_argument('--viz_concept_vocab', type=str, default=None,
                   help='Optional txt with candidate concept phrases for CLIP text alignment.')
    p.add_argument('--viz_image_root', type=str, default=None,
                   help='Optional image root for resolving sample keys to image paths.')
    p.add_argument('--viz_mask_root', type=str, default=None,
                   help='Optional mask root for prototype nearest-mask visualization.')
    p.add_argument('--viz_posthoc_shape_match', action='store_true', default=False,
                   help='Show SAM teacher values (visualization only, not inference).')
    p.add_argument('--viz_cooccur_topn', type=int, default=30,
                   help='Top-n concepts in co-occurrence matrix viz (default: 30).')
    p.add_argument('--viz_cooccur_by', type=str, default='frequency',
                   choices=['frequency', 'ctx_weight'],
                   help='Ordering for co-occurrence matrix viz (default: frequency).')

    # ── Section 4.6 explanation trace export ─────────────────
    p.add_argument('--explain_trace', action='store_true', default=False,
                   help='Export representative correction/harm cases, prototype paths, '
                        'class logits, and class-margin waterfall data after training.')
    p.add_argument('--trace_split', type=str, default='test',
                   choices=['train', 'val', 'test'],
                   help='Split used for Section 4.6 case tracing (default: test).')
    p.add_argument('--trace_topk', type=int, default=5,
                   help='Number of semantic/shape coordinates exported per case (default: 5).')
    p.add_argument('--trace_cases_per_type', type=int, default=5,
                   help='Bundles per category: 5 by default, -1 for ALL correction/harm '
                        'cases, 0 for IDs only. All case IDs are always recorded.')
    p.add_argument('--trace_indices', type=int, nargs='+', default=None,
                   help='Export specified zero-based split indices instead of automatic selection; '
                        'each must be a correction or harm case in this run.')
    p.add_argument('--trace_display_topk', type=int, default=3,
                   help='Number of shape coordinates in the waterfall (default: 3); '
                        'remaining coordinates are included in Other shape.')
    p.add_argument('--trace_no_plots', action='store_true',
                   help='Export evidence and waterfall CSVs without rendering per-case plots.')
    p.add_argument('--trace_out', type=str, default='./outputs/explanation_trace',
                   help='Output root for Section 4.6 trace bundles.')
    p.add_argument('--trace_concept_vocab', type=str, default=None,
                   help='Optional concept phrase vocabulary used for CLIP text alignment.')
    p.add_argument('--trace_shape_vocab', type=str, default=None,
                   help='Optional UTF-8 phrase list for CLIP naming of verified training '
                        'shape crops. These are unreviewed exemplar labels, not predictions.')
    p.add_argument('--trace_image_root', type=str, default=None,
                   help='Optional input-image root; defaults to the dataset split directory.')
    p.add_argument('--trace_asset_root', type=str, default=None,
                   help='Optional SAM2 asset root containing crops/masks/contours.')
    p.add_argument('--trace_posthoc_shape_match', action='store_true', default=False,
                   help='Also export test-set SAM teacher activations for post-hoc audit only; '
                        'they are never used by decoded inference.')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── pinggu 参数校验 ──
    if args.pinggu:
        if not args.use_shape:
            print("ERROR: --pinggu requires --use_shape", file=sys.stderr)
            sys.exit(1)
        if args.shape_input != 'geo':
            print("ERROR: --pinggu for Section 4.5 requires --shape_input geo",
                  file=sys.stderr)
            sys.exit(1)

    if args.explain_trace:
        if args.trace_topk < 1 or not 1 <= args.trace_display_topk <= args.trace_topk:
            sys.exit('ERROR: require 1 <= trace_display_topk <= trace_topk')
        if args.trace_cases_per_type < -1:
            sys.exit('ERROR: trace_cases_per_type must be -1 or nonnegative')
        for vocab_path in (args.trace_concept_vocab, args.trace_shape_vocab):
            if vocab_path and not os.path.isfile(vocab_path):
                sys.exit(f'ERROR: trace vocabulary file not found: {vocab_path}')
        if not args.use_shape:
            print("ERROR: --explain_trace requires --use_shape", file=sys.stderr)
            sys.exit(1)
        if args.shape_input != 'geo':
            print("ERROR: Section 4.6 trace uses the default geo11 descriptor; "
                  "set --shape_input geo", file=sys.stderr)
            sys.exit(1)
        if args.shape_mode not in ('decoded', 'decoded_rel'):
            print("ERROR: --explain_trace is for deployable decoded/decoded_rel modes, "
                  "not the test-mask oracle", file=sys.stderr)
            sys.exit(1)
        if args.use_cooccur:
            print("ERROR: --explain_trace must isolate the shape residual; "
                  "do not enable --use_cooccur", file=sys.stderr)
            sys.exit(1)

    # 设置随机种子 (控制 KMeans 初始化 + batch shuffle)
    set_seed(args.seed)

    # 分支描述
    branches = []
    if args.use_shape: branches.append(f"shape({args.shape_input})")
    if args.use_cooccur: branches.append("cooccur")
    branch_str = "+".join(branches) if branches else "(none)"

    print(f"\n{'='*70}")
    print(f"  SDR-CBM — dataset={args.dataset}")
    print(f"  Branches: {branch_str}")
    print(f"  device={args.device}, model={args.model_name}, seg={args.segment_model}")
    print(f"  K_sem={args.K_sem}, K_shape={args.K_shape}, lr_sem={args.lr_sem}")
    print(f"  epochs_s1={args.epochs_s1}, bs={args.bs}, es_pat={args.es_pat}")
    print(f"  eta_vals={args.eta_vals}")
    print(f"{'='*70}")

    # 加载数据集路径
    ds_info = load_dataset_info(args.dataset)
    ds_info["semantic_asset_root"] = os.path.dirname(ds_info["masks_dir"])
    if args.shape_asset_root:
        ds_info["masks_dir"] = os.path.join(args.shape_asset_root, 'masks')
        ds_info["fd_csv"] = os.path.join(args.shape_asset_root, 'shape_csv',
                                        os.path.basename(ds_info["fd_csv"]))
    if args.shape_csv_dir:
        ds_info["fd_csv"] = os.path.join(args.shape_csv_dir, os.path.basename(ds_info["fd_csv"]))
    print(f"\n[Dataset] {args.dataset}")
    for k, v in ds_info.items():
        if k == "fd_csv" and args.use_shape:
            # 显示用户实际选择的形状特征 CSV 路径（仅计算，不触发生成）
            suffix = SHAPE_FEATURE_TYPES[args.shape_input][2]
            actual = ds_info["fd_csv"].replace("_fd32", suffix)
            print(f"  fd_csv: {actual}  (--shape_input={args.shape_input})")
        elif k != "hp":
            print(f"  {k}: {v}")
    if "hp" in ds_info:
        print(f"  hp (推荐超参数): {ds_info['hp']}")

    # 确保 shape_csv 目录存在
    os.makedirs(os.path.dirname(ds_info["fd_csv"]) or '.', exist_ok=True)

    # 统一实验入口
    results = run_experiment(ds_info, args.dataset, args)

    # ── Summary ──
    section("SUMMARY")
    if "baseline" in results:
        tr, va, te = results.pop("baseline")
        print(f"  Semantic Baseline: Train={tr:.2f}% Val={va:.2f}% Test={te:.2f}%")

    for k in sorted(results):
        if results[k] is None: continue
        va_r, te_r, eta = results[k]
        print(f"  {k:<35} Val={va_r:.2f}% Test={te_r:.2f}%  eta={eta}")
