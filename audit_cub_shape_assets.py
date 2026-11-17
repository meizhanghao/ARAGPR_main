"""Check a freshly segmented CUB shape bank before training (no GPU needed)."""
import argparse
import json
from pathlib import Path

import numpy as np

from cancha_cub import _preprocess_mask, _extract_contour_raw, _extract_geometric_features
from data.Segments.scripts.utilities.segment import asset_sha256, load_paired_mask_records


def audit_bank(asset_root, image_root):
    bank, images = Path(asset_root).resolve(), Path(image_root).resolve()
    sources = {}
    for path in images.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        key = f'{path.parent.name}/{path.stem}'
        if key in sources:
            raise ValueError(f'Duplicate CUB image identity: {key}')
        sources[key] = path
    if not sources:
        raise ValueError(f'No source images in {images}')
    manifests = {}
    for path in (bank / 'masks').rglob('*_MASKS_DONE.json'):
        key = f'{path.parent.name}/{path.name.removesuffix("_MASKS_DONE.json")}'
        manifests[key] = json.loads(path.read_text(encoding='utf-8'))
    if sources.keys() != manifests.keys():
        missing = sorted(sources.keys() - manifests.keys())
        extra = sorted(manifests.keys() - sources.keys())
        raise ValueError(f'Incomplete image coverage: missing={missing[:10]}, extra={extra[:10]}')
    for key, source in sources.items():
        if manifests[key].get('source', {}).get('sha256') != asset_sha256(source):
            raise ValueError(f'Source image does not match segmentation provenance: {key}')
    records = load_paired_mask_records(bank / 'masks')
    usable_images = set()
    errors = []
    valid = 0
    for path in records:
        try:
            clean = _preprocess_mask(str(path), strict_binary=True)
            features = _extract_geometric_features(_extract_contour_raw(clean))
            if features.shape != (11,) or not np.isfinite(features).all():
                raise ValueError('Invalid geo11 descriptor')
            valid += 1
            usable_images.add(f'{path.parent.name}/{path.name.split("_OUT_MASK_", 1)[0]}')
        except ValueError as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    if not valid:
        raise ValueError('No valid geo11 descriptors after mask preprocessing')
    report = {
        'asset_root': str(bank), 'image_root': str(images),
        'source_images': len(sources), 'committed_images': len(manifests),
        'verified_pairs': len(records), 'valid_geo11_regions': valid,
        'rejected_geo11_regions': len(errors),
        'images_with_valid_geo11': len(usable_images),
        'images_without_valid_geo11': sorted(sources.keys() - usable_images),
        'rejected_region_examples': errors[:20],
        'asset_provenance_passed': True,
    }
    (bank / 'shape_asset_audit.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset_root', required=True)
    parser.add_argument('--image_root', required=True)
    args = parser.parse_args()
    audit_bank(args.asset_root, args.image_root)
