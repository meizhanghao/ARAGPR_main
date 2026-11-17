"""Export selected manuscript figure PDFs as lossless PNGs for documentation.

Requires PyMuPDF and Pillow; it does not train models or alter source PDFs.
Example: python tools/export_paper_figures.py --source-dir /path/to/manuscript
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path


FIGURES = [
    (1, "fig1.pdf", "residual_interfaces.png", "Residual interfaces and geometric references", 2200),
    (2, "fig2.pdf", "framework.png", "Training, mask-free inference, and attribution readout", 2400),
    (3, "residual_behavior_journal_decoded.pdf", "residual_dynamics.png", "Decoded validation residual dynamics", 2000),
    (4, "prototype_naming_clusters.pdf", "prototype_clusters.png", "Prototype inspection and auxiliary naming", 2800),
    (5, "prototype_contribution_audit.pdf", "contribution_audit.png", "Complete coordinate profiles and Top-5 references", 2600),
    (6, "qualitative_corrections.pdf", "corrected_predictions.png", "Corrected predictions and signed margin decompositions", 2200),
    (7, "qualitative_harmful_changes.pdf", "harmful_changes.png", "Harmful prediction changes and signed margin decompositions", 2200),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_figures(source, output):
    import fitz
    from PIL import Image

    source, output = Path(source).resolve(), Path(output).resolve()
    missing = [name for _, name, *_ in FIGURES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing manuscript figure PDFs: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, name, image_name, title, width in FIGURES:
        src, target = source / name, output / image_name
        source_hash = sha256(src)
        with fitz.open(src) as document:
            if len(document) != 1:
                raise ValueError(f"Expected a standalone one-page figure: {name}")
            page = document[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(width / page.rect.width,
                                                       width / page.rect.width),
                                     colorspace=fitz.csRGB, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.save(target, format="PNG", optimize=True, compress_level=9)
            dimensions = list(image.size)
        if source_hash != sha256(src):
            raise RuntimeError(f"Source changed during export: {name}")
        record = dict(manuscript_figure=number, source_file=name,
                      source_sha256=source_hash, documentation_image=image_name,
                      image_sha256=sha256(target), image_size_px=dimensions,
                      image_bytes=target.stat().st_size, title=title)
        records.append(record)
        print(f"Figure {number}: {image_name} ({dimensions[0]} x {dimensions[1]})")
    manifest = dict(manuscript_title="Auditing Residual Attribution with Geometric Prototype References",
                    transformation="Complete figure page rendered to RGB PNG; no edits to panels, labels or values.",
                    scope="Curated manuscript illustrations only; not a dataset or checkpoint release.",
                    figures=records)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/assets/figures")
    args = parser.parse_args()
    export_figures(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()

