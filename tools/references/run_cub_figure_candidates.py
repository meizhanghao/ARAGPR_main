"""Export auditable CUB Figure 1 candidates without training or changing a model.

Existing trace bundles can be read directly.  --paired-root additionally replays
saved paired-control checkpoints and exports their correction/harm traces through
the existing SDRCBMVisualizer.  All outputs go into a NEW directory.  Part names
are review annotations for training exemplars, never test-image localization or
labels assigned to every member of a shape cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


from aragpr.paths import repository_root

ROOT = repository_root()
DATASET = "CUB_200_2011"
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, GRAY, RULE = "#273640", "#65727B", "#D5DEE4"
PART_VOCAB = [
    "a bird head and beak", "a bird head and breast", "a bird neck",
    "a bird wing", "a bird tail", "a bird leg and foot", "a bird throat patch",
    "a bird eye region", "a bird torso", "a whole bird",
    "a tree branch", "a tree trunk", "a leaf", "a patch of sky",
    "a patch of water", "a patch of ground", "a rock", "a man-made object",
    "the background surrounding a bird", "an unidentifiable region",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class SourceLedger:
    """Hash read inputs once and verify them again when the export completes."""

    def __init__(self):
        self.files = {}

    def record(self, path):
        path = Path(path).resolve()
        if path not in self.files:
            self.files[path] = digest(path)
        return path

    def verify(self):
        changed = [str(p) for p, previous in self.files.items() if digest(p) != previous]
        if changed:
            raise ValueError(f"Input files changed during export: {changed}")
        return [{"path": str(p), "sha256": value} for p, value in self.files.items()]


def bundle_asset(folder, relative, boundary):
    """Resolve portable bundle paths, allowing ../prototype_assets inside a trace."""
    if not relative:
        return None
    relative = str(relative).replace("\\", "/")
    if relative.startswith("/") or re.match(r"^[A-Za-z]:", relative):
        raise ValueError("Expected a portable relative bundle path.")
    path = (Path(folder) / relative).resolve()
    if not path.is_relative_to(Path(boundary).resolve()):
        raise ValueError("Bundle asset escapes the trace root.")
    return path if path.is_file() else None


def checked_decomposition(summary, coordinates, class_rows, topk=3, tolerance=5e-5):
    """Check all coordinates, full-vector argmaxes, and the displayed remainder."""
    ids = [int(r["prototype_id"]) for r in coordinates]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Missing or duplicate shape coordinates.")
    for row in coordinates:
        # SDRCBMVisualizer already includes eta_s in the exported weight
        # difference.  Multiplying eta_s again would reject valid nonunit runs.
        expected = float(row["activation"]) * float(row["weight_difference_f_minus_b"])
        if abs(expected - float(row["signed_margin_contribution"])) > tolerance:
            raise ValueError("Coordinate contribution differs from activation times weight.")
    ordered = sorted(coordinates, key=lambda r: (-abs(float(r["signed_margin_contribution"])),
                                                int(r["prototype_id"])))
    qsum = math.fsum(float(r["signed_margin_contribution"]) for r in coordinates)
    semantic = float(summary["semantic_margin_f_minus_b"])
    context = float(summary.get("context_margin_f_minus_b", 0))
    bias = float(summary["shape_bias_margin_f_minus_b"])
    final = float(summary["final_margin_f_minus_b"])
    closure = abs(semantic + context + qsum + bias - final)
    if closure > tolerance:
        raise ValueError(f"Margin decomposition does not close: {closure}")
    # Older exports use these explicit column names.  Never infer a prediction
    # from the pairwise margin when the complete class vector is available.
    if class_rows:
        first = class_rows[0]
        sem_key = next((k for k in ("semantic_logit", "z_sem", "dcbm_logit") if k in first), None)
        final_key = next((k for k in ("final_logit", "z_final", "sdrcbm_logit") if k in first), None)
        if sem_key and final_key:
            base_row = max(class_rows, key=lambda r: float(r[sem_key]))
            final_row = max(class_rows, key=lambda r: float(r[final_key]))
            if int(base_row["class_id"]) != int(summary["dcbm_class"]) or int(final_row["class_id"]) != int(summary["sdrcbm_class"]):
                raise ValueError("Prediction label disagrees with the complete logits.")
    return ordered[:topk], qsum - math.fsum(float(r["signed_margin_contribution"]) for r in ordered[:topk]), closure


def replay_logits(decoded, coordinate_scaler, head):
    branch = coordinate_scaler.transform(np.asarray(decoded, dtype=np.float32)).astype(np.float32)
    selected = head["selected"]
    weight = np.asarray(selected["state"]["weight"], dtype=np.float32)
    bias = np.asarray(selected["state"]["bias"], dtype=np.float32)
    return branch, float(selected["eta"]) * (branch @ weight.T + bias)


def replay_paired(args, out, ledger):
    """Reuse the normal exporter with saved parameters; no optimizer is created."""
    import torch
    from aragpr.training import _preprocess_mask, _extract_contour_raw, _extract_geometric_features
    from experiments.controls.run_decoder_baselines import load_cached_inputs
    from utils.sdr_cbm_visualize import SDRCBMVisualizer

    def geometry(path):
        clean = _preprocess_mask(path)
        contour = _extract_contour_raw(clean)
        return clean.shape, contour, _extract_geometric_features(contour)

    h, teacher, valid, keys, labels, manifest, sources = load_cached_inputs(args.cache_root, DATASET)
    for value in sources.values():
        ledger.record(value["path"])
    bank_path = ledger.record(args.cache_root / DATASET / "teacher_bank.pt")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    checkpoint_path = ledger.record(args.project_root / "semantic_checkpoints/CUB_200_2011_semantic_baseline.pt")
    if digest(checkpoint_path) != manifest["sources"]["checkpoint"]["sha256"]:
        raise ValueError("Frozen checkpoint differs from the cache manifest.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_scaler = torch.load(args.cache_root / DATASET / "input_scaler.pt", map_location="cpu", weights_only=False)["h_scaler"]
    features = input_scaler.inverse_transform(h["test"]).astype(np.float32)
    concepts = np.asarray(checkpoint["concepts"], dtype=np.float32)
    raw = features @ concepts.T / np.square(np.linalg.norm(concepts, axis=1))[None, :]
    raw -= raw.max(axis=1, keepdims=True)
    responses = np.exp(raw)
    responses /= responses.sum(axis=1, keepdims=True)
    semantic_features = checkpoint["scaler"].transform(responses).astype(np.float32)
    semantic_head = torch.nn.Linear(concepts.shape[0], checkpoint["num_classes"])
    semantic_head.load_state_dict(checkpoint["model_state"])
    semantic = semantic_features @ semantic_head.weight.detach().numpy().T + semantic_head.bias.detach().numpy()
    with np.load(args.cache_root / DATASET / "test_frozen.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(semantic, saved["logits"], rtol=3e-5, atol=6e-5)
    known_semantic = {}
    for _, root in args.trace:
        for path in root.glob("*/semantic_prototypes.csv"):
            ledger.record(path)
            for row in read_csv(path):
                if row.get("representative_key"):
                    known_semantic[int(row["concept_id"])] = row["representative_key"]
    roots, checks = [], []
    for seed in args.seeds:
        folder = args.paired_root / f"seed_{seed}" / "paired"
        model_path = ledger.record(folder / "selected_model.pt")
        prediction_path = ledger.record(folder / "test_predictions.npz")
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if state["mode"] != "paired" or int(state["seed"]) != seed:
            raise ValueError("Expected a correctly paired saved checkpoint.")
        with np.load(prediction_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_keys"].astype(str), keys["test"]) or not np.array_equal(saved["labels"], labels["test"]):
                raise ValueError("Saved prediction membership differs from cache.")
            branch, shape_logits = replay_logits(saved["decoded"], state["coordinate_scaler"], state["head"])
            replay = semantic + shape_logits
            np.testing.assert_allclose(replay, saved["logits"], rtol=3e-5, atol=6e-5)
            if not np.array_equal(replay.argmax(1), saved["logits"].argmax(1)):
                raise ValueError("Prediction replay changes a top-1 label.")
            max_error = float(np.max(abs(replay - saved["logits"])))
        shape_head = torch.nn.Linear(branch.shape[1], checkpoint["num_classes"])
        shape_head.load_state_dict(state["head"]["selected"]["state"])
        cbm = SimpleNamespace(model=semantic_head, concepts=concepts, K=len(concepts), dataset=DATASET,
            class_names=[n.split(".", 1)[-1] for n in manifest["class_order"]],
            A_test_scaled=semantic_features, y_test=labels["test"],
            semantic_proto_representative=dict(known_semantic),
            **{f"data_{split}_raw": dict.fromkeys(keys[split]) for split in keys})
        rt = SimpleNamespace(cbm=cbm, sb=shape_head, eta_s=state["head"]["selected"]["eta"],
            Sclip_test=branch, Ash_test_s=teacher["test"],
            shape_proto_representative=bank["prototype_representatives"],
            shape_proto_metadata=bank["prototype_metadata"])
        label = f"R{seed}"
        visualizer = SDRCBMVisualizer(rt, str(out / "replayed_traces" / label),
            device="cpu", dataset=DATASET, image_root=str(args.image_root),
            shape_asset_root=str(args.shape_asset_root),
            semantic_asset_root=str(args.project_root / "data/Segments/Seg_embs/CUB_200_2011_SAM2"),
            semantic_segment_file=str(args.project_root / "data/Segments/Seg_embs/segcrop_CUB_200_2011_SAM2_CLIP-ViT-L14.torch"),
            shape_geometry_fn=geometry)
        result = visualizer.export_case_study(cases_per_type=-1, topk=5, display_topk=3,
            render_plots=False, run_config={"source": "saved_paired_control_replay",
                "seed": seed, "model_path": str(model_path), "model_sha256": digest(model_path),
                "teacher_bank_sha256": digest(bank_path), "prediction_sha256": digest(prediction_path),
                "prediction_replay_max_absolute_error": max_error, "new_training": False,
                "scope": "local validation-selected single checkpoint, not server last-15 means"})
        roots.append((label, Path(result["root"])))
        checks.append({"run": label, "seed": seed, "prediction_max_absolute_error": max_error,
                       "all_top1_predictions_reproduced": True, "n_bundles": len(result["selected"])})
        print(f"Replay {label}: {len(result['selected'])} cases; max logit error {max_error:.3g}", flush=True)
    write_json(out / "replay_checks.json", checks)
    return roots


def font(size, chinese=False, bold=False):
    choices = (["msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf"] if chinese else
               ["timesbd.ttf" if bold else "times.ttf", "DejaVuSerif.ttf"])
    for name in choices:
        for candidate in [Path("C:/Windows/Fonts") / name, Path(name)]:
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def fit_text(draw, text, xy, width, size=20, color=INK, chinese=False, max_lines=3):
    f = font(size, chinese)
    words = list(text) if chinese else text.replace("_", " ").split()
    lines, line = [], ""
    for word in words:
        candidate = (line + ("" if chinese or not line else " ") + word)
        if line and draw.textlength(candidate, font=f) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    for i, line in enumerate(lines[:max_lines]):
        draw.text((xy[0], xy[1] + i * (size + 5)), line, font=f, fill=color)


def paste_fit(canvas, source, box, background="white"):
    x, y, width, height = map(int, box)
    ImageDraw.Draw(canvas).rectangle((x, y, x + width, y + height), fill=background)
    if source is None or (not isinstance(source, Image.Image) and not Path(source).is_file()):
        fit_text(ImageDraw.Draw(canvas), "Asset unavailable", (x + 8, y + height // 2), width - 10, 17, GRAY)
        return
    if isinstance(source, Image.Image):
        im = source.convert("RGB")
    else:
        with Image.open(source) as image:
            im = image.convert("RGB")
    thumb = ImageOps.contain(im, (width, height), Image.Resampling.LANCZOS)
    canvas.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))


def export_reference(folder, out, args, reviews, ledger):
    evidence = read_json(ledger.record(folder / "evidence.json"))
    if evidence.get("geometry_status") != "verified_against_training_descriptor" or evidence["metadata"].get("source_split") != "train":
        raise ValueError(f"Unverified or nontraining reference: {folder}")
    key = evidence["representative_key"]
    identity = hashlib.sha256((key + "\n" + str(evidence["metadata"]["raw_descriptor"])).encode()).hexdigest()[:16]
    dest = out / "references" / identity
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["crop.jpg", "mask.png", "neutral_crop.png", "verified_contour.png", "overlay.png", "evidence.json"]:
        source = folder / name
        if source.is_file():
            ledger.record(source)
            if not (dest / name).exists():
                shutil.copy2(source, dest / name)
    match = re.fullmatch(r"(.+)_OUT_CROP_(\d+)_X_\.jpg", key)
    context_status = "unavailable"
    if match:
        stem, region = match.groups()
        manifest_path = args.shape_asset_root / "masks" / (stem + "_MASKS_DONE.json")
        original = args.image_root / "train" / (stem + ".jpg")
        if manifest_path.is_file() and original.is_file():
            manifest = read_json(ledger.record(manifest_path))
            ledger.record(original)
            regions = [r for r in manifest["regions"] if int(r["region_id"]) == int(region)]
            if len(regions) != 1:
                raise ValueError("Ambiguous training region in mask manifest.")
            rr = regions[0]
            if rr["crop_sha256"] != digest(folder / "crop.jpg") or rr["mask_sha256"] != digest(folder / "mask.png"):
                raise ValueError("Training crop/mask differs from its completion manifest.")
            if manifest["source"].get("sha256") and manifest["source"]["sha256"] != digest(original):
                raise ValueError("Training image differs from its completion manifest.")
            shutil.copy2(original, dest / "training_original.jpg")
            with Image.open(original) as source:
                context = source.convert("RGB").resize(tuple(manifest["processed_size_wh"]), Image.Resampling.LANCZOS)
            x, y, w, h = rr["bbox_xywh"]
            draw = ImageDraw.Draw(context)
            draw.rectangle((x, y, x + w, y + h), outline=BLUE, width=3)
            context.save(dest / "training_context.png")
            write_json(dest / "region_provenance.json", {"source": str(original),
                "source_sha256": digest(original), "manifest": str(manifest_path),
                "processed_size_wh": manifest["processed_size_wh"], "region": rr,
                "rectangle_scope": "training reference only; not test localization"})
            context_status = "hash_verified_training_bbox"
    review = reviews.get(key, {})
    reference = {"identity": identity, "representative_key": key, "path": dest,
        "label_en": review.get("label_en", "Unreviewed region"),
        "label_zh": review.get("label_zh", "待复核区域"),
        "category": review.get("category", "unreviewed"),
        "usable_part": bool(review.get("usable_part", False)),
        "review_note": review.get("note", ""), "context_status": context_status,
        "name_scope": "training exemplar only; not all cluster members or test localization",
        "geometry_status": evidence["geometry_status"]}
    write_json(dest / "review.json", {k: str(v) if isinstance(v, Path) else v for k, v in reference.items()})
    return reference


def clip_screen(references, args, out, ledger):
    if args.clip_rankings:
        cached = ledger.record(args.clip_rankings)
        rows = read_csv(cached)
        tops = {}
        for identity, ref in references.items():
            matches = [r for r in rows if r["reference_id"] == identity]
            if len(matches) != len(PART_VOCAB) or any(r["representative_key"] != ref["representative_key"] for r in matches):
                raise ValueError("Cached CLIP ranking has different reference identities.")
            previous_image = ledger.record(cached.parent / "references" / identity / "neutral_crop.png")
            if digest(previous_image) != digest(ref["path"] / "neutral_crop.png"):
                raise ValueError("Cached CLIP ranking used different reference pixels.")
            best = next(r for r in matches if int(r["rank"]) == 1)
            tops[identity] = best["candidate"]
        write_csv(out / "clip_reference_rankings.csv", rows)
        vocab = ledger.record(cached.parent / "clip_vocabulary.json")
        shutil.copy2(vocab, out / "clip_vocabulary.json")
        write_json(out / "clip_reuse.json", {"source": str(cached), "sha256": digest(cached),
            "all_reference_pixels_match": True, "new_CLIP_inference": False})
        return tops
    if args.clip_weights is None:
        return {}
    import clip
    import torch
    weights = ledger.record(args.clip_weights)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    torch.set_num_threads(4)
    model, preprocess = clip.load(str(weights), device=args.device, jit=False)
    model.eval()
    prompts = [f"a photo of {phrase}." for phrase in PART_VOCAB]
    with torch.inference_mode():
        text = model.encode_text(clip.tokenize(prompts).to(args.device)).float()
        text /= text.norm(dim=-1, keepdim=True)
    refs = list(references.values())
    ranking, tops = [], {}
    for start in range(0, len(refs), args.batch_size):
        batch = refs[start:start + args.batch_size]
        images = []
        for ref in batch:
            with Image.open(ref["path"] / "neutral_crop.png") as im:
                images.append(preprocess(im.convert("RGB")))
        with torch.inference_mode():
            visual = model.encode_image(torch.stack(images).to(args.device)).float()
            visual /= visual.norm(dim=-1, keepdim=True)
            values = (visual @ text.T).cpu().numpy()
        for ref, scores in zip(batch, values):
            order = np.argsort(-scores, kind="stable")
            tops[ref["identity"]] = PART_VOCAB[int(order[0])]
            for rank, j in enumerate(order, 1):
                ranking.append({"reference_id": ref["identity"], "representative_key": ref["representative_key"],
                    "rank": rank, "candidate": PART_VOCAB[int(j)], "cosine_similarity": float(scores[j]),
                    "top1_top2_gap": float(scores[order[0]] - scores[order[1]]),
                    "status": "unreviewed_CLIP_candidate_not_probability", "scope": "training_exemplar_only"})
        print(f"CLIP reference screening {min(start + len(batch), len(refs))}/{len(refs)}", flush=True)
    write_csv(out / "clip_reference_rankings.csv", ranking)
    write_json(out / "clip_vocabulary.json", {"phrases": PART_VOCAB, "prompt": "a photo of {phrase}.",
        "source": "author-designed review candidates, not official CUB part annotations",
        "scores_are_probabilities": False, "used_to_change_predictions": False,
        "used_as_verified_part_names": False})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tops


def rank_key(case):
    # This is explicitly a qualitative presentation search, not model selection.
    # It never changes top-k coordinates, checkpoints, or numeric predictions.
    return (case["category"] != "correction", -case["top3_usable_parts"],
            -case["top3_bird_references"], -case["positive_part_abs_sum"],
            -case["top3_coverage"], case["candidate_id"])


def make_shortlist(cases, limit):
    """Keep different test images and give distinct reviewed parts a first pass."""
    ranked = sorted(cases, key=rank_key)
    selected, seen_images, seen_parts = [], set(), set()
    for case in ranked:
        parts = {t["reference"]["identity"] for t in case["top"] if t["reference"]["usable_part"]}
        if case["category"] == "correction" and case["index"] not in seen_images and parts - seen_parts:
            selected.append(case)
            seen_images.add(case["index"])
            seen_parts.update(parts)
            if len(selected) == limit:
                return selected
    for case in ranked:
        if case["category"] == "correction" and case["index"] not in seen_images:
            selected.append(case)
            seen_images.add(case["index"])
            if len(selected) == limit:
                break
    return selected


def draw_waterfall(canvas, case, box):
    draw = ImageDraw.Draw(canvas)
    x, y, width, height = box
    components = [("Semantic", case["semantic_margin"])] + [
        (f"P{r['prototype_id']}", float(r["signed_margin_contribution"])) for r in case["top"]]
    components += [("Other", case["other"]), ("Bias", case["bias"])]
    if case["context_margin"]:
        components.insert(1, ("Context", case["context_margin"]))
    running, entries = 0., []
    for label, delta in components:
        entries.append((label, delta, running, running + delta))
        running += delta
    entries.append(("Final", case["final_margin"], 0., case["final_margin"]))
    lo = min(0., *(min(a, b) for _, _, a, b in entries))
    hi = max(0., *(max(a, b) for _, _, a, b in entries))
    span = max(hi - lo, .05)
    def yy(value):
        return y + 22 + (hi + .12 * span - value) / (1.24 * span) * (height - 56)
    draw.line((x, yy(0), x + width, yy(0)), fill=RULE, width=1)
    step = width / len(entries)
    for i, (label, delta, before, after) in enumerate(entries):
        xx = x + (i + .18) * step
        color = GREEN if label == "Final" else GRAY if label == "Semantic" else BLUE if delta >= 0 else ORANGE
        draw.rectangle((xx, min(yy(before), yy(after)), xx + .57 * step, max(yy(before), yy(after)) + 1), fill=color)
        if i < len(entries) - 2:
            draw.line((xx + .57 * step, yy(after), x + (i + 1.18) * step, yy(after)), fill=RULE)
        draw.text((xx, min(yy(before), yy(after)) - 20), f"{delta:+.3f}", font=font(18), fill=INK)
        draw.text((xx, y + height - 26), label, font=font(18), fill=INK)


def case_card(case, out):
    canvas = Image.new("RGB", (1680, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    fit_text(draw, f"{case['candidate_id']} | {case['category']} | {case['true_name']}", (25, 18), 1600, 30)
    fit_text(draw, f"{case['run_label']} | test index {case['index']} | exact absolute-contribution Top-3", (25, 60), 1600, 20, GRAY)
    for xx, ww, title in [(20, 320, "Input image"), (365, 460, "DCBM: visual-concept references"), (850, 805, "SDR-CBM: shape references + contribution")]:
        draw.rectangle((xx, 110, xx + ww, 700), outline=RULE, width=2)
        fit_text(draw, title, (xx + 12, 123), ww - 20, 23)
    paste_fit(canvas, case["path"] / "input.jpg", (30, 185, 300, 280))
    fit_text(draw, "Ground truth: " + case["true_name"], (35, 490), 285, 22)
    fit_text(draw, "DCBM prediction: " + case["dcbm_name"], (380, 170), 430, 23)
    fit_text(draw, "SDR prediction: " + case["sdrcbm_name"], (865, 170), 770, 23, GREEN if case["category"] == "correction" else INK)
    for i, row in enumerate(case["semantic"][:3]):
        xx = 380 + i * 145
        paste_fit(canvas, case["path"] / f"semantic_{i+1}.jpg", (xx, 270, 130, 170))
        fit_text(draw, f"C{row['concept_id']}", (xx, 455), 130, 21)
        fit_text(draw, f"q_base={float(row['margin_support_for_dcbm_class']):+.3f}", (xx, 487), 135, 19)
    fit_text(draw, "Training reference crops; not test-image localization.", (380, 600), 420, 20, GRAY)
    for i, row in enumerate(case["top"]):
        xx = 865 + i * 260
        ref = row["reference"]
        fit_text(draw, f"#{i+1} P{row['prototype_id']}  {float(row['signed_margin_contribution']):+.3f}", (xx, 235), 250, 23,
                 BLUE if float(row["signed_margin_contribution"]) >= 0 else ORANGE)
        paste_fit(canvas, ref["path"] / "neutral_crop.png", (xx, 282, 120, 155))
        paste_fit(canvas, ref["path"] / "verified_contour.png", (xx + 125, 282, 120, 155))
        fit_text(draw, ref["label_en"], (xx, 449), 245, 20)
        paste_fit(canvas, ref["path"] / "training_context.png", (xx, 520, 245, 155))
    fit_text(draw, "Margin decomposition: final class versus DCBM class (all coordinates retained in CSV)", (30, 720), 1620, 22)
    draw_waterfall(canvas, case, (130, 757, 1450, 205))
    fit_text(draw, "Local saved checkpoint. Names describe training exemplars; positive contribution does not imply high similarity.", (30, 971), 1620, 17, GRAY)
    path = out / "cards" / (case["candidate_id"] + ".png")
    canvas.save(path)
    return path


def reference_sheets(references, out):
    refs = list(references.values())
    for start in range(0, len(refs), 16):
        canvas = Image.new("RGB", (1680, 1500), "white")
        draw = ImageDraw.Draw(canvas)
        fit_text(draw, "Training references: original context | region | verified contour", (15, 8), 1650, 25)
        for i, ref in enumerate(refs[start:start + 16]):
            x, y = i % 4 * 420, 50 + i // 4 * 360
            draw.rectangle((x + 3, y, x + 416, y + 355), outline=RULE)
            fit_text(draw, ref["label_en"], (x + 10, y + 8), 400, 19)
            fit_text(draw, " / ".join(ref.get("aliases", [ref["identity"]])), (x + 10, y + 35), 400, 14, GRAY, max_lines=1)
            paste_fit(canvas, ref["path"] / "training_context.png", (x + 10, y + 60, 395, 155))
            paste_fit(canvas, ref["path"] / "neutral_crop.png", (x + 12, y + 225, 185, 117))
            paste_fit(canvas, ref["path"] / "verified_contour.png", (x + 207, y + 225, 195, 117))
        canvas.save(out / "sheets" / f"references_{start//16+1:02d}.png")


def html_gallery(cases, references, shortlist, out):
    short_ids = {c["candidate_id"] for c in shortlist}
    options = "".join(f'<option>{html.escape(r)}</option>' for r in dict.fromkeys(c["run_label"] for c in cases))
    tiles = []
    for c in cases:
        cid = c["candidate_id"]
        refs = " / ".join(f"P{r['prototype_id']}: {r['reference']['label_zh']}" for r in c["top"])
        tiles.append(f'''<article data-category="{c['category']}" data-run="{html.escape(c['run_label'])}"
          data-parts="{c['top3_usable_parts']}" data-short="{str(cid in short_ids).lower()}">
          <h2>{cid} · {html.escape(c['true_name'])} <small>{'推荐候选' if cid in short_ids else ''}</small></h2>
          <p>DCBM: {html.escape(c['dcbm_name'])} → SDR-CBM: {html.escape(c['sdrcbm_name'])}；
          {'纠错' if c['category']=='correction' else '误改'}；前三项清楚部件 {c['top3_usable_parts']} 个</p>
          <p>{html.escape(refs)}</p><a href="cards/{cid}.png"><img loading="lazy" src="cards/{cid}.png"></a>
          <p><a href="cases/{cid}/candidate.json">来源与统计</a> ·
          <a href="cases/{cid}/shape_coordinates.csv">全部形状坐标</a> ·
          <a href="cases/{cid}/class_logits.csv">完整分类分数</a></p></article>''')
    source = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CUB 图1候选全集</title>
    <style>body{font:16px "Microsoft YaHei",sans-serif;color:#273640;background:#f4f7f9;margin:24px}
    main{max-width:1500px;margin:auto}header{position:sticky;top:0;background:white;padding:14px;border:1px solid #d5dee4;z-index:1}
    h1{font-size:24px;margin:4px 0}h2{font-size:21px}small{color:#009E73}p{line-height:1.7}select,input{font:inherit;margin:5px;padding:5px}
    article{background:white;border:1px solid #d5dee4;margin:18px 0;padding:15px}img{width:100%;height:auto}a{color:#0072B2}
    #count{font-weight:bold}button{padding:7px;font:inherit}</style><main><header><h1>CUB 图 1 候选全集</h1>
    <p>每张卡片均保留同次预测的真实 Top-3；部件名描述训练参考图，不代表测试图像的部件定位。蓝色为正贡献，橙色为负贡献。</p>
    <label><input id="short" type="checkbox" checked>只看推荐</label>
    <select id="category"><option value="all">全部类别</option><option value="correction">纠错</option><option value="harm">误改</option></select>
    <select id="run"><option value="all">全部运行</option>RUN_OPTIONS</select>
    <label><input id="parts" type="checkbox">前三项至少一个清楚部件</label><input id="search" placeholder="编号或鸟类名称"><span id="count"></span>
    <p><a href="candidate_ranking.csv">全部排序 CSV</a> · <a href="README_zh.md">筛选说明</a> · <a href="reference_index.html">全部原型参考</a></p></header>ARTICLES</main>
    <script>const articles=[...document.querySelectorAll('article')];function filter(){let n=0;for(const a of articles){const show=(!document.querySelector('#short').checked||a.dataset.short==='true')&&
    (category.value==='all'||a.dataset.category===category.value)&&(run.value==='all'||a.dataset.run===run.value)&&
    (!parts.checked||+a.dataset.parts>0)&&a.textContent.toLowerCase().includes(search.value.toLowerCase());a.hidden=!show;if(show)n++}count.textContent=n+' / '+articles.length+' 个候选'}
    for(const e of document.querySelectorAll('input,select'))e.addEventListener('input',filter);filter();</script></html>'''
    (out / "index.html").write_text(source.replace("RUN_OPTIONS", options).replace("ARTICLES", "\n".join(tiles)), encoding="utf-8")
    groups = []
    for ref in references.values():
        identity = ref["identity"]
        groups.append(f'<article><h2>{html.escape(ref["label_zh"])} · {html.escape(ref["label_en"])}</h2>'
            f'<p>{html.escape(" / ".join(ref.get("aliases", [])))}</p>'
            f'<p>{html.escape(ref["representative_key"])}</p><p>{html.escape(ref["review_note"])}</p>'
            f'<img style="width:32%" loading="lazy" src="references/{identity}/training_context.png">'
            f'<img style="width:32%" loading="lazy" src="references/{identity}/neutral_crop.png">'
            f'<img style="width:32%" loading="lazy" src="references/{identity}/verified_contour.png"></article>')
    (out / "reference_index.html").write_text('<!doctype html><meta charset="utf-8"><title>CUB 原型参考</title>'
        '<style>body{font:16px "Microsoft YaHei";margin:30px;color:#273640}article{border:1px solid #d5dee4;margin:15px;padding:15px}img{height:240px;object-fit:contain}</style>'
        '<h1>全部经几何核验的训练原型参考</h1><a href="index.html">返回候选</a>' + "\n".join(groups), encoding="utf-8")


def run(args):
    out = args.output.resolve()
    for _, root in args.trace:
        if out.is_relative_to(root.resolve()) or root.resolve().is_relative_to(out):
            raise ValueError("Output and input trace directories must be separate.")
    out.mkdir(parents=True, exist_ok=False)
    for name in ["references", "cases", "cards", "sheets"]:
        (out / name).mkdir()
    ledger = SourceLedger()
    ledger.record(Path(__file__))
    reviews = read_json(ledger.record(args.review_labels)) if args.review_labels else {}
    roots = list(args.trace)
    if args.paired_root:
        roots += replay_paired(args, out, ledger)
    if len({label for label, _ in roots}) != len(roots):
        raise ValueError("Trace run labels must be unique.")
    references, ref_lookup, runs = {}, {}, []
    for run_label, root in roots:
        run_info = read_json(ledger.record(root / "trace_run.json"))
        if run_info["dataset"] != DATASET or not run_info.get("export_complete"):
            raise ValueError("Expected a completed CUB trace.")
        config = run_info.get("config", {})
        if config.get("source") == "saved_paired_control_replay":
            model_path = ledger.record(config["model_path"])
            if digest(model_path) != config["model_sha256"]:
                raise ValueError("Replayed source checkpoint has changed.")
        runs.append({"run": run_label, "trace_root": str(root), "trace_run": run_info})
        for folder in sorted((root / "prototype_assets").glob("P*")):
            ref = export_reference(folder, out, args, reviews, ledger)
            references.setdefault(ref["identity"], ref)
            current = references[ref["identity"]]
            current.setdefault("aliases", []).append(f"{run_label}/P{int(folder.name[1:])}")
            ref_lookup[run_label, int(folder.name[1:])] = current
    for ref in references.values():
        write_json(ref["path"] / "review.json", {k: str(v) if isinstance(v, Path) else v for k, v in ref.items()})
    clip_tops = clip_screen(references, args, out, ledger)
    cases = []
    for run_label, root in roots:
        rows = read_csv(ledger.record(root / "selected_cases.csv"))
        for row in rows:
            source = (root / row["bundle_relative_path"]).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Case path escapes trace.")
            summary = read_json(ledger.record(source / "case_summary.json"))
            coordinates = read_csv(ledger.record(source / "shape_coordinates.csv"))
            class_rows = read_csv(ledger.record(source / "class_logits.csv"))
            top, other, closure = checked_decomposition(summary, coordinates, class_rows)
            if row["category"] != summary["category"] or int(row["index"]) != int(summary["index"]):
                raise ValueError("Case manifest differs from its bundle.")
            cid = f"{run_label}-{int(row['index']):04d}"
            dest = out / "cases" / cid
            dest.mkdir()
            for name in ["case_summary.json", "shape_coordinates.csv", "shape_prototypes.csv", "semantic_prototypes.csv", "class_logits.csv", "waterfall.csv"]:
                ledger.record(source / name)
                shutil.copy2(source / name, dest / name)
            image = bundle_asset(source, summary["input_bundle_relative_path"], root)
            if image is None:
                raise ValueError("A candidate is missing its input image.")
            ledger.record(image)
            shutil.copy2(image, dest / "input.jpg")
            semantic = sorted(read_csv(source / "semantic_prototypes.csv"), key=lambda r: int(r["rank"]))
            for i, sem in enumerate(semantic[:3], 1):
                asset = bundle_asset(source, sem.get("bundle_crop_relative_path"), root)
                if asset:
                    ledger.record(asset)
                    shutil.copy2(asset, dest / f"semantic_{i}.jpg")
            for t in top:
                pid = int(t["prototype_id"])
                if (run_label, pid) not in ref_lookup:
                    raise ValueError(f"Missing verified prototype P{pid} in {run_label}.")
                t["reference"] = ref_lookup[run_label, pid]
            abs_sum = math.fsum(abs(float(r["signed_margin_contribution"])) for r in coordinates)
            case = {"candidate_id": cid, "run_label": run_label, "source_case": source,
                "path": dest, "index": int(summary["index"]), "sample_key": summary["sample_key"],
                "true_name": summary["true_name"], "dcbm_name": summary["dcbm_name"],
                "sdrcbm_name": summary["sdrcbm_name"], "category": summary["category"],
                "semantic_margin": float(summary["semantic_margin_f_minus_b"]),
                "context_margin": float(summary.get("context_margin_f_minus_b", 0)),
                "final_margin": float(summary["final_margin_f_minus_b"]),
                "bias": float(summary["shape_bias_margin_f_minus_b"]), "other": other,
                "top3_usable_parts": sum(t["reference"]["usable_part"] for t in top),
                "top3_bird_references": sum(t["reference"]["category"] in ("bird_part", "whole_bird") for t in top),
                "positive_part_abs_sum": math.fsum(float(t["signed_margin_contribution"]) for t in top
                    if t["reference"]["usable_part"] and float(t["signed_margin_contribution"]) > 0),
                "top3_coverage": math.fsum(abs(float(t["signed_margin_contribution"])) for t in top) / max(abs_sum, 1e-12),
                "closure_error": closure, "top": top, "semantic": semantic}
            cases.append(case)
    cases.sort(key=rank_key)
    shortlist = make_shortlist(cases, args.shortlist)
    flat = []
    for rank, c in enumerate(cases, 1):
        record = {k: v for k, v in c.items() if k not in ("top", "semantic")}
        record = {k: str(v) if isinstance(v, Path) else v for k, v in record.items()}
        record.update(rank=rank, shortlisted=c in shortlist,
            top3_ids=";".join("P" + t["prototype_id"] for t in c["top"]),
            top3_labels=";".join(t["reference"]["label_en"] for t in c["top"]),
            top3_contributions=";".join(t["signed_margin_contribution"] for t in c["top"]))
        flat.append(record)
        write_json(c["path"] / "candidate.json", {**record, "references": [{"prototype_id": t["prototype_id"],
            "reference_id": t["reference"]["identity"], "label_en": t["reference"]["label_en"],
            "clip_unreviewed_top1": clip_tops.get(t["reference"]["identity"]),
            "signed_margin_contribution": float(t["signed_margin_contribution"])} for t in c["top"]]})
        case_card(c, out)
    write_csv(out / "candidate_ranking.csv", flat)
    write_json(out / "runs.json", runs)
    write_json(out / "review_labels_used.json", reviews)
    reference_sheets(references, out)
    # A compact visual entry point; complete high-resolution cards remain separate.
    sheet = Image.new("RGB", (1680, 110 + 500 * math.ceil(min(6, len(shortlist)) / 2)), "white")
    fit_text(ImageDraw.Draw(sheet), "CUB Figure 1: shortlisted real prediction examples", (22, 15), 1640, 30)
    fit_text(ImageDraw.Draw(sheet), "Open index.html to inspect every candidate and full-resolution assets.", (22, 62), 1640, 22, GRAY)
    for i, c in enumerate(shortlist[:6]):
        paste_fit(sheet, out / "cards" / (c["candidate_id"] + ".png"), (i % 2 * 840, 110 + i // 2 * 500, 840, 500))
    sheet.save(out / "shortlist_overview.png")
    html_gallery(cases, references, shortlist, out)
    (out / "README_zh.md").write_text(
        f"# 图 1 的 CUB 候选\n\n共保留 {len(cases)} 个运行—样例组合，{len({c['index'] for c in cases})} 张不同测试图像，"
        f"其中 {sum(c['category']=='correction' for c in cases)} 个纠错、{sum(c['category']=='harm' for c in cases)} 个误改。"
        f"保留 {len(references)} 个不同的训练参考区域。打开 index.html 可筛选全部，默认显示 {len(shortlist)} 个不同图像的推荐候选。\n\n"
        "筛选依据依次为：纠错；真实 Top-3 内清楚的鸟类部件数量；鸟体参考数量；正向部件贡献；Top-3 绝对贡献覆盖率。"
        "推荐列表先覆盖不同的清楚部件参考，再按上述顺序补足，且同一测试图像只推荐一次。"
        "这是展示样例筛选，不是模型选参或总体解释性评估。没有替换任何样例的高贡献坐标。\n\n"
        "T 是已有论文解释导出；R42/R43/R44 是此前保存的真实配对对照模型的逐样本回放，不是新训练。"
        "各运行的参数、teacher bank、预测文件与校验见 runs.json / replay_checks.json。"
        "预测来自各运行选定的单检查点。相同P编号在不同原型库中可能对应不同参考区域，需结合reference_id读取。\n\n"
        "卡片左侧为 DCBM 在相同类别对上支持原始预测的语义参考；右侧为最终类别相对原始类别的有符号形状贡献。"
        "它们是训练参考，不是在测试图上定位到的部件。原图矩形仅用于核查训练参考区域。"
        "区域名称用于描述所展示训练参考的可见内容。"
        "头部、喉部、腿足等名称只描述所展示训练区域；不能据此将整个几何原型命名为一个已验证的解剖学概念。\n\n"
        "全部坐标、偏置和 Other 均保留；贡献经过乘积与求和闭合校验。CLIP 仅给候选文字排序，"
        "分数不是概率，未自动变成核实名称，也不参与分类。review_labels_used.json 可在人工审阅后继续编辑，重新导出到另一个新目录。\n\n"
        "cards/ 为完整预选卡片；cases/ 含原输入、原 CSV 与完整类别分数；references/ 含彩色区域、二值 mask、"
        "核验轮廓、训练原图、原图位置和来源；sheets/ 为全部参考图集。\n",
        encoding="utf-8")
    inputs = ledger.verify()
    write_json(out / "source_integrity.json", {"all_input_files_unchanged": True, "files": inputs})
    write_json(out / "manifest.json", {"status": "complete", "n_cases": len(cases), "n_references": len(references),
        "n_unique_test_images": len({c["index"] for c in cases}),
        "shortlist": [c["candidate_id"] for c in shortlist], "max_margin_closure_error": max(c["closure_error"] for c in cases),
        "new_training": False, "new_segmentation": False, "existing_predictions_changed": False,
        "existing_inputs_unchanged": True, "n_hashed_inputs": len(inputs),
        "selection_scope": "qualitative presentation candidates, not population evidence"})
    print(json.dumps({"output": str(out), "cases": len(cases), "references": len(references),
                      "shortlist": [c["candidate_id"] for c in shortlist]}, ensure_ascii=False), flush=True)


def trace_argument(value):
    label, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise argparse.ArgumentTypeError("Use LABEL=TRACE_ROOT, with an alphanumeric label.")
    return label, Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", type=trace_argument, default=[], help="LABEL=TRACE_ROOT; repeatable")
    parser.add_argument("--paired-root", type=Path, help="Optional saved paired-control root; replay only")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/residual_controls_20261018_paired01")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--image-root", type=Path, default=ROOT / "CUB_200_2011_split")
    parser.add_argument("--shape-asset-root", type=Path, default=ROOT / "data/Segments/Seg_embs/CUB_200_2011_SAM2_binary_v3")
    parser.add_argument("--review-labels", type=Path, help="JSON keyed by exact representative_key, not by P-number")
    naming = parser.add_mutually_exclusive_group()
    naming.add_argument("--clip-weights", type=Path, help="Optional already-downloaded CLIP checkpoint; no download")
    naming.add_argument("--clip-rankings", type=Path, help="Reuse an earlier export's rankings after checking reference pixels")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shortlist", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    if not args.trace and not args.paired_root:
        parser.error("Provide --trace or --paired-root.")
    if args.shortlist < 1 or args.batch_size < 1:
        parser.error("shortlist and batch-size must be positive.")
    run(args)


if __name__ == "__main__":
    main()
