"""
SDR-CBM Visualization — contribution decomposition, co-occurrence matrix, text reports.

Usage (via train.py --viz):
    python train.py ... --viz --viz_split test --viz_indices 0 10 25
"""
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import clip
except Exception:
    clip = None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _linear_params(module):
    """Return (W, b) from nn.Linear or module.linear."""
    if module is None:
        return None, None
    layer = module.linear if hasattr(module, "linear") else module
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
    return W, b


class SDRCBMVisualizer:
    """Contribution decomposition + reports for SDR-CBM."""

    def __init__(self, rt, out_dir="./outputs/sdr_cbm_viz",
                 concept_vocab_path=None, image_root=None,
                 mask_root=None, device="cuda:0",
                 class_labels_path=None, semantic_segment_file=None,
                 shape_asset_root=None, dataset=None, shape_geometry_fn=None,
                 shape_vocab_path=None, semantic_asset_root=None):
        self.rt = rt
        self.cbm = rt.cbm
        self.out_dir = _ensure_dir(out_dir)
        self.image_root = image_root
        self.mask_root = mask_root
        self.shape_asset_root = shape_asset_root or (
            str(Path(mask_root).parent) if mask_root else None)
        self.semantic_asset_root = semantic_asset_root or self.shape_asset_root
        self.class_labels_path = class_labels_path
        self.semantic_segment_file = semantic_segment_file
        self.dataset = dataset or getattr(self.cbm, "dataset", "dataset")
        self.device = device
        self.shape_geometry_fn = shape_geometry_fn
        self.shape_vocab_path = shape_vocab_path
        self._sample_keys = {}
        self._asset_paths = {}
        self._input_path_indexes = {}
        self._shape_evidence_cache = {}
        self._clip_runtime = None
        self._shape_text = None

        self.class_names = self._load_class_names()
        self.concept_names = self._load_or_make_concept_names()

        if concept_vocab_path is not None:
            self.try_clip_text_align(concept_vocab_path)

    # ── metadata ──────────────────────────────────────────────
    def _load_class_names(self):
        for name in ["class_labels", "classes", "class_names"]:
            if hasattr(self.cbm, name):
                val = getattr(self.cbm, name)
                if val is not None:
                    return list(val)
        if self.class_labels_path and os.path.exists(self.class_labels_path):
            with open(self.class_labels_path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                if self.dataset == "CUB_200_2011":
                    names = [n.split(".", 1)[1] if "." in n else n for n in names]
                return names
        return None

    def _load_or_make_concept_names(self):
        for name in ["concept_names", "feature_names", "concept_name_list"]:
            if hasattr(self.cbm, name):
                val = getattr(self.cbm, name)
                if val is not None and len(val) > 0:
                    return list(val)
        K = getattr(self.rt, "K", None)
        if K is None:
            for s in ["test", "val", "train"]:
                try:
                    A = self._get_sem_feature(s)
                    K = A.shape[1]
                    break
                except Exception:
                    pass
        if K is None:
            K = 1024
        return [f"concept_{i}" for i in range(K)]

    def _get_split_raw_dict(self, split):
        name = f"data_{split}_raw"
        if hasattr(self.cbm, name):
            return getattr(self.cbm, name)
        return None

    def _get_sample_key(self, split, idx):
        raw = self._get_split_raw_dict(split)
        if raw is None:
            return f"{split}_{idx}"
        if split not in self._sample_keys:
            self._sample_keys[split] = list(raw.keys())
        keys = self._sample_keys[split]
        if idx >= len(keys):
            return f"{split}_{idx}"
        return keys[idx]

    def _get_label_array(self, split):
        for name in [f"y_{split}", f"{split}_labels_int",
                     f"y_{split}_int", f"data_{split}_labels_raw"]:
            if hasattr(self.cbm, name):
                arr = getattr(self.cbm, name)
                if arr is not None:
                    arr = _to_numpy(arr)
                    if arr.ndim > 1:
                        arr = np.argmax(arr, axis=1)
                    return arr.astype(int)
        return None

    def _get_attr_first(self, obj, names):
        for n in names:
            if hasattr(obj, n):
                val = getattr(obj, n)
                if val is not None:
                    return val
        return None

    # ── feature getters ───────────────────────────────────────
    def _get_sem_feature(self, split):
        val = self._get_attr_first(self.cbm,
            [f"A_{split}_scaled", f"X_{split}", f"A_{split}_s"])
        if val is None:
            raise RuntimeError(f"Cannot find semantic feature for split={split}")
        return _to_numpy(val).astype(np.float32)

    def _get_sem_raw(self, split):
        val = self._get_attr_first(self.cbm,
            [f"A_{split}", f"A_{split}_raw"])
        if val is None:
            val = self._get_sem_feature(split)
        return _to_numpy(val).astype(np.float32)

    def _get_ctx_feature(self, split):
        val = self._get_attr_first(self.rt,
            [f"Actx_{split}", f"Aco_{split}"])
        return _to_numpy(val).astype(np.float32) if val is not None else None

    def _get_shape_feature(self, split, shape_mode):
        if shape_mode == "decoded_rel":
            names = [f"Sclip_{split}_rel"]
        elif shape_mode == "decoded":
            names = [f"Sclip_{split}"]
        elif shape_mode == "oracle":
            names = [f"Ash_{split}_s"]
        else:
            raise ValueError(f"Unknown shape_mode={shape_mode}")
        val = self._get_attr_first(self.rt, names)
        return _to_numpy(val).astype(np.float32) if val is not None else None

    # ── CLIP text alignment (optional) ─────────────────────────
    def _clip_encoder(self):
        if self._clip_runtime is None:
            if clip is None:
                raise RuntimeError('Install OpenAI CLIP to use optional prototype naming.')
            mn = getattr(self.cbm, 'model_name',
                         getattr(self.cbm, 'clip_model', 'CLIP-ViT-L14'))
            aliases = {'CLIP-ViT-L14': 'ViT-L/14', 'CLIP-ViT-B16': 'ViT-B/16',
                       'CLIP-ViT-B32': 'ViT-B/32', 'CLIP-RN50': 'RN50',
                       'CLIP-RN101': 'RN101', 'CLIP-ViT-L14-336': 'ViT-L/14@336px'}
            arch = aliases.get(mn, mn)
            if arch not in clip.available_models():
                raise ValueError(f'Unsupported CLIP backbone for naming: {mn}')
            model, preprocess = clip.load(arch, device=self.device, jit=False)
            model.eval()
            self._clip_runtime = (model, preprocess, arch)
        return self._clip_runtime

    def try_clip_text_align(self, vocab_path, top_prompt="a photo of {}."):
        if clip is None:
            print("[Viz] clip not available. Skip text alignment.")
            return
        if not os.path.exists(vocab_path):
            print(f"[Viz] vocab not found: {vocab_path}. Skip.")
            return
        prototypes = None
        for name in ["concepts", "clustered_concepts", "full_concepts"]:
            if hasattr(self.cbm, name):
                prototypes = getattr(self.cbm, name)
                if prototypes is not None:
                    break
        if prototypes is None:
            print("[Viz] No concept prototype embeddings. Skip text alignment.")
            return
        prototypes = _to_numpy(prototypes).astype(np.float32)
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [l.strip() for l in f if l.strip()]
        prompts = [top_prompt.format(v) for v in vocab]

        if not vocab:
            raise ValueError(f'Empty concept vocabulary: {vocab_path}')
        model, _, _ = self._clip_encoder()
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_emb = model.encode_text(tokens).float().cpu().numpy()
        text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
        sim = prototypes @ text_emb.T
        best = sim.argmax(axis=1)
        aligned = [f"{vocab[j]} ({sim[i,j]:.2f})" for i, j in enumerate(best)]
        if len(aligned) == len(self.concept_names):
            self.concept_names = aligned
            print(f"[Viz] CLIP text alignment done ({len(aligned)} concepts).")

    # ── core decomposition ────────────────────────────────────
    def explain_instance(self, split="test", idx=0, topk=5, shape_mode="decoded_rel"):
        A_sem_all = self._get_sem_feature(split)
        A_raw_all = self._get_sem_raw(split)
        A_ctx_all = self._get_ctx_feature(split)
        A_shape_all = self._get_shape_feature(split, shape_mode)

        A_sem  = A_sem_all[idx]
        A_raw  = A_raw_all[idx]
        A_ctx  = A_ctx_all[idx] if A_ctx_all is not None else None
        A_shape = A_shape_all[idx] if A_shape_all is not None else None

        W_sem, b_sem = _linear_params(self.cbm.model)
        if W_sem is None:
            raise RuntimeError("Cannot find cbm.model linear parameters.")
        z_sem = W_sem @ A_sem + b_sem
        z_final = z_sem.copy()

        z_ctx = None; W_ctx = None; b_ctx = None
        if hasattr(self.rt, "cb") and self.rt.cb is not None and A_ctx is not None:
            W_ctx, b_ctx = _linear_params(self.rt.cb)
            z_ctx = W_ctx @ A_ctx + b_ctx
            z_final = z_final + float(getattr(self.rt, "eta_c", 1.0)) * z_ctx

        z_shape = None; W_shape = None; b_shape = None
        if hasattr(self.rt, "sb") and self.rt.sb is not None and A_shape is not None:
            W_shape, b_shape = _linear_params(self.rt.sb)
            z_shape = W_shape @ A_shape + b_shape
            z_final = z_final + float(getattr(self.rt, "eta_s", 1.0)) * z_shape

        pred = int(np.argmax(z_final))
        labels = self._get_label_array(split)
        true = int(labels[idx]) if labels is not None else None

        sem_contrib = W_sem[pred] * A_sem
        out = {
            "split": split, "idx": int(idx),
            "sample_key": self._get_sample_key(split, idx),
            "pred_class": pred, "true_class": true,
            "pred_name": self._class_name(pred),
            "true_name": self._class_name(true) if true is not None else None,
            "z_sem": z_sem, "z_ctx": z_ctx, "z_shape": z_shape, "z_final": z_final,
            "sem_contrib": sem_contrib,
            "top_sem_pos": self._top_idx(sem_contrib, topk, True),
            "top_sem_neg": self._top_idx(sem_contrib, topk, False),
        }

        if A_ctx is not None and W_ctx is not None:
            eta_c = float(getattr(self.rt, "eta_c", 1.0))
            ctx_contrib = eta_c * W_ctx[pred] * A_ctx
            out["ctx_contrib"] = ctx_contrib
            out["top_ctx_pos"] = self._top_idx(ctx_contrib, topk, True)
            out["top_ctx_neg"] = self._top_idx(ctx_contrib, topk, False)
            out["ctx_neighbors"] = self._ctx_neighbors(A_raw, W_ctx, pred,
                                                       out["top_ctx_pos"], topk)

        if A_shape is not None and W_shape is not None:
            eta_s = float(getattr(self.rt, "eta_s", 1.0))
            shape_contrib = eta_s * W_shape[pred] * A_shape
            out["shape_contrib"] = shape_contrib
            out["top_shape_pos"] = self._top_idx(shape_contrib, topk, True)
            out["top_shape_neg"] = self._top_idx(shape_contrib, topk, False)

        return out

    def _top_idx(self, arr, k, largest):
        arr = np.asarray(arr)
        return np.argsort(arr)[::-1][:k].astype(int).tolist() if largest else \
               np.argsort(arr)[:k].astype(int).tolist()

    def _class_name(self, idx):
        if idx is None:
            return None
        if self.class_names and 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"

    def _concept_name(self, idx):
        if self.concept_names and 0 <= idx < len(self.concept_names):
            return self.concept_names[idx]
        return f"concept_{idx}"

    def _ctx_neighbors(self, A_raw, W_ctx, pred, top_ctx, topk=5):
        if not hasattr(self.rt, "Co") or self.rt.Co is None:
            return {}
        C = self.rt.Co.detach().cpu().numpy() if torch.is_tensor(self.rt.Co) \
            else np.asarray(self.rt.Co)
        neighbors = {}
        for j in top_ctx:
            support = A_raw * C[:, j]
            scale_j = 1.0
            if hasattr(self.rt, "ctx_scaler") and hasattr(self.rt.ctx_scaler, "scale_"):
                scale_j = float(self.rt.ctx_scaler.scale_[j] + 1e-8)
            eta_c = float(getattr(self.rt, "eta_c", 1.0))
            pair_score = eta_c * (W_ctx[pred, j] / scale_j) * C[:, j] * A_raw * A_raw[j]
            nbrs = np.argsort(np.abs(pair_score))[::-1][:topk]
            neighbors[int(j)] = [{
                "neighbor": int(i),
                "neighbor_name": self._concept_name(int(i)),
                "C_ij": float(C[i, j]),
                "a_i": float(A_raw[i]),
                "support_a_i_Cij": float(support[i]),
                "pair_score_approx": float(pair_score[i]),
            } for i in nbrs]
        return neighbors

    # ── Section 4.6 trace export ─────────────────────────────
    @staticmethod
    def _safe_name(value):
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
        return value or "item"

    @staticmethod
    def _write_csv(path, rows, fieldnames=None):
        rows = list(rows)
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _compute_split_state(self, split, shape_mode):
        A_sem = self._get_sem_feature(split)
        A_shape = self._get_shape_feature(split, shape_mode)
        if A_shape is None:
            raise RuntimeError(
                f"No shape feature for split={split}, mode={shape_mode}. "
                "Run shape-teacher/decoder setup before --explain_trace."
            )

        W_sem, b_sem = _linear_params(self.cbm.model)
        W_shape, b_shape = _linear_params(getattr(self.rt, "sb", None))
        if W_sem is None or W_shape is None:
            raise RuntimeError("Semantic and shape heads must both be available for trace export.")

        z_sem = A_sem @ W_sem.T + b_sem[None, :]
        eta_s = float(getattr(self.rt, "eta_s", 1.0))
        z_shape = eta_s * (A_shape @ W_shape.T + b_shape[None, :])

        A_ctx = self._get_ctx_feature(split)
        W_ctx, b_ctx = _linear_params(getattr(self.rt, "cb", None))
        if A_ctx is not None and W_ctx is not None:
            eta_c = float(getattr(self.rt, "eta_c", 1.0))
            z_ctx = eta_c * (A_ctx @ W_ctx.T + b_ctx[None, :])
        else:
            z_ctx = np.zeros_like(z_sem)

        z_final = z_sem + z_ctx + z_shape
        labels = self._get_label_array(split)
        if labels is None:
            raise RuntimeError(f"Cannot find labels for split={split}.")
        if len(labels) != len(z_final):
            raise ValueError('Label/logit lengths differ; cannot record all sample IDs reliably.')
        raw = self._get_split_raw_dict(split)
        if raw is not None and len(raw) != len(labels):
            raise ValueError('Sample-key order/length does not match the evaluation split.')
        if not np.isfinite(z_final).all() or not np.isfinite(z_sem).all():
            raise ValueError('Non-finite logits in explanation export.')

        return {
            "split": split,
            "A_sem": A_sem,
            "A_shape": A_shape,
            "A_ctx": A_ctx,
            "W_sem": W_sem,
            "b_sem": b_sem,
            "W_shape": W_shape,
            "b_shape": b_shape,
            "W_ctx": W_ctx,
            "b_ctx": b_ctx,
            "eta_s": eta_s,
            "z_sem": z_sem,
            "z_ctx": z_ctx,
            "z_shape": z_shape,
            "z_final": z_final,
            "labels": labels,
            "base_pred": z_sem.argmax(axis=1),
            "final_pred": z_final.argmax(axis=1),
        }

    def _case_rows(self, state):
        rows = []
        split = state["split"]
        for idx, (true, base, final) in enumerate(zip(
                state["labels"], state["base_pred"], state["final_pred"])):
            true = int(true); base = int(base); final = int(final)
            base_ok = base == true
            final_ok = final == true
            if not base_ok and final_ok:
                category = "correction"
            elif base_ok and not final_ok:
                category = "harm"
            elif base == final and base_ok:
                category = "unchanged_correct"
            elif base == final:
                category = "unchanged_error"
            else:
                category = "other_flip"

            sem_margin = float(state["z_sem"][idx, final] - state["z_sem"][idx, base])
            ctx_margin = float(state["z_ctx"][idx, final] - state["z_ctx"][idx, base])
            shape_margin = float(state["z_shape"][idx, final] - state["z_shape"][idx, base])
            final_margin = float(state["z_final"][idx, final] - state["z_final"][idx, base])
            sample_key = str(self._get_sample_key(split, idx)).replace('\\', '/')
            stable_id = hashlib.sha256(
                f'{self.dataset}\n{split}\n{sample_key}'.encode('utf-8')).hexdigest()[:20]
            rows.append({
                "split": split,
                "index": idx,
                "sample_id": stable_id,
                "sample_key": sample_key,
                "input_source_path": self._resolve_input_path(split, sample_key),
                "true_class": true,
                "true_name": self._class_name(true),
                "dcbm_class": base,
                "dcbm_name": self._class_name(base),
                "sdrcbm_class": final,
                "sdrcbm_name": self._class_name(final),
                "category": category,
                "semantic_margin_f_minus_b": sem_margin,
                "context_margin_f_minus_b": ctx_margin,
                "shape_margin_f_minus_b": shape_margin,
                "final_margin_f_minus_b": final_margin,
                "dcbm_correct": int(base_ok),
                "sdrcbm_correct": int(final_ok),
            })
        return rows

    @staticmethod
    def _representative_subset(rows, category, count):
        candidates = [r for r in rows if r["category"] == category]
        if not candidates or count <= 0:
            return []
        values = np.array([r["shape_margin_f_minus_b"] for r in candidates], dtype=np.float64)
        median = float(np.median(values))
        ordered = sorted(
            candidates,
            key=lambda r: (abs(r["shape_margin_f_minus_b"] - median), r["index"]),
        )
        selected = []
        for rank, row in enumerate(ordered[:count], 1):
            item = dict(row)
            item["selection_rule"] = "nearest_to_category_median_shape_margin"
            item["category_shape_margin_median"] = median
            item["selection_rank"] = rank
            selected.append(item)
        return selected

    def export_case_study(self, split="test", shape_mode="decoded", topk=5,
                          cases_per_type=5, posthoc_shape_match=False,
                          display_topk=3, indices=None, render_plots=True,
                          run_config=None):
        """Export correction/harm evidence bundles for Section 4.6.

        IDs are always exhaustive. Bundle selection is by median margin,
        explicit indices, or all correction/harm cases (cases_per_type=-1).
        """
        if topk < 1 or not 1 <= display_topk <= topk or cases_per_type < -1:
            raise ValueError('Invalid trace top-k or case count.')
        if shape_mode not in ('decoded', 'decoded_rel'):
            raise ValueError('Explanation trace is for decoded/decoded_rel inference only.')
        state = self._compute_split_state(split, shape_mode)
        rows = self._case_rows(state)
        root = _ensure_dir(os.path.join(self.out_dir, self.dataset, shape_mode))
        if split != 'test':
            root = _ensure_dir(os.path.join(root, split))
        self._shape_evidence_cache.clear()

        if indices is not None:
            lookup = {r['index']: r for r in rows if r['category'] in ('correction', 'harm')}
            invalid = [i for i in indices if i not in lookup]
            if invalid:
                raise ValueError(f'These indices are not correction/harm cases in this run: {invalid}')
            selected = [dict(lookup[i], selection_rule='explicit_index', selection_rank=rank)
                        for rank, i in enumerate(dict.fromkeys(indices), 1)]
        elif cases_per_type == -1:
            selected = [dict(r, selection_rule='all_correction_and_harm', selection_rank=rank)
                        for rank, r in enumerate(
                            (r for r in rows if r['category'] in ('correction', 'harm')), 1)]
        else:
            selected = (self._representative_subset(rows, 'correction', cases_per_type)
                        + self._representative_subset(rows, 'harm', cases_per_type))
        for category in ('correction', 'harm'):
            values = [r['shape_margin_f_minus_b'] for r in rows if r['category'] == category]
            if values:
                median = float(np.median(values))
                for row in selected:
                    if row['category'] == category:
                        row['category_shape_margin_median'] = median
        selected_by_index = {r['index']: r for r in selected}
        for row in rows:
            row['bundle_requested'] = int(row['index'] in selected_by_index)
            row['bundle_relative_path'] = (
                f"{row['category']}_{split}_{row['index']:05d}" if row['bundle_requested'] else '')
            if row['bundle_requested']:
                selected_by_index[row['index']].update(row)
        fields = list(rows[0]) if rows else ['split', 'index', 'sample_id', 'sample_key',
                                           'category', 'bundle_requested', 'bundle_relative_path']
        self._write_csv(os.path.join(root, 'all_cases.csv'), rows, fields)
        for category in ('correction', 'harm', 'other_flip'):
            self._write_csv(os.path.join(root, f'{category}_cases.csv'),
                            (r for r in rows if r['category'] == category), fields)
        self._write_csv(os.path.join(root, 'selected_cases.csv'), selected,
                        fields + ['selection_rule', 'selection_rank', 'category_shape_margin_median'])

        counts = dict.fromkeys(('correction', 'harm', 'unchanged_correct', 'unchanged_error',
                               'other_flip'), 0)
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        with open(os.path.join(root, "case_counts.json"), "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2, ensure_ascii=False)

        total = len(rows)
        run_manifest = {
            'dataset': self.dataset, 'split': split, 'shape_mode': shape_mode,
            'eta_s': state['eta_s'], 'config': run_config or {},
            'export_topk': topk, 'display_topk': display_topk,
            'all_ids_recorded': True, 'n_samples': total,
            'n_bundles_requested': len(selected), 'n_bundles_completed': 0,
            'export_complete': False,
            'baseline_accuracy': sum(r['dcbm_correct'] for r in rows) / max(total, 1),
            'final_accuracy': sum(r['sdrcbm_correct'] for r in rows) / max(total, 1),
            'accuracy_delta': (counts['correction'] - counts['harm']) / max(total, 1),
            'rate_units': 'fraction',
            'selection_basis': 'validation-selected residual head; no test-score eta selection',
            'naming_scope': 'unreviewed training-exemplar labels, not test-image localization',
        }
        manifest_path = os.path.join(root, 'trace_run.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(run_manifest, f, indent=2, ensure_ascii=False)

        if not selected:
            run_manifest['export_complete'] = True
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(run_manifest, f, indent=2, ensure_ascii=False)
            print(f"[Trace] all IDs saved; no evidence bundles requested for {split}.")
            return {"root": root, "counts": counts, "selected": []}

        unique_semantic_ids = set()
        for row in selected:
            exp = self._build_margin_explanation(state, row, topk)
            unique_semantic_ids.update(exp["top_semantic_ids"])
        self._ensure_semantic_representatives(sorted(unique_semantic_ids))

        exported = []
        for i, row in enumerate(selected, 1):
            exp = self._build_margin_explanation(state, row, topk)
            exp['display_topk'] = display_topk
            exported.append(self._export_trace_case(
                exp, root, shape_mode, posthoc_shape_match, render_plots=render_plots))
            if i % 25 == 0:
                print(f'[Trace] evidence bundles: {i}/{len(selected)}', flush=True)
        run_manifest.update(export_complete=True, n_bundles_completed=len(exported))
        geometry_counts = {}
        for evidence in self._shape_evidence_cache.values():
            status = evidence['geometry_status']
            geometry_counts[status] = geometry_counts.get(status, 0) + 1
        run_manifest['shape_geometry_counts'] = geometry_counts
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(run_manifest, f, indent=2, ensure_ascii=False)
        print(f"[Trace] exported {len(exported)} cases -> {root}")
        print(f'[Trace] shape prototype geometry: {geometry_counts}')
        return {"root": root, "counts": counts, "selected": exported}

    def _build_margin_explanation(self, state, row, topk):
        idx = int(row["index"])
        base = int(row["dcbm_class"])
        final = int(row["sdrcbm_class"])
        A_sem = state["A_sem"][idx]
        A_shape = state["A_shape"][idx]

        sem_support_base = (state["W_sem"][base] - state["W_sem"][final]) * A_sem
        top_sem = np.argsort(-sem_support_base, kind='stable')[:topk].astype(int).tolist()

        shape_weight_diff = state["eta_s"] * (
            state["W_shape"][final] - state["W_shape"][base])
        shape_coord_margin = shape_weight_diff * A_shape
        top_shape_abs = np.argsort(-np.abs(shape_coord_margin), kind='stable')[:topk].astype(int).tolist()
        top_shape_support = np.argsort(-shape_coord_margin, kind='stable')[:topk].astype(int).tolist()
        shape_bias_margin = float(state["eta_s"] * (
            state["b_shape"][final] - state["b_shape"][base]))
        semantic_bias_margin = float(state["b_sem"][final] - state["b_sem"][base])

        top_abs_sum = float(shape_coord_margin[top_shape_abs].sum())
        all_abs_sum = float(np.abs(shape_coord_margin).sum())
        coverage = float(np.abs(shape_coord_margin[top_shape_abs]).sum() /
                         (all_abs_sum + 1e-12))

        exp = dict(row)
        exp.update({
            "topk": int(topk),
            "top_semantic_ids": top_sem,
            "semantic_support_base": sem_support_base,
            "semantic_activation": A_sem,
            "shape_activation": A_shape,
            "shape_weight_diff_f_minus_b": shape_weight_diff,
            "shape_coordinate_margin": shape_coord_margin,
            "top_shape_abs_ids": top_shape_abs,
            "top_shape_support_ids": top_shape_support,
            "shape_bias_margin_f_minus_b": shape_bias_margin,
            "semantic_bias_margin_f_minus_b": semantic_bias_margin,
            "top_shape_abs_sum": top_abs_sum,
            "coverage_at_k": coverage,
            "z_sem_row": state["z_sem"][idx],
            "z_ctx_row": state["z_ctx"][idx],
            "z_shape_row": state["z_shape"][idx],
            "z_final_row": state["z_final"][idx],
        })
        return exp

    def _ensure_semantic_representatives(self, prototype_ids):
        current = getattr(self.cbm, "semantic_proto_representative", None) or {}
        current = {int(k): v for k, v in current.items()}
        missing = [int(k) for k in prototype_ids if int(k) not in current]
        if not missing:
            self.cbm.semantic_proto_representative = current
            return
        if not self.semantic_segment_file or not os.path.exists(self.semantic_segment_file):
            print(f"[Trace] semantic segment file unavailable; unresolved IDs={missing}")
            self.cbm.semantic_proto_representative = current
            return

        prototypes = _to_numpy(self.cbm.concepts).astype(np.float32)
        wanted = {}
        for k in missing:
            digest = hashlib.sha1(np.ascontiguousarray(prototypes[k]).tobytes()).hexdigest()
            wanted.setdefault(digest, []).append(k)

        print(f"[Trace] recovering {len(missing)} semantic prototype crop keys from "
              f"{self.semantic_segment_file}")
        data = torch.load(self.semantic_segment_file, weights_only=False, map_location="cpu")
        seg_to_img = dict(zip(
            getattr(self.cbm, "seg_keys", []), getattr(self.cbm, "seg_to_img", [])))
        train_keys = set((getattr(self.cbm, "data_train_raw", None) or {}).keys())
        from utils.dcbm_1 import _build_unique_basename_index, _resolve_image_key
        all_keys = set().union(*(set((self._get_split_raw_dict(s) or {}).keys())
                                 for s in ('train', 'val', 'test')))
        basename_index = _build_unique_basename_index(all_keys)
        unresolved = set(missing)
        for key, value in data.items():
            mapped = seg_to_img.get(key)
            if mapped is None:
                source_image = str(key).split('_OUT_CROP_', 1)[0]
                mapped = _resolve_image_key(source_image, all_keys, basename_index)
            if mapped not in train_keys:
                continue
            emb = _to_numpy(value[0]).squeeze().astype(np.float32)
            digest = hashlib.sha1(np.ascontiguousarray(emb).tobytes()).hexdigest()
            for k in wanted.get(digest, []):
                current[k] = key
                unresolved.discard(k)
            if not unresolved:
                break
        del data
        if unresolved:
            print(f"[Trace] exact representative key not found for semantic IDs={sorted(unresolved)}")
        self.cbm.semantic_proto_representative = current

    @staticmethod
    def _candidate_extensions(path_like):
        path = Path(path_like)
        if path.suffix:
            return [str(path)]
        return [str(path.with_suffix(ext)) for ext in (".jpg", ".jpeg", ".png", ".bmp")]

    def _resolve_input_path(self, split, sample_key):
        if not self.image_root:
            return None
        key = str(sample_key).replace("\\", "/").lstrip("/")
        roots = [Path(self.image_root) / split, Path(self.image_root)]
        for root in roots:
            for candidate in self._candidate_extensions(root / key):
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)
        if len(Path(key).parts) == 1:
            search_root = roots[0] if roots[0].is_dir() else roots[1]
            if search_root not in self._input_path_indexes:
                index = {}
                for path in search_root.rglob('*'):
                    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        index.setdefault(path.name, []).append(str(path.resolve()))
                self._input_path_indexes[search_root] = index
            index = self._input_path_indexes[search_root]
            matches = {p for name in self._candidate_extensions(key)
                       for p in index.get(name, [])}
            if len(matches) == 1:
                return matches.pop()
        return None

    def _resolve_asset_path(self, key, kind, asset_root=None):
        cache_key = (key, kind, asset_root)
        if cache_key not in self._asset_paths:
            self._asset_paths[cache_key] = self._find_asset_path(key, kind, asset_root)
        return self._asset_paths[cache_key]

    def _find_asset_path(self, key, kind, asset_root=None):
        asset_root = asset_root or self.shape_asset_root
        if not key or not asset_root:
            return None
        root = Path(asset_root)
        key = str(key).replace("\\", "/").lstrip("/")
        key_path = Path(key)

        if kind == "crop":
            folder = "crops"
            variants = [key.replace("_OUT_SHAPE_", "_OUT_CROP_")]
        elif kind == "mask":
            folder = "masks"
            variants = [key.replace("_OUT_CROP_", "_OUT_MASK_").replace("_OUT_SHAPE_", "_OUT_MASK_")]
        elif kind == "contour":
            folder = "contours"
            variants = [key.replace("_OUT_CROP_", "_OUT_CONTOUR_").replace("_OUT_SHAPE_", "_OUT_CONTOUR_")]
        elif kind == "shape":
            folder = "shapes"
            variants = [key.replace("_OUT_CROP_", "_OUT_SHAPE_")]
        else:
            raise ValueError(f"Unknown asset kind={kind}")

        for variant in variants:
            relative = Path(variant)
            stem = relative.with_suffix("")
            extensions = (['.png', relative.suffix, '.jpg', '.jpeg'] if kind != 'crop'
                          else [relative.suffix, '.jpg', '.png', '.jpeg'])
            for ext in dict.fromkeys(e for e in extensions if e):
                candidate = root / folder / stem.parent / (stem.name + ext)
                if candidate.is_file():
                    return str(candidate.resolve())

        # A qualified key must never fall back to another Places365 class.
        if len(key_path.parts) > 1:
            return None
        basename_stems = {Path(v).stem for v in variants}
        search_root = root / folder
        if search_root.is_dir():
            matches = []
            for stem in basename_stems:
                for ext in (".png", ".jpg", ".jpeg"):
                    matches.extend(search_root.glob(f"**/{stem}{ext}"))
                    if len(matches) > 1:
                        break
            unique = sorted({str(p.resolve()) for p in matches})
            if len(unique) == 1:
                return unique[0]
        return None

    @staticmethod
    def _copy_asset(source, target_dir, target_name):
        if not source or not os.path.isfile(source):
            return None
        os.makedirs(target_dir, exist_ok=True)
        suffix = Path(source).suffix.lower()
        target = os.path.join(target_dir, f"{target_name}{suffix}")
        shutil.copy2(source, target)
        return os.path.abspath(target)

    def _name_shape_exemplar(self, neutral_crop):
        if not self.shape_vocab_path:
            return {'status': 'not_requested', 'candidates': []}
        model, preprocess, arch = self._clip_encoder()
        if self._shape_text is None:
            with open(self.shape_vocab_path, encoding='utf-8-sig') as f:
                vocab = list(dict.fromkeys(line.strip() for line in f if line.strip()))
            if not vocab:
                raise ValueError(f'Empty shape vocabulary: {self.shape_vocab_path}')
            prompts = [f'a photo of {phrase}.' for phrase in vocab]
            embeddings = []
            with torch.no_grad():
                for start in range(0, len(prompts), 128):
                    tokens = clip.tokenize(prompts[start:start + 128]).to(self.device)
                    embeddings.append(model.encode_text(tokens).float().cpu().numpy())
            text = np.concatenate(embeddings)
            text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
            self._shape_text = (vocab, text)
        vocab, text = self._shape_text
        with Image.open(neutral_crop) as im:
            image_tensor = preprocess(im.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = model.encode_image(image_tensor).float().cpu().numpy()[0]
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        scores = text @ vector
        order = np.argsort(-scores, kind='stable')[:3]
        return {
            'status': 'candidate_unreviewed', 'backbone': arch,
            'vocabulary_path': os.path.abspath(self.shape_vocab_path),
            'prompt_template': 'a photo of {}.', 'input': neutral_crop,
            'scope': 'representative_training_crop_only',
            'score_type': 'cosine_similarity_not_probability',
            'top1_top2_gap': float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None,
            'candidates': [{'phrase': vocab[i], 'cosine': float(scores[i])} for i in order],
        }

    def _prepare_shape_evidence(self, prototype_id, root):
        if prototype_id in self._shape_evidence_cache:
            return self._shape_evidence_cache[prototype_id]
        representatives = getattr(self.rt, 'shape_proto_representative', {}) or {}
        metadata = getattr(self.rt, 'shape_proto_metadata', {}) or {}
        meta = dict(metadata.get(prototype_id, metadata.get(str(prototype_id), {})))
        key = representatives.get(prototype_id, representatives.get(str(prototype_id)))
        if not key:
            nearest = getattr(self.rt, 'shape_proto_nearest', {}) or {}
            keys = nearest.get(prototype_id, nearest.get(str(prototype_id), []))
            key = keys[0] if keys else None
            meta['representative_role'] = 'nearest_training_exemplar_unverified'
        meta['representative_key'] = key
        folder = _ensure_dir(os.path.join(root, 'prototype_assets', f'P{prototype_id:04d}'))
        evidence = {'representative_key': key,
                    'representative_role': meta.get('representative_role', 'unverified'),
                    'geometry_status': 'unavailable', 'metadata': meta,
                    'pair_path': None, 'overlay_path': None,
                    'neutral_crop_path': None, 'verified_contour_path': None}
        for kind in ('crop', 'mask', 'contour'):
            recorded = meta.get(kind + '_relpath')
            if recorded and self.shape_asset_root:
                bank = Path(self.shape_asset_root).resolve()
                candidate = (bank / recorded).resolve()
                source = str(candidate) if candidate.is_relative_to(bank) and candidate.is_file() else None
            else:
                source = self._resolve_asset_path(key, kind)
            if kind == 'contour' and not source:
                source = self._resolve_asset_path(key, 'shape')
            evidence[f'source_{kind}_path'] = source
            evidence[f'bundle_{kind}_path'] = self._copy_asset(source, folder, kind)

        expected = meta.get('raw_descriptor')
        if not key or not evidence['source_crop_path'] or not evidence['source_mask_path']:
            evidence['geometry_status'] = 'missing_crop_or_mask'
        elif self.shape_geometry_fn is None or expected is None:
            evidence['geometry_status'] = 'no_descriptor_verification_available'
        else:
            try:
                from data.Segments.scripts.utilities.segment import asset_sha256
                for kind in ('mask', 'crop'):
                    digest = meta.get(kind + '_sha256')
                    if digest and asset_sha256(evidence['source_' + kind + '_path']) != digest:
                        raise ValueError(f'{kind} differs from the training asset hash')
                mask_shape, contour, actual = self.shape_geometry_fn(evidence['source_mask_path'])
                actual = np.asarray(actual)
                expected = np.asarray(expected)
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError('Recomputed descriptor has invalid dimensions or values.')
                evidence['descriptor_max_abs_error'] = float(np.max(np.abs(actual - expected)))
                if not np.allclose(actual, expected, rtol=1e-4, atol=1e-6):
                    evidence['geometry_status'] = 'descriptor_mismatch'
                else:
                    with Image.open(evidence['source_crop_path']) as im:
                        crop = im.convert('RGB')
                    if crop.size != (mask_shape[1], mask_shape[0]):
                        evidence['geometry_status'] = 'crop_mask_size_mismatch'
                    else:
                        foreground = np.zeros(mask_shape, dtype=np.uint8)
                        cv2.drawContours(foreground, [contour], -1, 255, -1)
                        edge = np.zeros(mask_shape, dtype=np.uint8)
                        cv2.drawContours(edge, [contour], -1, 255, 1)
                        rgb = np.array(crop)
                        neutral = rgb.copy()
                        neutral[foreground == 0] = 224
                        overlay = rgb.copy()
                        overlay[edge > 0] = [0, 145, 110]
                        contour_image = Image.fromarray(255 - edge).convert('RGB')
                        for name, img in (('neutral_crop', Image.fromarray(neutral)),
                                          ('overlay', Image.fromarray(overlay)),
                                          ('verified_contour', contour_image)):
                            path = os.path.abspath(os.path.join(folder, name + '.png'))
                            img.save(path)
                            evidence[name + '_path'] = path
                        pair = Image.new('RGB', (560, 304), 'white')
                        draw = ImageDraw.Draw(pair)
                        font = ImageFont.truetype(
                            matplotlib.font_manager.findfont('DejaVu Sans'), 16)
                        for x, img, title in ((8, crop, 'Training crop'),
                                              (288, contour_image, 'Verified contour')):
                            draw.text((x, 8), f'P{prototype_id}: {title}', fill='black', font=font)
                            thumb = ImageOps.contain(img, (256, 256))
                            pair.paste(thumb, (x + (256 - thumb.width) // 2,
                                              36 + (256 - thumb.height) // 2))
                        evidence['pair_path'] = os.path.abspath(os.path.join(folder, 'crop_contour_pair.png'))
                        pair.save(evidence['pair_path'])
                        evidence['geometry_status'] = 'verified_against_training_descriptor'
            except (ValueError, OSError, cv2.error) as exc:
                evidence['geometry_status'] = 'geometry_error'
                evidence['geometry_error'] = str(exc)

        evidence['naming'] = (self._name_shape_exemplar(evidence['neutral_crop_path'])
                              if evidence['neutral_crop_path'] else
                              {'status': 'skipped_unverified_geometry', 'candidates': []})
        evidence['evidence_json_path'] = os.path.abspath(os.path.join(folder, 'evidence.json'))
        with open(evidence['evidence_json_path'], 'w', encoding='utf-8') as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)
        self._shape_evidence_cache[prototype_id] = evidence
        return evidence

    def _export_trace_case(self, exp, root, shape_mode, posthoc_shape_match, render_plots=True):
        case_name = f"{exp['category']}_{exp['split']}_{int(exp['index']):05d}"
        case_dir = _ensure_dir(os.path.join(root, case_name))
        asset_dir = _ensure_dir(os.path.join(case_dir, "assets"))

        input_source = self._resolve_input_path(exp["split"], exp["sample_key"])
        input_bundle = self._copy_asset(input_source, asset_dir, "input")

        representatives = getattr(self.cbm, "semantic_proto_representative", {}) or {}
        representatives = {int(k): v for k, v in representatives.items()}
        semantic_rows = []
        for rank, concept_id in enumerate(exp["top_semantic_ids"], 1):
            rep_key = representatives.get(int(concept_id))
            source = self._resolve_asset_path(rep_key, "crop", self.semantic_asset_root)
            bundle = self._copy_asset(
                source, asset_dir, f"semantic_{rank:02d}_concept_{concept_id:04d}")
            semantic_rows.append({
                "rank": rank,
                "concept_id": int(concept_id),
                "concept_name": self._concept_name(int(concept_id)),
                "margin_support_for_dcbm_class": float(exp["semantic_support_base"][concept_id]),
                "activation": float(exp["semantic_activation"][concept_id]),
                "representative_key": rep_key,
                "source_crop_path": source,
                "bundle_crop_path": bundle,
                "bundle_crop_relative_path": os.path.relpath(bundle, case_dir).replace('\\', '/') if bundle else None,
            })
        self._write_csv(os.path.join(case_dir, "semantic_prototypes.csv"), semantic_rows)

        reliability = getattr(self.rt, "shape_reliability", None)
        teacher = None
        if posthoc_shape_match:
            teacher_all = self._get_attr_first(self.rt, [f"Ash_{exp['split']}_s"])
            if teacher_all is not None:
                teacher = _to_numpy(teacher_all)[int(exp["index"])]

        absolute_rank = {int(k): rank for rank, k in enumerate(exp["top_shape_abs_ids"], 1)}
        support_rank = {int(k): rank for rank, k in enumerate(exp["top_shape_support_ids"], 1)}
        export_shape_ids = list(dict.fromkeys(
            exp["top_shape_abs_ids"] + exp["top_shape_support_ids"]))
        shape_rows = []
        for prototype_id in export_shape_ids:
            evidence = self._prepare_shape_evidence(int(prototype_id), root)
            q = float(exp["shape_coordinate_margin"][prototype_id])
            item = {
                "absolute_rank": absolute_rank.get(int(prototype_id)),
                "support_final_rank": support_rank.get(int(prototype_id)),
                "prototype_id": int(prototype_id),
                "signed_margin_contribution": q,
                "direction": ('supports_sdr_cbm' if q > 0 else
                              'supports_dcbm' if q < 0 else 'neutral'),
                "activation": float(exp["shape_activation"][prototype_id]),
                "weight_difference_f_minus_b": float(exp["shape_weight_diff_f_minus_b"][prototype_id]),
                "reliability": float(reliability[prototype_id])
                    if reliability is not None and prototype_id < len(reliability) else None,
                "posthoc_sam_teacher_activation": float(teacher[prototype_id])
                    if teacher is not None else None,
                "representative_key": evidence['representative_key'],
                "representative_role": evidence['representative_role'],
                "geometry_status": evidence['geometry_status'],
                "candidate_name_status": evidence['naming']['status'],
                "candidate_name": (evidence['naming']['candidates'][0]['phrase']
                                   if evidence['naming']['candidates'] else None),
                "candidate_names_json": json.dumps(evidence['naming']['candidates'], ensure_ascii=False),
            }
            for field in ('source_crop_path', 'source_mask_path', 'source_contour_path',
                          'bundle_crop_path', 'bundle_mask_path', 'bundle_contour_path',
                          'pair_path', 'overlay_path', 'verified_contour_path',
                          'neutral_crop_path', 'evidence_json_path'):
                item[field] = evidence[field]
                if not field.startswith('source_'):
                    item[field.replace('_path', '_relative_path')] = (
                        os.path.relpath(evidence[field], case_dir).replace('\\', '/')
                        if evidence[field] else None)
            shape_rows.append(item)
        self._write_csv(os.path.join(case_dir, "shape_prototypes.csv"), shape_rows)
        self._write_csv(os.path.join(case_dir, 'shape_coordinates.csv'), [
            {'prototype_id': k, 'activation': float(exp['shape_activation'][k]),
             'weight_difference_f_minus_b': float(exp['shape_weight_diff_f_minus_b'][k]),
             'signed_margin_contribution': float(q)}
            for k, q in enumerate(exp['shape_coordinate_margin'])])

        class_rows = self._class_logit_rows(exp)
        self._write_csv(os.path.join(case_dir, "class_logits.csv"), class_rows)
        waterfall_rows = self._waterfall_rows(exp, display_topk=exp['display_topk'])
        self._write_csv(os.path.join(case_dir, "waterfall.csv"), waterfall_rows)
        if render_plots:
            self._plot_margin_waterfall(exp, waterfall_rows, case_dir)

        summary = {
            key: exp[key] for key in [
                "split", "index", "sample_id", "sample_key", "true_class", "true_name",
                "dcbm_class", "dcbm_name", "sdrcbm_class", "sdrcbm_name",
                "category", "selection_rule", "selection_rank",
                "semantic_margin_f_minus_b", "context_margin_f_minus_b",
                "shape_margin_f_minus_b", "final_margin_f_minus_b",
                "semantic_bias_margin_f_minus_b", "shape_bias_margin_f_minus_b",
                "coverage_at_k",
            ]
        }
        summary.update({
            "shape_mode": shape_mode,
            "eta_s": float(getattr(self.rt, "eta_s", 1.0)),
            "input_source_path": input_source,
            "input_bundle_path": input_bundle,
            "waterfall_closure_error": abs(
                math.fsum(float(row['delta']) for row in waterfall_rows
                          if row['component_type'] != 'final_total')
                - exp["final_margin_f_minus_b"]),
            "export_topk": exp['topk'], "display_topk": exp['display_topk'],
            "waterfall_rendered": render_plots,
            "input_bundle_relative_path": os.path.relpath(input_bundle, case_dir).replace('\\', '/') if input_bundle else None,
            "test_mask_used_for_inference": False,
            "posthoc_test_teacher_exported": bool(posthoc_shape_match),
        })
        with open(os.path.join(case_dir, "case_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        layout = {
            "column_1": {"input": input_bundle, "labels": {
                "ground_truth": exp["true_name"],
                "dcbm_prediction": exp["dcbm_name"],
                "sdr_cbm_prediction": exp["sdrcbm_name"],
            }},
            "column_2": f"Use ranks 1-{exp['display_topk']} from semantic_prototypes.csv",
            "column_3": f"Use absolute_rank 1-{exp['display_topk']} from shape_prototypes.csv, "
                        "the same IDs as the waterfall. Pair the training crop with its "
                        "verified contour. Candidate names are exemplar labels only.",
            "column_4": "Use waterfall.csv or margin_waterfall.pdf when rendered. "
                        "Other shape and Shape bias must be retained for exact closure.",
        }
        with open(os.path.join(case_dir, "layout_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2, ensure_ascii=False)

        return {"case_dir": os.path.abspath(case_dir), **summary}

    def _class_logit_rows(self, exp):
        z_sem = np.asarray(exp["z_sem_row"])
        z_ctx = np.asarray(exp["z_ctx_row"])
        z_shape = np.asarray(exp["z_shape_row"])
        z_final = np.asarray(exp["z_final_row"])
        sem_order = np.argsort(z_sem)[::-1]
        final_order = np.argsort(z_final)[::-1]
        sem_rank = np.empty_like(sem_order); sem_rank[sem_order] = np.arange(1, len(sem_order) + 1)
        final_rank = np.empty_like(final_order); final_rank[final_order] = np.arange(1, len(final_order) + 1)
        rows = []
        for c in range(len(z_sem)):
            rows.append({
                "class_id": c,
                "class_name": self._class_name(c),
                "semantic_logit": float(z_sem[c]),
                "context_residual_logit": float(z_ctx[c]),
                "shape_residual_logit": float(z_shape[c]),
                "final_logit": float(z_final[c]),
                "semantic_rank": int(sem_rank[c]),
                "final_rank": int(final_rank[c]),
                "is_ground_truth": int(c == exp["true_class"]),
                "is_dcbm_prediction": int(c == exp["dcbm_class"]),
                "is_sdrcbm_prediction": int(c == exp["sdrcbm_class"]),
            })
        return rows

    def _waterfall_rows(self, exp, display_topk=None):
        rows = []
        cumulative = float(exp["semantic_margin_f_minus_b"])
        rows.append({
            "order": 0, "component_type": "semantic_total", "component_id": "semantic",
            "label": "DCBM semantic margin", "delta": cumulative,
            "cumulative_before": 0.0, "cumulative_after": cumulative,
        })
        order = 1
        ctx_delta = float(exp["context_margin_f_minus_b"])
        if abs(ctx_delta) > 1e-12:
            before = cumulative; cumulative += ctx_delta
            rows.append({
                "order": order, "component_type": "context_total", "component_id": "context",
                "label": "Context residual", "delta": ctx_delta,
                "cumulative_before": before, "cumulative_after": cumulative,
            })
            order += 1

        top_ids = exp["top_shape_abs_ids"]
        if display_topk is not None:
            top_ids = top_ids[:display_topk]
        top_sum = 0.0
        for prototype_id in top_ids:
            delta = float(exp["shape_coordinate_margin"][prototype_id])
            top_sum += delta
            before = cumulative; cumulative += delta
            rows.append({
                "order": order, "component_type": "shape_coordinate",
                "component_id": int(prototype_id), "label": f"Shape P{prototype_id}",
                "delta": delta, "cumulative_before": before,
                "cumulative_after": cumulative,
            })
            order += 1

        other_ids = np.ones(len(exp['shape_coordinate_margin']), dtype=bool)
        other_ids[top_ids] = False
        other_delta = float(np.sum(exp['shape_coordinate_margin'][other_ids], dtype=np.float64))
        before = cumulative; cumulative += other_delta
        rows.append({
            "order": order, "component_type": "shape_other", "component_id": "other",
            "label": "Other shape", "delta": other_delta,
            "cumulative_before": before, "cumulative_after": cumulative,
        })
        order += 1

        bias_delta = float(exp["shape_bias_margin_f_minus_b"])
        before = cumulative; cumulative += bias_delta
        rows.append({
            "order": order, "component_type": "shape_bias", "component_id": "bias",
            "label": "Shape bias", "delta": bias_delta,
            "cumulative_before": before, "cumulative_after": cumulative,
        })
        order += 1

        expected = float(exp["final_margin_f_minus_b"])
        closure = expected - cumulative
        if abs(closure) > 5e-5:
            raise RuntimeError(
                f"Waterfall does not close for index={exp['index']}: "
                f"computed={cumulative:.8f}, expected={expected:.8f}, error={closure:.3e}"
            )
        rows.append({
            "order": order, "component_type": "final_total", "component_id": "final",
            "label": "Final margin", "delta": expected,
            "cumulative_before": 0.0, "cumulative_after": expected,
        })
        return rows

    @staticmethod
    def _plot_margin_waterfall(exp, rows, case_dir):
        fig, ax = plt.subplots(figsize=(9.4, 4.9))
        x = np.arange(len(rows))
        labels = []
        for i, row in enumerate(rows):
            kind = row["component_type"]
            before = float(row["cumulative_before"])
            after = float(row["cumulative_after"])
            delta = float(row["delta"])
            labels.append(row["label"])
            if kind in ("semantic_total", "final_total"):
                bottom = min(0.0, after)
                height = abs(after)
                ax.bar(i, height, bottom=bottom, width=0.72,
                       facecolor="#777777", edgecolor="#333333", linewidth=1.0)
            else:
                bottom = min(before, after)
                height = abs(after - before)
                positive = delta >= 0
                ax.bar(i, height, bottom=bottom, width=0.72,
                       facecolor="white" if positive else "#B0B0B0",
                       edgecolor="#333333", linewidth=1.4)
                if i + 1 < len(rows):
                    ax.plot([i + 0.36, i + 0.64], [after, after],
                            color="#999999", linewidth=0.8)
            shown = after if kind in ("semantic_total", "final_total") else delta
            above = shown >= 0
            ax.annotate(f"{shown:+.2f}", (i, bottom + height if above else bottom),
                        xytext=(0, 6 if above else -6), textcoords='offset points',
                        ha='center', va='bottom' if above else 'top',
                        fontsize=10, fontweight='bold', color='black', clip_on=False)

        ax.axhline(0, color="#222222", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Logit margin (final class - DCBM class)")
        ax.set_title(
            f"{exp['category'].capitalize()}: {exp['dcbm_name']} to {exp['sdrcbm_name']}",
            fontsize=10, pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.6, zorder=0)
        y0, y1 = ax.get_ylim()
        pad = max((y1 - y0) * 0.08, 1e-3)
        ax.set_ylim(y0 - pad, y1 + pad)
        fig.tight_layout()
        fig.savefig(os.path.join(case_dir, "margin_waterfall.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(case_dir, "margin_waterfall.png"), dpi=320,
                    bbox_inches="tight")
        plt.close(fig)

    # ── plotting ──────────────────────────────────────────────
    def visualize_instance(self, split="test", idx=0, topk=5,
                           shape_mode="decoded_rel", posthoc_shape_match=False):
        N = self._get_sem_feature(split).shape[0]
        if idx < 0 or idx >= N:
            print(f"[Viz] skip {split}[{idx}], out of range 0..{N-1}")
            return None
        exp = self.explain_instance(split=split, idx=idx, topk=topk,
                                    shape_mode=shape_mode)
        case_dir = _ensure_dir(os.path.join(self.out_dir, f"{split}_{idx:05d}"))

        self._plot_decomp(exp, case_dir)
        self._plot_bar(exp, "sem_contrib", exp.get("top_sem_pos"), exp.get("top_sem_neg"),
                       "DCBM semantic contributions", case_dir, "semantic_contributions.png")
        if "ctx_contrib" in exp:
            self._plot_bar(exp, "ctx_contrib", exp.get("top_ctx_pos"), exp.get("top_ctx_neg"),
                           "Contextual residual contributions", case_dir, "context_contributions.png")
        if "shape_contrib" in exp:
            self._plot_bar(exp, "shape_contrib", exp.get("top_shape_pos"), exp.get("top_shape_neg"),
                           f"Shape residual contributions ({shape_mode})", case_dir,
                           "shape_contributions.png")
            self._save_shape_proto_info(exp, case_dir, shape_mode, posthoc_shape_match)
        self._save_report(exp, case_dir, topk, shape_mode, posthoc_shape_match)
        print(f"[Viz] {split}[{idx}] -> {case_dir}")
        return exp

    def _plot_decomp(self, exp, case_dir):
        pred = exp["pred_class"]
        zs = float(exp["z_sem"][pred])
        zc = 0.0 if exp["z_ctx"] is None else float(getattr(self.rt, "eta_c", 1.0) * exp["z_ctx"][pred])
        zh = 0.0 if exp["z_shape"] is None else float(getattr(self.rt, "eta_s", 1.0) * exp["z_shape"][pred])
        names = ["DCBM semantic", "Context residual", "Shape residual"]
        vals = [zs, zc, zh]
        plt.figure(figsize=(7, 4))
        plt.bar(names, vals, color=["tab:blue" if v >= 0 else "tab:red" for v in vals])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Logit contribution")
        plt.title(f"Logit decomposition: pred={exp['pred_name']}")
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, "logit_decomposition.png"), dpi=200)
        plt.close()

    def _plot_bar(self, exp, key, pos_list, neg_list, title, case_dir, fname):
        top = pos_list + neg_list
        top = list(dict.fromkeys(top))[:len(pos_list) + len(neg_list)]
        labels = [self._concept_name(i) for i in top]
        vals = [float(exp[key][i]) for i in top]
        plt.figure(figsize=(9, 4.8))
        plt.bar(range(len(vals)), vals, color=["tab:blue" if v >= 0 else "tab:red" for v in vals])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
        plt.ylabel("Contribution")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, fname), dpi=200)
        plt.close()

    def _save_shape_proto_info(self, exp, case_dir, shape_mode, posthoc_shape_match):
        proto_info = {}
        nearest = getattr(self.rt, "shape_proto_nearest", None)
        reliability = getattr(self.rt, "shape_reliability", None)
        teacher_vec = None
        if posthoc_shape_match:
            ta = self._get_attr_first(self.rt, [f"Ash_{exp['split']}_s"])
            if ta is not None:
                teacher_vec = _to_numpy(ta)[exp["idx"]]
        for k in exp.get("top_shape_pos", []) + exp.get("top_shape_neg", []):
            proto_info[int(k)] = {
                "contribution": float(exp["shape_contrib"][k]),
                "reliability": float(reliability[k])
                    if reliability is not None and k < len(reliability) else None,
                "nearest_training_masks": nearest.get(int(k), [])
                    if isinstance(nearest, dict) else [],
                "posthoc_sam_teacher_activation": float(teacher_vec[k])
                    if teacher_vec is not None else None,
            }
        with open(os.path.join(case_dir, "shape_prototype_info.json"), "w", encoding="utf-8") as f:
            json.dump(proto_info, f, indent=2, ensure_ascii=False)

    # ── global: co-occurrence matrix ──────────────────────────
    def visualize_cooccur_matrix(self, topn=30, by="frequency"):
        if not hasattr(self.rt, "Co") or self.rt.Co is None:
            print("[Viz] No co-occurrence matrix.")
            return
        C = self.rt.Co.detach().cpu().numpy() if torch.is_tensor(self.rt.Co) \
            else np.asarray(self.rt.Co)
        if by == "ctx_weight":
            if getattr(self.rt, "cb", None) is not None:
                W_ctx, _ = _linear_params(self.rt.cb)
                score = np.mean(np.abs(W_ctx), axis=0)
            else:
                print("[Viz] ctx_weight requested but no contextual head. Fallback to frequency.")
                score = self._get_sem_raw("train").mean(axis=0)
        else:
            score = self._get_sem_raw("train").mean(axis=0)
        idx = np.argsort(score)[::-1][:topn]
        sub = C[np.ix_(idx, idx)]
        labels = [self._concept_name(i) for i in idx]
        plt.figure(figsize=(10, 8))
        plt.imshow(sub, aspect="auto")
        plt.colorbar(label="C_ij = P(j|i)")
        plt.xticks(range(topn), labels, rotation=90, fontsize=7)
        plt.yticks(range(topn), labels, fontsize=7)
        plt.title(f"Co-occurrence matrix, top-{topn} by {by}")
        plt.tight_layout()
        path = os.path.join(self.out_dir, f"cooccur_matrix_top{topn}_{by}.png")
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[Viz] Co-occurrence matrix -> {path}")

    # ── text report ───────────────────────────────────────────
    def _save_report(self, exp, case_dir, topk, shape_mode, posthoc_shape_match):
        path = os.path.join(case_dir, "explanation_report.txt")
        pred = exp["pred_class"]
        zs = float(exp["z_sem"][pred])
        zc = 0.0 if exp["z_ctx"] is None else float(getattr(self.rt, "eta_c", 1.0) * exp["z_ctx"][pred])
        zh = 0.0 if exp["z_shape"] is None else float(getattr(self.rt, "eta_s", 1.0) * exp["z_shape"][pred])
        zf = float(exp["z_final"][pred])
        reliability = getattr(self.rt, "shape_reliability", None)
        nearest = getattr(self.rt, "shape_proto_nearest", None)
        teacher_vec = None
        if posthoc_shape_match:
            ta = self._get_attr_first(self.rt, [f"Ash_{exp['split']}_s"])
            if ta is not None:
                teacher_vec = _to_numpy(ta)[exp["idx"]]

        with open(path, "w", encoding="utf-8") as f:
            f.write("SDR-CBM Explanation Report\n" + "=" * 80 + "\n")
            f.write(f"split/index : {exp['split']} / {exp['idx']}\n")
            f.write(f"sample_key  : {exp['sample_key']}\n")
            f.write(f"true        : {exp['true_class']} / {exp['true_name']}\n")
            f.write(f"pred        : {exp['pred_class']} / {exp['pred_name']}\n\n")
            f.write("Logit decomposition\n" + "-" * 80 + "\n")
            f.write(f"z_sem           : {zs:+.4f}\n")
            f.write(f"eta_c * z_ctx   : {zc:+.4f}\n")
            f.write(f"eta_s * z_shape : {zh:+.4f}\n")
            f.write(f"final logit     : {zf:+.4f}\n\n")

            f.write("Top semantic evidence\n" + "-" * 80 + "\n")
            for i in exp["top_sem_pos"]:
                f.write(f"+ {i:04d} {self._concept_name(i):40s} {exp['sem_contrib'][i]:+.4f}\n")
            for i in exp["top_sem_neg"]:
                f.write(f"- {i:04d} {self._concept_name(i):40s} {exp['sem_contrib'][i]:+.4f}\n")
            f.write("\n")

            if "ctx_contrib" in exp:
                f.write("Top contextual evidence\n" + "-" * 80 + "\n")
                for j in exp["top_ctx_pos"]:
                    f.write(f"+ {j:04d} {self._concept_name(j):40s} {exp['ctx_contrib'][j]:+.4f}\n")
                    if "ctx_neighbors" in exp and j in exp["ctx_neighbors"]:
                        for nb in exp["ctx_neighbors"][j]:
                            f.write(f"    <- {nb['neighbor']:04d} {nb['neighbor_name']:35s} "
                                    f"C={nb['C_ij']:.3f} a_i={nb['a_i']:.3f} pair={nb['pair_score_approx']:+.4f}\n")
                for j in exp["top_ctx_neg"]:
                    f.write(f"- {j:04d} {self._concept_name(j):40s} {exp['ctx_contrib'][j]:+.4f}\n")
                f.write("\n")

            if "shape_contrib" in exp:
                f.write(f"Top shape evidence (mode={shape_mode})\n" + "-" * 80 + "\n")
                for k in exp["top_shape_pos"]:
                    rel = reliability[k] if reliability is not None and k < len(reliability) else None
                    teach = teacher_vec[k] if teacher_vec is not None else None
                    f.write(f"+ shape_{k:04d} contrib={exp['shape_contrib'][k]:+.4f} "
                            f"rel={rel if rel is not None else 'NA'} posthoc_sam={teach if teach is not None else 'NA'}\n")
                    if isinstance(nearest, dict) and k in nearest:
                        f.write(f"    nearest train masks: {nearest[k]}\n")
                for k in exp["top_shape_neg"]:
                    f.write(f"- shape_{k:04d} contrib={exp['shape_contrib'][k]:+.4f}\n")
                f.write("\n")

            f.write("Notes\n" + "-" * 80 + "\n")
            f.write("Shape residual uses CLIP-decoded shape features at inference.\n")
            if posthoc_shape_match:
                f.write("Post-hoc SAM teacher activations shown for visualization only, not used in inference.\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render a saved trace waterfall without retraining.')
    parser.add_argument('--render_case', required=True, type=Path,
                        help='Case directory containing case_summary.json and waterfall.csv.')
    options = parser.parse_args()
    with (options.render_case / 'case_summary.json').open(encoding='utf-8') as f:
        summary = json.load(f)
    with (options.render_case / 'waterfall.csv').open(encoding='utf-8-sig', newline='') as f:
        waterfall = list(csv.DictReader(f))
    SDRCBMVisualizer._plot_margin_waterfall(summary, waterfall, str(options.render_case))
    summary['waterfall_rendered'] = True
    with (options.render_case / 'case_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Waterfall saved to {options.render_case.resolve()}')
