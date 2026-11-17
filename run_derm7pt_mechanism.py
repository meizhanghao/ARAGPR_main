"""Matched Derm7pt teacher-pairing controls, using read-only retained caches.

This diagnostic uses validation-selected single checkpoints. It does not update
the manuscript's server results or estimate seed variance from epoch variation.
Run from this project with its existing PyTorch environment.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
import torch

from run_decoder_baselines import (SPLITS, coverage_rows, fidelity_metrics,
                                  load_cached_inputs, sha256, write_csv, write_json)
from utils.residual_controls import (TrainingConfig, apply_affine_map, apply_residual,
    fit_residual, fit_teacher_decoder, frozen_semantic_logits, prediction_metrics,
    standardize_splits)

ROOT = Path(__file__).resolve().parent
MODES = ("paired", "shuffled", "within_class_shuffled")


def training_permutation(labels, mode, seed):
    """Permute COMPLETE teacher vectors inside training only, never coordinates.

    A coordinate permutation followed by a refitted linear head is a re-labeling
    of axes, not a valid test of whether image--teacher correspondence matters.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed + 3000001)
    if mode == "paired":
        return np.arange(len(labels))
    if mode == "shuffled":
        return rng.permutation(len(labels))
    if mode != "within_class_shuffled":
        raise ValueError(mode)
    indices = np.arange(len(labels))
    for cls in np.unique(labels):
        members = np.flatnonzero(labels == cls)
        indices[members] = rng.permutation(members)
    if not np.array_equal(labels[indices], labels):
        raise AssertionError("Within-class shuffle changed class membership.")
    return indices


def group_rows(mode, seed, split, logits, base, labels, valid):
    rows = []
    groups = [("all", np.ones(len(labels), bool)), ("valid_teacher", valid),
              ("missing_teacher", ~valid)]
    for cls in np.unique(labels):
        groups.append((f"class_{cls}", labels == cls))
    for name, members in groups:
        if members.any():
            rows.append({"mode": mode, "seed": seed, "split": split,
                         "subset": name,
                         **prediction_metrics(logits[members], labels[members], base[members])})
    return rows


def case_metadata(keys, metadata_path=None):
    """Map clinic and dermoscopic images to recorded cases, without guessing."""
    import csv
    index = {}
    if metadata_path is None:
        import os
        data_root = Path(os.environ.get("ARAGPR_DATA_ROOT", ROOT))
        metadata_path = data_root / "dataset_all/derm7pt/meta/meta.csv"
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            for modality in ("clinic", "derm"):
                if row[modality]:
                    key = Path(row[modality]).name.lower()
                    value = (str(row["case_num"]), modality)
                    if key in index and index[key] != value:
                        if index[key][0] != value[0]:
                            raise ValueError("One metadata image name belongs to multiple cases.")
                        # Nine records repeat one image in both modality columns.
                        # Its case remains unambiguous; do not invent a modality.
                        value = (value[0], "clinic_and_derm_metadata")
                    index[key] = value
    cases, modalities = [], []
    for key in keys:
        case, modality = index[Path(str(key)).name.lower()]
        cases.append(case)
        modalities.append(modality)
    return np.array(cases), np.array(modalities)


def cluster_bootstrap_difference(differences, cases, seed=314159, replicates=4000):
    """Case-cluster resampling of a fixed set of predictions; no refitting.

    Differences can be seed-averaged per-image correctness. This interval is
    conditional on the retained split and fitted models, not a new independent
    patient cohort or a population estimate across training seeds.
    """
    cases = np.asarray(cases)
    unique, inverse = np.unique(cases, return_inverse=True)
    totals = np.bincount(inverse, weights=differences)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique), (replicates, len(unique)))
    values = 100.0 * totals[samples].sum(1) / counts[samples].sum(1)
    return {"difference_pp": 100.0 * float(np.mean(differences)),
            "case_bootstrap_95_low_pp": float(np.quantile(values, .025)),
            "case_bootstrap_95_high_pp": float(np.quantile(values, .975)),
            "n_test_cases": len(unique), "replicates": replicates}


def summarize(out, args, labels, keys, valid, semantic, rows, fidelity, predictions):
    case_ids, modalities = case_metadata(keys["test"])
    np.savez_compressed(out / "test_metadata.npz", keys=keys["test"],
                        case_ids=case_ids, modality=modalities, labels=labels["test"],
                        valid_teacher=valid["test"])
    cases_by_split = {s: set(case_metadata(keys[s])[0]) for s in SPLITS}
    overlaps = {f"{a}_{b}": len(cases_by_split[a] & cases_by_split[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    aggregate = []
    for mode in MODES:
        for subset in ("all", "valid_teacher", "missing_teacher"):
            chosen = [r for r in rows if r["mode"] == mode and r["split"] == "test"
                      and r["subset"] == subset]
            acc = [r["accuracy_pct"] for r in chosen]
            aggregate.append({"mode": mode, "subset": subset, "n": chosen[0]["n"],
                "n_seeds": len(acc), "mean_accuracy_pct": float(np.mean(acc)),
                "sample_std_across_seeds_pct": float(np.std(acc, ddof=1)) if len(acc)>1 else None,
                "mean_delta_frozen_pp": float(np.mean([r["delta_frozen_pp"] for r in chosen])),
                "mean_corrected": float(np.mean([r["n_corrected"] for r in chosen])),
                "mean_harmed": float(np.mean([r["n_harmed"] for r in chosen]))})
    comparisons = []
    for control in MODES[1:]:
        diffs = []
        for seed in args.seeds:
            good = predictions[seed, "paired"].argmax(1) == labels["test"]
            other = predictions[seed, control].argmax(1) == labels["test"]
            diffs.append(good.astype(float) - other.astype(float))
            wins, losses = int((good & ~other).sum()), int((~good & other).sum())
            comparisons.append({"comparison": "paired-minus-" + control, "seed": seed,
                "n_paired_only_correct": wins, "n_control_only_correct": losses,
                "image_McNemar_exact_p_descriptive": float(binomtest(wins,wins+losses).pvalue)
                    if wins+losses else 1.0,
                **cluster_bootstrap_difference(diffs[-1], case_ids)})
        comparisons.append({"comparison": "paired-minus-" + control, "seed": "seed_average",
            "n_paired_only_correct": None, "n_control_only_correct": None,
            "image_McNemar_exact_p_descriptive": None,
            **cluster_bootstrap_difference(np.mean(diffs, axis=0), case_ids)})
    write_csv(out / "classification_groups.csv", rows)
    write_csv(out / "classification_summary.csv", aggregate)
    write_csv(out / "decoder_fidelity.csv", fidelity)
    write_csv(out / "paired_comparisons.csv", comparisons)
    write_json(out / "summary.json", {"classification": aggregate,
        "comparisons": comparisons, "case_overlap_counts": overlaps,
        "warning": "Retained image partitions overlap in recorded cases. These are local diagnostics, not independent clinical validation."})
    return aggregate, comparisons, overlaps


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    h, teacher, valid, keys, labels, cache_manifest, sources = load_cached_inputs(
        args.cache_root, "derm7pt")
    cache = args.cache_root / "derm7pt"
    checkpoint_path = Path(cache_manifest["sources"]["checkpoint"]["path"])
    if sha256(checkpoint_path) != cache_manifest["sources"]["checkpoint"]["sha256"]:
        raise ValueError("Semantic checkpoint changed.")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    semantic = {}
    input_scaler = torch.load(cache/"input_scaler.pt", weights_only=False, map_location="cpu")["h_scaler"]
    frozen_checks = []
    for s in SPLITS:
        # Inverse transform recovers the unnormalised CLIP embeddings.
        raw_h = input_scaler.inverse_transform(h[s]).astype(np.float32)
        replay = frozen_semantic_logits(raw_h, checkpoint, args.device)
        with np.load(cache/f"{s}_frozen.npz", allow_pickle=False) as z:
            semantic[s] = z["logits"].copy()
        error = float(np.max(np.abs(replay-semantic[s])))
        if error > 2e-4 or not np.array_equal(replay.argmax(1),semantic[s].argmax(1)):
            raise ValueError(f"Frozen replay mismatch: {s} {error}")
        frozen_checks.append({"split":s,"max_logit_error":error,"predictions_identical":True})
    config = TrainingConfig(epochs=args.head_epochs, patience=30)
    protocol = {"status":"running", "dataset":"derm7pt", "created_utc":datetime.now(timezone.utc).isoformat(),
        "seeds":args.seeds, "modes":MODES, "sources":sources,
        "semantic_checkpoint":{"path":str(checkpoint_path),"sha256":sha256(checkpoint_path)},
        "code":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__))},
        "teacher_dimension":128,"decoder_epochs":args.decoder_epochs,"decoder_patience":30,
        "decoder_selection":"Lowest loss against the SAME UNPERMUTED valid validation teachers for all modes.",
        "shuffle":"Full training vectors only; validation/test teacher membership and values unchanged.",
        "classifier":asdict(config),"coordinate_scaling":"Identical train-only coordinate StandardScaler rule for every mode; final test never fits a scaler.",
        "head_selection":"Highest validation accuracy, first eta/epoch on ties. No test selection.",
        "reporting":"Validation-selected single checkpoints, not the manuscript server last-15-epoch means.",
        "frozen_replay":frozen_checks,"torch":torch.__version__,"python":platform.python_version(),
        "device":args.device,"gpu":torch.cuda.get_device_name(0) if "cuda" in args.device else None}
    write_json(out/"protocol.json",protocol)
    write_csv(out/"teacher_coverage.csv",coverage_rows("derm7pt",labels,valid,cache_manifest["class_order"]))
    results, fidelities, predictions = [], [], {}
    for seed in args.seeds:
        for mode in MODES:
            folder = out/f"seed_{seed}"/mode
            folder.mkdir(parents=True)
            perm = training_permutation(labels["train"][valid["train"]],mode,seed)
            np.savez_compressed(folder/"training_teacher_pairing.npz",permutation=perm,
                input_keys=keys["train"][valid["train"]],
                target_source_keys=keys["train"][valid["train"]][perm])
            progress = lambda msg: print(f"[{seed}/{mode}] {msg}",flush=True)
            fitted = fit_teacher_decoder(h["train"][valid["train"]],
                teacher["train"][valid["train"]][perm],
                h["val"][valid["val"]],teacher["val"][valid["val"]],
                seed=seed,device=args.device,epochs=args.decoder_epochs,patience=30,progress=progress)
            coordinates = apply_affine_map(h,fitted["state"],args.device)
            for s in ("val","test"):
                metrics, per_coord = fidelity_metrics(teacher[s][valid[s]],coordinates[s][valid[s]])
                fidelities.append({"mode":mode,"seed":seed,"split":s,**metrics})
                write_csv(folder/f"{s}_coordinate_fidelity.csv",[
                    {"prototype":i,**{name:float(values[i]) if np.isfinite(values[i]) else None
                        for name,values in per_coord.items()}} for i in range(128)])
            branch, scaler = standardize_splits(coordinates)
            head = fit_residual(branch["train"],semantic["train"],labels["train"],
                branch["val"],semantic["val"],labels["val"],config=config,seed=seed,
                device=args.device,progress=progress)
            state = {"decoder_state":fitted["state"],"decoder_best_epoch":fitted["best_epoch"],
                "decoder_val_loss":fitted["val_loss"],"coordinate_scaler":scaler,
                "input_scaler":input_scaler,"head":head,"mode":mode,"seed":seed}
            torch.save(state,folder/"selected_model.pt")
            write_csv(folder/"decoder_history.csv",fitted["history"])
            write_csv(folder/"head_history.csv",head["history"])
            write_json(folder/"selection.json",{k:v for k,v in head["selected"].items() if k!="state"})
            for s in SPLITS:
                logits = apply_residual(branch[s],semantic[s],head,args.device)
                np.savez_compressed(folder/f"{s}_predictions.npz",logits=logits,
                    semantic_logits=semantic[s],labels=labels[s],sample_keys=keys[s],
                    valid_teacher=valid[s],decoded=coordinates[s])
                # Reload the exact serialized head and verify inference independently.
                stored = torch.load(folder/"selected_model.pt",weights_only=False,map_location="cpu")
                x = stored["coordinate_scaler"].transform(coordinates[s])
                selected = stored["head"]["selected"]
                manual = semantic[s] + selected["eta"] * (x @ selected["state"]["weight"].numpy().T
                                                           + selected["state"]["bias"].numpy())
                np.testing.assert_allclose(logits,manual,rtol=2e-5,atol=2e-5)
                results.extend(group_rows(mode,seed,s,logits,semantic[s],labels[s],valid[s]))
                if s=="test": predictions[seed,mode] = logits
            test_row = next(r for r in results if r["seed"]==seed and r["mode"]==mode and
                            r["split"]=="test" and r["subset"]=="all")
            progress(f"FINAL test={test_row['accuracy_pct']:.4f}% corrected={test_row['n_corrected']} harmed={test_row['n_harmed']}")
    aggregate, comparisons, overlaps = summarize(out,args,labels,keys,valid,semantic,results,fidelities,predictions)
    # Every input file used by this runner must still have its initial hash.
    for source in sources.values():
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("A read-only source changed during the run.")
    if sha256(checkpoint_path) != protocol["semantic_checkpoint"]["sha256"]:
        raise ValueError("Semantic checkpoint changed during the run.")
    protocol.update(status="complete",elapsed_seconds=time.monotonic()-start,
                    input_hashes_unchanged=True,case_overlap_counts=overlaps)
    write_json(out/"protocol.json",protocol)
    print(json.dumps(aggregate,indent=2),flush=True)


def finalize_existing(args):
    """Recover a completed fit after an output/metadata-only interruption.

    Replays every serialized decoder and head before producing summaries;
    never resumes or changes model fitting or selection.
    """
    out=args.output.resolve()
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    if list(protocol["seeds"])!=list(args.seeds): raise ValueError("Seed list differs.")
    h,teacher,valid,keys,labels,manifest,sources=load_cached_inputs(args.cache_root,"derm7pt")
    semantic={}
    for s in SPLITS:
        with np.load(args.cache_root/"derm7pt"/f"{s}_frozen.npz",allow_pickle=False) as z:
            semantic[s]=z["logits"].copy()
    rows,fidelity,predictions=[],[],{}
    for seed in args.seeds:
        for mode in MODES:
            folder=out/f"seed_{seed}"/mode
            model=torch.load(folder/"selected_model.pt",weights_only=False,map_location="cpu")
            decoded=apply_affine_map(h,model["decoder_state"],args.device)
            branch={s:model["coordinate_scaler"].transform(x).astype(np.float32) for s,x in decoded.items()}
            for s in SPLITS:
                logits=apply_residual(branch[s],semantic[s],model["head"],args.device)
                with np.load(folder/f"{s}_predictions.npz",allow_pickle=False) as saved:
                    assert np.array_equal(saved["sample_keys"].astype(str),keys[s])
                    assert np.array_equal(saved["labels"],labels[s])
                    np.testing.assert_allclose(logits,saved["logits"],rtol=2e-5,atol=2e-5)
                    np.testing.assert_allclose(decoded[s],saved["decoded"],rtol=2e-5,atol=2e-5)
                rows.extend(group_rows(mode,seed,s,logits,semantic[s],labels[s],valid[s]))
                if s in ("val","test"):
                    metrics,_=fidelity_metrics(teacher[s][valid[s]],decoded[s][valid[s]])
                    fidelity.append({"mode":mode,"seed":seed,"split":s,**metrics})
                if s=="test": predictions[seed,mode]=logits
    aggregate,comparisons,overlaps=summarize(out,args,labels,keys,valid,semantic,rows,fidelity,predictions)
    for value in sources.values():
        assert sha256(Path(value["path"]))==value["sha256"]
    protocol.update(status="complete",input_hashes_unchanged=True,case_overlap_counts=overlaps,
        finalized_utc=datetime.now(timezone.utc).isoformat(),
        finalization_code_sha256=sha256(Path(__file__)),
        finalization_note="Recovered summaries after allowing same-case images repeated in both metadata modality columns. All nine saved fits replayed unchanged; no retraining or reselection.")
    write_json(out/"protocol.json",protocol)
    print(json.dumps(aggregate,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root",type=Path,default=ROOT/"outputs/residual_controls_20261018_paired01")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--decoder-epochs",type=int,default=300)
    parser.add_argument("--head-epochs",type=int,default=100)
    parser.add_argument("--finalize-only",action="store_true")
    arguments=parser.parse_args()
    if len(set(arguments.seeds))!=len(arguments.seeds): parser.error("Duplicate seeds.")
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        (finalize_existing if arguments.finalize_only else run)(arguments)
