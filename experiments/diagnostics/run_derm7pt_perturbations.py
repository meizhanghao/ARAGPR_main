"""Read-only Derm7pt geometry/appearance sensitivity pilot; no segmentation.

Uses fixed models from run_derm7pt_mechanism.py. All valid-teacher test images
are considered before inspecting model outcomes. Uncertain legacy RGB crop
registration is explicitly excluded. Originals are freshly encoded as the
reference for every perturbation, preventing old CLIP numerical drift from
being counted as perturbation sensitivity.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import clip
import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch

from aragpr.training import _extract_contour_raw,_extract_geometric_features
from experiments.controls.run_decoder_baselines import sha256,write_csv,write_json
from utils.residual_controls import frozen_semantic_logits
from utils.shape_perturbations import (appearance_blur,clean_binary,
                                      recover_crop_position,warp_image_and_masks)

from aragpr.paths import repository_root

ROOT = repository_root()
VARIANTS=("original","warp_x_010","warp_y_010","warp_x_020","warp_y_020",
          "inside_blur_15","inside_blur_30","outside_blur_15","outside_blur_30")


def region_descriptors(masks):
    values=[]
    for mask in masks:
        contour=_extract_contour_raw(mask)
        if cv2.contourArea(contour)<=100:
            raise ValueError("An edited retained contour fell below the area threshold.")
        values.append(_extract_geometric_features(contour))
    return np.stack(values).astype(np.float32)


def teacher_responses(descriptors,bank):
    scaled=bank["descriptor_scaler"].transform(descriptors).astype(np.float32)
    normed=scaled/(np.linalg.norm(scaled,axis=1,keepdims=True)+1e-8)
    pooled=(normed@bank["shape_prototypes"].T).max(0,keepdims=True)
    return bank["teacher_scaler"].transform(pooled).astype(np.float32)[0]


def load_assets(key,source,rows,image_index,asset_root,track):
    paths=image_index.get(Path(key).name.lower(),[])
    if len(paths)!=1: raise ValueError("Original image filename is missing or ambiguous.")
    image_path=paths[0]
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Cannot read original RGB image.")
    track[image_path]=sha256(image_path)
    source_path=Path(source)
    manifest_path=asset_root/"masks"/source_path.parent/f"{source_path.name}_MASKS_DONE.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    track[manifest_path]=sha256(manifest_path)
    masks,registrations=[],[]
    for row in rows:
        region=int(row["region_id"])
        if region not in manifest["region_ids"]: raise ValueError("Region absent from completion manifest.")
        name=f"{source_path.name}_OUT_MASK_{region}_X_.png"
        mask_path=asset_root/"masks"/source_path.parent/name
        crop_path=asset_root/"crops"/source_path.parent/name.replace("_MASK_","_CROP_").replace(".png",".jpg")
        raw=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
        crop=cv2.imread(str(crop_path),cv2.IMREAD_COLOR)
        if raw is None or crop is None or raw.shape!=crop.shape[:2]:
            raise ValueError("Missing or mismatched mask/crop pair.")
        clean=clean_binary(raw)
        actual=_extract_geometric_features(_extract_contour_raw(clean))
        expected=np.array([float(row[c]) for c in row if c not in ("source_image","region_id")],dtype=np.float32)
        error=float(np.max(np.abs(expected-actual)))
        if error>2e-5: raise ValueError(f"CSV descriptor replay failed: {error}")
        recorded=next((x for x in manifest.get("regions",[]) if x["region_id"]==region),None)
        if recorded is not None:
            bbox=recorded["bbox_xywh"]
            x,y,w,h=map(int,bbox)
            roi=image[y:y+h,x:x+w]
            if roi.shape!=crop.shape: raise ValueError("Recorded bbox exceeds original image.")
            diagnostic={"match_score":None,"alternative_score":None,"match_gap":None,
                        "crop_mae":float(np.abs(roi.astype(float)-crop).mean())}
            method="manifest_bbox"
        else:
            bbox,diagnostic=recover_crop_position(image,crop)
            x,y,w,h=bbox
            method="verified_legacy_RGB_crop_registration"
        if [w,h]!=[raw.shape[1],raw.shape[0]]: raise ValueError("ROI dimensions differ from binary mask.")
        full=np.zeros(image.shape[:2],np.uint8)
        full[y:y+h,x:x+w]=clean
        masks.append(full)
        track[mask_path]=sha256(mask_path)
        track[crop_path]=sha256(crop_path)
        registrations.append({"key":key,"source":source,"region_id":region,
            "bbox_x":x,"bbox_y":y,"bbox_w":w,"bbox_h":h,"method":method,
            **diagnostic,"descriptor_replay_max_error":error,
            "image_sha256":track[image_path],"mask_sha256":track[mask_path],
            "crop_sha256":track[crop_path]})
    return image,masks,registrations


def make_variants(image,masks):
    result={"original":(image,masks,{"jacobian_min":1.,"editable_fraction":0.})}
    identity,identity_masks,jac=warp_image_and_masks(image,masks,0.)
    if not np.array_equal(identity,image) or any(not np.array_equal(a,b) for a,b in zip(masks,identity_masks)):
        raise AssertionError("Identity warp changed input pixels/masks.")
    for name,strength in (("warp_x_010",.1),("warp_y_010",-.1),("warp_x_020",.2),("warp_y_020",-.2)):
        rgb,regions,jac=warp_image_and_masks(image,masks,strength)
        result[name]=(rgb,regions,{"jacobian_min":jac,"editable_fraction":float(np.any(rgb!=image,axis=2).mean())})
    for scope in ("inside","outside"):
        for suffix,sigma in (("15",1.5),("30",3.0)):
            rgb,regions,fraction=appearance_blur(image,masks,sigma,scope)
            if any(not np.array_equal(a,b) for a,b in zip(masks,regions)):
                raise AssertionError("Appearance-only edit changed a binary mask.")
            result[f"{scope}_blur_{suffix}"]=(rgb,regions,{"jacobian_min":1.,"editable_fraction":fraction})
    return result


def encode_images(model,preprocess,images,device,batch_size):
    pieces=[]
    with torch.inference_mode():
        for first in range(0,len(images),batch_size):
            batch=torch.stack([preprocess(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
                               for im in images[first:first+batch_size]]).to(device)
            pieces.append(model.encode_image(batch).float().cpu().numpy())
    return np.concatenate(pieces)


def safe_cosine(left,right):
    denom=np.linalg.norm(left,axis=-1)*np.linalg.norm(right,axis=-1)
    values=np.full(denom.shape,np.nan)
    np.divide(np.sum(left*right,axis=-1),denom,out=values,where=denom>1e-9)
    return values


def mean_interval(values,cases,replicates=2000):
    values=np.asarray(values,dtype=float)
    good=np.isfinite(values)
    values,cases=values[good],np.asarray(cases)[good]
    if not len(values): return (None,None,None)
    _,inv=np.unique(cases,return_inverse=True)
    totals=np.bincount(inv,weights=values)
    counts=np.bincount(inv)
    rng=np.random.default_rng(271828)
    inds=rng.integers(0,len(counts),(replicates,len(counts)))
    boots=totals[inds].sum(1)/counts[inds].sum(1)
    return float(values.mean()),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))


def analyze(args,out,features,targets,keys,details,cases,checkpoint):
    from experiments.controls.run_derm7pt_mechanism import MODES
    protocol=json.loads((args.models/"protocol.json").read_text())
    semantics=frozen_semantic_logits(features.reshape(-1,768),checkpoint,args.device).reshape(len(keys),len(VARIANTS),2)
    all_rows,aggregates=[],[]
    for seed in protocol["seeds"]:
        for mode in MODES:
            state=torch.load(args.models/f"seed_{seed}"/mode/"selected_model.pt",map_location="cpu",weights_only=False)
            h=state["input_scaler"].transform(features.reshape(-1,768)).astype(np.float32)
            decoded=(h@state["decoder_state"]["weight"].numpy().T+state["decoder_state"]["bias"].numpy()).reshape(len(keys),len(VARIANTS),128)
            selected=state["head"]["selected"]
            scaler=state["coordinate_scaler"]
            w=selected["state"]["weight"].numpy()
            b=selected["state"]["bias"].numpy()
            eta=selected["eta"]
            head_input=scaler.transform(decoded.reshape(-1,128)).reshape(decoded.shape)
            correction=eta*(head_input@w.T+b)
            logits=semantics+correction
            effective=eta*(w[1]-w[0])/scaler.scale_
            np.savez_compressed(out/f"sensitivity_seed_{seed}_{mode}.npz",decoded=decoded,
                semantic_logits=semantics,residual_logits=correction,final_logits=logits,
                keys=keys,variants=np.array(VARIANTS),teacher=targets)
            for vi,variant in enumerate(VARIANTS[1:],1):
                delta_t=targets[:,vi]-targets[:,0]
                delta_d=decoded[:,vi]-decoded[:,0]
                cos=safe_cosine(delta_t,delta_d)
                residual_delta=(correction[:,vi,1]-correction[:,vi,0])-(correction[:,0,1]-correction[:,0,0])
                teacher_delta=delta_t@effective
                semantic_delta=(semantics[:,vi,1]-semantics[:,vi,0])-(semantics[:,0,1]-semantics[:,0,0])
                final_delta=(logits[:,vi,1]-logits[:,vi,0])-(logits[:,0,1]-logits[:,0,0])
                np.testing.assert_allclose(final_delta,semantic_delta+residual_delta,rtol=1e-5,atol=2e-5)
                if "blur" in variant and np.any(delta_t!=0):
                    raise AssertionError("Appearance edit changed teacher although masks were fixed.")
                each=[]
                for i,key in enumerate(keys):
                    row={"key":key,"case":cases[i],"mode":mode,"seed":seed,"variant":variant,
                        "teacher_delta_rms":float(np.sqrt(np.mean(delta_t[i]**2))),
                        "decoded_delta_rms":float(np.sqrt(np.mean(delta_d[i]**2))),
                        "decoded_delta_in_train_sd_rms":float(np.sqrt(np.mean((delta_d[i]/scaler.scale_)**2))),
                        "delta_direction_cosine":float(cos[i]) if np.isfinite(cos[i]) else None,
                        "delta_semantic_margin":float(semantic_delta[i]),
                        "delta_residual_margin":float(residual_delta[i]),
                        "delta_teacher_projected_margin":float(teacher_delta[i]),
                        "delta_final_margin":float(final_delta[i]),
                        "prediction_changed":int(logits[i,vi].argmax()!=logits[i,0].argmax()),
                        **details[i][variant]}
                    all_rows.append(row)
                    each.append(row)
                rms=np.sqrt(np.mean(delta_d**2,axis=1))
                teacher_rms=np.sqrt(np.mean(delta_t**2,axis=1))
                cosine_mean,ci_low,ci_high=mean_interval(cos,cases)
                corr=None
                if "warp" in variant and np.std(residual_delta)>1e-8 and np.std(teacher_delta)>1e-8:
                    corr=float(spearmanr(residual_delta,teacher_delta).statistic)
                aggregates.append({"mode":mode,"seed":seed,"variant":variant,"n":len(keys),
                    "mean_teacher_delta_rms":float(teacher_rms.mean()),
                    "mean_decoded_delta_rms":float(rms.mean()),
                    "mean_decoded_delta_in_train_sd_rms":float(np.sqrt(np.mean((delta_d/scaler.scale_)**2,axis=1)).mean()),
                    "mean_delta_direction_cosine":cosine_mean,
                    "cosine_case_bootstrap_95_low":ci_low,"cosine_case_bootstrap_95_high":ci_high,
                    "spearman_teacher_vs_decoded_projected_margin_changes":corr,
                    "mean_abs_semantic_margin_change":float(np.abs(semantic_delta).mean()),
                    "mean_abs_residual_margin_change":float(np.abs(residual_delta).mean()),
                    "mean_abs_final_margin_change":float(np.abs(final_delta).mean()),
                    "prediction_changed_count":sum(r["prediction_changed"] for r in each)})
    write_csv(out/"per_image_sensitivity.csv",all_rows)
    write_csv(out/"sensitivity_summary.csv",aggregates)
    return aggregates


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    cache=args.cache_root/"derm7pt"
    model_protocol=json.loads((args.models/"protocol.json").read_text(encoding="utf-8"))
    if model_protocol["status"]!="complete": raise ValueError("Pairing experiment has not finished.")
    manifest=json.loads((cache/"dataset.json").read_text())
    bank_path=cache/"teacher_bank.pt"
    bank=torch.load(bank_path,map_location="cpu",weights_only=False)
    csv_path=Path(bank["shape_csv"])
    if sha256(csv_path)!=bank["shape_csv_sha256"]: raise ValueError("Shape source changed.")
    with np.load(cache/"test_frozen.npz",allow_pickle=False) as z:
        test_keys=z["sample_keys"].astype(str)
        labels=z["labels"].copy()
    with np.load(cache/"teacher_targets.npz",allow_pickle=False) as z:
        covered=z["test_covered"].astype(bool)
        cached_teacher=z["test_targets"].copy()
    source_rows={}
    with csv_path.open(newline="",encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            source_rows.setdefault(row["source_image"],[]).append(row)
    source_index={}
    for source in source_rows:
        source_index.setdefault(Path(source).name.lower(),[]).append(source)
    image_index={}
    for path in (ROOT/"dataset_all/derm7pt/images").rglob("*"):
        if path.is_file() and path.suffix.lower() in (".jpg",".jpeg",".png"):
            image_index.setdefault(path.name.lower(),[]).append(path)
    asset_root=ROOT/"data/Segments/Seg_embs/derm7pt_SAM2"
    protocol={"status":"running","created_utc":datetime.now(timezone.utc).isoformat(),
        "models":str(args.models.resolve()),"model_protocol_sha256":sha256(args.models/"protocol.json"),
        "code_sha256":sha256(Path(__file__)),"perturbation_code_sha256":sha256(ROOT/"utils/shape_perturbations.py"),
        "variants":VARIANTS,"population":"All retained valid-teacher test images; excluded only for failed input/mask/registration/replay checks before model inference.",
        "teacher":"Fixed train-only bank and scalers; existing masks only; max pooling across the same retained regions.",
        "warps":"Smooth local anisotropy +/-0.1 and +/-0.2 around dominant SAM region; same field for RGB and all retained masks; positive Jacobian. Not a pure or clinical shape counterfactual.",
        "appearance":"Gaussian blur at sigma1.5 or3.0 pixels at short-side224, inside eroded SAM union or outside dilated SAM union; mask and boundary band unchanged.",
        "registration":"Use recorded bbox when present; otherwise legacy RGB crop template match score>=0.98, peak gap>=0.005, MAE<=6. Reject uncertain coordinates.",
        "preprocessing":"Fresh original and edited images encoded with identical CLIP ViT-L/14 preprocessing; cached features only check numerical compatibility.",
        "interpretation":"Synthetic sensitivity, not proof of causal/clinical geometry or a label-preserving accuracy benchmark.",
        "mask_regeneration":False,"original_data_modified":False,"n_eligible":int(covered.sum())}
    write_json(out/"protocol.json",protocol)
    # Model outcomes cannot enter asset admission or edit generation.
    registrations,excluded,keys,images,targets,details,tracked=[],[],[],[],[],[],{}
    samples_dir=out/"examples"
    samples_dir.mkdir()
    example_keys=set(sorted(test_keys[covered],key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:4])
    for index in np.flatnonzero(covered):
        key=test_keys[index]
        try:
            sources=source_index.get(Path(key).stem.lower(),[])
            if len(sources)!=1: raise ValueError("Teacher source name missing or ambiguous.")
            source=sources[0]
            rgb,masks,records=load_assets(key,source,source_rows[source],image_index,asset_root,tracked)
            # Translation into full coordinates must reproduce cached original teachers.
            original_teacher=teacher_responses(region_descriptors(masks),bank)
            replay_error=float(np.max(np.abs(original_teacher-cached_teacher[index])))
            if replay_error>2e-3: raise ValueError(f"Full-coordinate teacher replay failed: {replay_error}")
            variants=make_variants(rgb,masks)
            edited_teachers=np.stack([teacher_responses(region_descriptors(variants[name][1]),bank)
                                     for name in VARIANTS])
            for vi,name in enumerate(VARIANTS):
                if "blur" in name and not np.array_equal(edited_teachers[vi],edited_teachers[0]):
                    raise AssertionError("Fixed-mask teacher changed under appearance blur.")
            for record in records:
                record["teacher_replay_max_error"]=replay_error
            registrations.extend(records)
            keys.append(key)
            images.extend([variants[name][0] for name in VARIANTS])
            targets.append(edited_teachers)
            details.append({name:variants[name][2] for name in VARIANTS})
            if key in example_keys:
                destination=samples_dir/Path(key).stem
                destination.mkdir()
                for name in VARIANTS:
                    cv2.imwrite(str(destination/f"{name}.png"),variants[name][0])
                cv2.imwrite(str(destination/"original_retained_region_union.png"),np.maximum.reduce(masks))
        except (ValueError,FileNotFoundError,KeyError) as error:
            excluded.append({"key":key,"reason":str(error)})
        if (len(keys)+len(excluded))%25==0:
            print(f"assets considered={len(keys)+len(excluded)} admitted={len(keys)} excluded={len(excluded)}",flush=True)
    if not keys: raise ValueError("No test images passed independent asset checks.")
    write_csv(out/"mask_registration.csv",registrations)
    write_csv(out/"excluded_images.csv",excluded)
    write_json(out/"input_assets.json",[{"path":str(path),"sha256":value} for path,value in tracked.items()])
    write_json(out/"edit_diagnostics.json",[{"key":key,"details":detail} for key,detail in zip(keys,details)])
    keys=np.array(keys)
    targets=np.stack(targets)
    clip_path=args.clip_weights.resolve()
    clip_hash=sha256(clip_path)
    model,preprocess=clip.load(str(clip_path),device=args.device,jit=False)
    model.eval()
    print(f"Encoding {len(images)} original/edited images; batch={args.batch_size}",flush=True)
    features=encode_images(model,preprocess,images,args.device,args.batch_size).reshape(len(keys),len(VARIANTS),768)
    del model,images
    torch.cuda.empty_cache()
    raw=torch.load(Path(manifest["sources"]["test"]["path"]),weights_only=False,map_location="cpu")
    old=np.stack([raw[key][0].float().numpy() for key in keys])
    cosine=np.sum(old*features[:,0],axis=1)/(np.linalg.norm(old,axis=1)*np.linalg.norm(features[:,0],axis=1))
    relative=np.linalg.norm(old-features[:,0],axis=1)/np.linalg.norm(old,axis=1)
    if float(cosine.min())<.9999 or float(relative.max())>.02:
        raise ValueError("Fresh CLIP original embeddings differ materially from retained inputs.")
    np.savez_compressed(out/"edited_features_and_teachers.npz",features=features,teacher=targets,
                        keys=keys,variants=np.array(VARIANTS),original_feature_cosine=cosine,
                        original_feature_relative_error=relative)
    from experiments.controls.run_derm7pt_mechanism import case_metadata
    cases,modalities=case_metadata(keys)
    checkpoint_path=Path(manifest["sources"]["checkpoint"]["path"])
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    aggregates=analyze(args,out,features,targets,keys,details,cases,checkpoint)
    for path,digest in tracked.items():
        if sha256(path)!=digest: raise AssertionError(f"Original input changed: {path}")
    if sha256(clip_path)!=clip_hash: raise AssertionError("CLIP weights changed.")
    protocol.update(status="complete",n_admitted=len(keys),n_excluded=len(excluded),
        n_edited_and_original_images=len(keys)*len(VARIANTS),
        clip_weights={"path":str(clip_path),"sha256":clip_hash},
        original_clip_min_cosine=float(cosine.min()),original_clip_max_relative_error=float(relative.max()),
        input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start,
        n_clinical=int((modalities=="clinic").sum()),n_dermoscopic=int((modalities=="derm").sum()))
    write_json(out/"protocol.json",protocol)
    print(json.dumps(protocol,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache-root",type=Path,default=ROOT/"outputs/residual_controls_20261018_paired01")
    parser.add_argument("--clip-weights",type=Path,default=Path.home() / ".cache/clip/ViT-L-14.pt")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--batch-size",type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    with threadpool_limits(limits=4): run(args)
