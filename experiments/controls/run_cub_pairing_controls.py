"""Matched CUB image--teacher pairing controls, compatible with the Derm7pt pilot.

All inputs are retained read-only caches. Fresh 128-dimensional decoders and
classification heads are fitted for paired, global-shuffled and within-class
shuffled training targets. Validation teachers are never shuffled. No original
model, mask, dataset split, or manuscript result is modified.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import binomtest,t,ttest_1samp
from threadpoolctl import threadpool_limits
import torch

from experiments.controls.run_decoder_baselines import (SPLITS,coverage_rows,fidelity_metrics,
                                  load_cached_inputs,sha256,write_csv,write_json)
from experiments.controls.run_derm7pt_mechanism import MODES,training_permutation,group_rows
from utils.residual_controls import (TrainingConfig,apply_affine_map,apply_residual,
    fit_residual,fit_teacher_decoder,frozen_semantic_logits,standardize_splits)

from aragpr.paths import repository_root

ROOT = repository_root()
DATASET='CUB_200_2011'


def stratified_bootstrap_indices(labels,replicates=4000,seed=314159):
    """Resample images within each class, preserving the retained class mix.

    A single shared design is used for both predictions in every contrast.
    This quantifies conditional test-image uncertainty, not full retraining
    uncertainty, and does not require fictitious medical case identifiers.
    """
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or replicates<1:
        raise ValueError('Bootstrap requires nonempty one-dimensional labels.')
    rng=np.random.default_rng(seed)
    indices=np.empty((replicates,len(labels)),dtype=np.int32)
    offset=0
    for cls in np.unique(labels):
        members=np.flatnonzero(labels==cls)
        indices[:,offset:offset+len(members)]=members[rng.integers(0,len(members),(replicates,len(members)))]
        offset+=len(members)
    assert offset==len(labels)
    return indices


def bootstrap_contrast(difference,indices):
    difference=np.asarray(difference,dtype=float)
    if difference.shape!=(indices.shape[1],): raise ValueError('Misaligned bootstrap contrast.')
    samples=100*difference[indices].mean(axis=1)
    return {'difference_pp':100*float(difference.mean()),
        'image_stratified_bootstrap_95_low_pp':float(np.quantile(samples,.025)),
        'image_stratified_bootstrap_95_high_pp':float(np.quantile(samples,.975)),
        'n_images':len(difference),'replicates':len(indices)}


def summarize(folder,args,labels,keys,valid,semantic,groups,fidelity,predictions):
    indices=stratified_bootstrap_indices(labels['test'])
    np.savez_compressed(folder/'test_membership.npz',keys=keys['test'],labels=labels['test'],
                        valid_teacher=valid['test'])
    write_json(folder/'bootstrap_design.json',{'unit':'test image','class_stratified':True,
        'replicates':4000,'random_seed':314159,'scope':'Fixed fitted models and retained test class mix.',
        'n_classes':len(np.unique(labels['test'])),'n_images':len(labels['test'])})
    aggregate=[]
    for mode in MODES:
        for subset in ('all','valid_teacher','missing_teacher'):
            rows=[r for r in groups if r['mode']==mode and r['split']=='test' and r['subset']==subset]
            if not rows: continue
            acc=np.array([r['accuracy_pct'] for r in rows])
            aggregate.append({'dataset':DATASET,'mode':mode,'subset':subset,'n':rows[0]['n'],
                'n_seeds':len(acc),'mean_accuracy_pct':float(acc.mean()),
                'sample_std_across_seeds_pct':float(acc.std(ddof=1)) if len(acc)>1 else None,
                'mean_delta_frozen_pp':float(np.mean([r['delta_frozen_pp'] for r in rows])),
                'mean_corrected':float(np.mean([r['n_corrected'] for r in rows])),
                'mean_harmed':float(np.mean([r['n_harmed'] for r in rows]))})
    comparisons,seed_differences=[],[]
    for control in MODES[1:]:
        differences=[]
        for seed in args.seeds:
            paired=predictions[seed,'paired'].argmax(1)==labels['test']
            other=predictions[seed,control].argmax(1)==labels['test']
            delta=paired.astype(float)-other.astype(float)
            differences.append(delta)
            wins,losses=int((paired&~other).sum()),int((~paired&other).sum())
            comparisons.append({'comparison':'paired-minus-'+control,'seed':seed,
                'n_paired_only_correct':wins,'n_control_only_correct':losses,
                'image_McNemar_exact_p':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,
                **bootstrap_contrast(delta,indices)})
        comparisons.append({'comparison':'paired-minus-'+control,'seed':'seed_average',
            'n_paired_only_correct':None,'n_control_only_correct':None,'image_McNemar_exact_p':None,
            **bootstrap_contrast(np.mean(differences,axis=0),indices)})
        per_seed=100*np.mean(differences,axis=1)
        mean=float(per_seed.mean())
        sd=float(per_seed.std(ddof=1)) if len(per_seed)>1 else None
        if sd is not None and sd>1e-12:
            halfwidth=float(t.ppf(.975,len(per_seed)-1)*sd/np.sqrt(len(per_seed)))
            p=float(ttest_1samp(per_seed,0).pvalue)
        else: halfwidth,p=None,None
        seed_differences.append({'comparison':'paired-minus-'+control,'n_seeds':len(per_seed),
            'mean_difference_pp':mean,'sample_std_of_paired_seed_differences_pp':sd,
            'paired_seed_t_95_low_pp':mean-halfwidth if halfwidth is not None else None,
            'paired_seed_t_95_high_pp':mean+halfwidth if halfwidth is not None else None,
            'paired_seed_t_p':p,'scope':'Exploratory three-seed comparison; fixed Frozen/teacher bank and test set.'})
    write_csv(folder/'classification_groups.csv',groups)
    write_csv(folder/'classification_summary.csv',aggregate)
    write_csv(folder/'decoder_fidelity.csv',fidelity)
    write_csv(folder/'paired_comparisons.csv',comparisons)
    write_csv(folder/'paired_seed_differences.csv',seed_differences)
    write_json(folder/'summary.json',{'classification':aggregate,'paired_comparisons':comparisons,
        'paired_seed_differences':seed_differences,
        'interpretation':'Local validation-selected checkpoints. Do not replace manuscript server last-15-epoch means.'})
    return aggregate,comparisons


def replay_predictions(state,h,semantic,keys,labels,folder,device):
    decoded=apply_affine_map(h,state['decoder_state'],device)
    branch={s:state['coordinate_scaler'].transform(x).astype(np.float32) for s,x in decoded.items()}
    logits={}
    for split in SPLITS:
        logits[split]=apply_residual(branch[split],semantic[split],state['head'],device)
        selected=state['head']['selected']
        manual=semantic[split]+selected['eta']*(branch[split]@selected['state']['weight'].numpy().T+
                                               selected['state']['bias'].numpy())
        np.testing.assert_allclose(logits[split],manual,rtol=2e-5,atol=3e-5)
        with np.load(folder/f'{split}_predictions.npz',allow_pickle=False) as saved:
            assert np.array_equal(saved['sample_keys'].astype(str),keys[split])
            assert np.array_equal(saved['labels'],labels[split])
            np.testing.assert_allclose(saved['decoded'],decoded[split],rtol=2e-5,atol=3e-5)
            np.testing.assert_allclose(saved['logits'],logits[split],rtol=2e-5,atol=3e-5)
    return decoded,logits


def run(args):
    start=time.monotonic()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    h,teacher,valid,keys,labels,original,sources=load_cached_inputs(args.cache_root,DATASET)
    if teacher['train'].shape[1]!=128 or len(original['class_order'])!=200:
        raise ValueError('This protocol requires geo11 K_s=128 and 200 CUB classes.')
    checkpoint_path=Path(original['sources']['checkpoint']['path'])
    if sha256(checkpoint_path)!=original['sources']['checkpoint']['sha256']:
        raise ValueError('Frozen semantic checkpoint changed.')
    sources['semantic_checkpoint']={'path':str(checkpoint_path),'sha256':sha256(checkpoint_path)}
    if 'shape_csv' in original:
        path=Path(original['shape_csv']['path'])
        assert sha256(path)==original['shape_csv']['sha256']
        sources['shape_csv']={'path':str(path),'sha256':sha256(path)}
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    scaler=torch.load(args.cache_root/DATASET/'input_scaler.pt',map_location='cpu',weights_only=False)['h_scaler']
    semantic,checks={},[]
    for split in SPLITS:
        raw=scaler.inverse_transform(h[split]).astype(np.float32)
        replay=frozen_semantic_logits(raw,checkpoint,args.device)
        with np.load(args.cache_root/DATASET/f'{split}_frozen.npz',allow_pickle=False) as z:
            semantic[split]=z['logits'].copy()
        error=float(np.max(np.abs(replay-semantic[split])))
        if error>2e-4 or not np.array_equal(replay.argmax(1),semantic[split].argmax(1)):
            raise ValueError(f'Frozen replay failed: {split} {error}')
        checks.append({'split':split,'max_logit_error':error,'all_predictions_match':True,
                       'accuracy_pct':100*float((replay.argmax(1)==labels[split]).mean())})
    config=TrainingConfig(epochs=args.head_epochs,patience=30)
    files=[Path(__file__),ROOT/'experiments/controls/run_derm7pt_mechanism.py',ROOT/'experiments/controls/run_decoder_baselines.py',
           ROOT/'utils/residual_controls.py']
    protocol={'status':'running','dataset':DATASET,'created_utc':datetime.now(timezone.utc).isoformat(),
        'seeds':args.seeds,'modes':MODES,'sources':sources,'n_classes':200,'teacher_dimension':128,
        'class_order':original['class_order'],'split_membership':original['split_membership'],
        'teacher_coverage':original['teacher_coverage'],'fixed_frozen_checkpoint_across_seeds':True,
        'decoder_epochs':args.decoder_epochs,'decoder_patience':30,'decoder_loss':'SmoothL1',
        'decoder_optimizer':'AdamW(lr=0.001, weight_decay=0.0001), batch64, cosine schedule',
        'decoder_selection':'Same UNPERMUTED valid validation teachers in every mode; minimum loss.',
        'shuffle':'Complete vectors, training valid-teacher images only; no validation/test target permutation.',
        'classifier':asdict(config),'coordinate_scaling':'Each branch uses the SAME train-only coordinate standardization rule.',
        'head_selection':'Highest validation accuracy; first eta/epoch wins ties; test never selects.',
        'reporting':'Validation-selected single checkpoints; sample SD across actual decoder/head seeds. Not manuscript server last-15-epoch means.',
        'frozen_replay':checks,'code_sources':[{'path':str(p),'sha256':sha256(p)} for p in files],
        'python':platform.python_version(),'torch':torch.__version__,'device':args.device,
        'gpu':torch.cuda.get_device_name(0) if 'cuda' in args.device else None}
    write_json(out/'protocol.json',protocol)
    write_csv(out/'teacher_coverage.csv',coverage_rows(DATASET,labels,valid,original['class_order']))
    groups,fidelities,predictions,pairing_diagnostics=[],[],{},[]
    for seed in args.seeds:
        for mode in MODES:
            folder=out/f'seed_{seed}'/mode
            folder.mkdir(parents=True)
            def progress(message):print(f'[{DATASET}/{seed}/{mode}] {message}',flush=True)
            yteacher=teacher['train'][valid['train']]
            yclass=labels['train'][valid['train']]
            permutation=training_permutation(yclass,mode,seed)
            assert np.array_equal(np.sort(permutation),np.arange(len(permutation)))
            if mode=='within_class_shuffled':assert np.array_equal(yclass[permutation],yclass)
            delta=yteacher[permutation]-yteacher
            pairing_diagnostics.append({'seed':seed,'mode':mode,'n_pairs':len(permutation),
                'same_image_pair_count':int((permutation==np.arange(len(permutation))).sum()),
                'same_class_pair_count':int((yclass[permutation]==yclass).sum()),
                'numerically_unchanged_target_count':int((np.max(np.abs(delta),axis=1)<1e-7).sum()),
                'target_change_mean_rms':float(np.sqrt(np.mean(delta**2,axis=1)).mean())})
            np.savez_compressed(folder/'training_teacher_pairing.npz',permutation=permutation,
                input_keys=keys['train'][valid['train']],target_source_keys=keys['train'][valid['train']][permutation])
            fitted=fit_teacher_decoder(h['train'][valid['train']],yteacher[permutation],
                h['val'][valid['val']],teacher['val'][valid['val']],seed=seed,device=args.device,
                epochs=args.decoder_epochs,patience=30,progress=progress)
            decoded=apply_affine_map(h,fitted['state'],args.device)
            for split in ('val','test'):
                metrics,per_coord=fidelity_metrics(teacher[split][valid[split]],decoded[split][valid[split]])
                fidelities.append({'mode':mode,'seed':seed,'split':split,**metrics})
                write_csv(folder/f'{split}_coordinate_fidelity.csv',[
                    {'prototype':i,**{k:float(v[i]) if np.isfinite(v[i]) else None for k,v in per_coord.items()}}
                    for i in range(128)])
            branch,coordinate_scaler=standardize_splits(decoded)
            head=fit_residual(branch['train'],semantic['train'],labels['train'],
                branch['val'],semantic['val'],labels['val'],config=config,seed=seed,
                device=args.device,progress=progress)
            state={'dataset':DATASET,'mode':mode,'seed':seed,'decoder_state':fitted['state'],
                'decoder_best_epoch':fitted['best_epoch'],'decoder_val_loss':fitted['val_loss'],
                'input_scaler':scaler,'coordinate_scaler':coordinate_scaler,'head':head}
            torch.save(state,folder/'selected_model.pt')
            write_csv(folder/'decoder_history.csv',fitted['history'])
            write_csv(folder/'head_history.csv',head['history'])
            write_json(folder/'selection.json',{k:v for k,v in head['selected'].items() if k!='state'})
            for split in SPLITS:
                logits=apply_residual(branch[split],semantic[split],head,args.device)
                np.savez_compressed(folder/f'{split}_predictions.npz',logits=logits,
                    semantic_logits=semantic[split],labels=labels[split],sample_keys=keys[split],
                    valid_teacher=valid[split],decoded=decoded[split])
                groups.extend(group_rows(mode,seed,split,logits,semantic[split],labels[split],valid[split]))
                if split=='test':predictions[seed,mode]=logits
            stored=torch.load(folder/'selected_model.pt',map_location='cpu',weights_only=False)
            replay_predictions(stored,h,semantic,keys,labels,folder,args.device)
            write_json(folder/'verification.json',{'status':'complete','prediction_replay_passed':True,
                'splits':SPLITS,'decoder_parameters':(768+1)*128,'head_parameters':(128+1)*200,
                'selected_eta':head['selected']['eta'],'selected_epoch':head['selected']['best_epoch']})
            write_csv(out/'classification_groups.csv',groups)
            write_csv(out/'decoder_fidelity.csv',fidelities)
            write_csv(out/'pairing_diagnostics.csv',pairing_diagnostics)
            row=next(r for r in groups if r['seed']==seed and r['mode']==mode and r['split']=='test' and r['subset']=='all')
            progress(f"FINAL test={row['accuracy_pct']:.4f}% corrected={row['n_corrected']} harmed={row['n_harmed']}")
    aggregate,comparisons=summarize(out,args,labels,keys,valid,semantic,groups,fidelities,predictions)
    for record in sources.values():
        if sha256(Path(record['path']))!=record['sha256']:raise AssertionError('An immutable input changed.')
    protocol.update(status='complete',elapsed_seconds=time.monotonic()-start,input_hashes_unchanged=True,
                    n_models_completed=len(args.seeds)*len(MODES),prediction_replay_passed=True)
    write_json(out/'protocol.json',protocol)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,default=ROOT/'outputs/residual_controls_20261018_paired01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--decoder-epochs',type=int,default=300)
    parser.add_argument('--head-epochs',type=int,default=100)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):parser.error('Duplicate seeds.')
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(args)
