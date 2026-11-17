"""Local candidate naming of six verified TRAINING exemplars, never classifier inference.

The images come from existing explanation bundles. This is a feasibility probe,
not a validation of names or of semantic consistency across a prototype bank.
No external LLM is called, no weights are downloaded, no labels are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from run_decoder_baselines import sha256, write_csv, write_json


ROOT=Path(__file__).resolve().parent
EXEMPLARS={
    'CUB_200_2011':(ROOT/'outputs/explanation_trace_cub_binary_v3/CUB_200_2011/decoded/prototype_assets',[127,91,49]),
    'derm7pt':(ROOT/'outputs/explanation_trace/derm7pt/decoded/prototype_assets',[45,113,5]),
}
SHAPES=['a round region','an oval region','an elongated narrow region',
        'a thin curved strip','a roughly triangular region','a rectangular region',
        'a compact irregular region','a lobulated region','a region with a concave outline']
PARTS={
    'CUB_200_2011':['a bird beak','a bird head','a bird wing','a bird tail','a bird leg',
        'a bird foot','a bird torso','a whole bird','a branch','a leaf','a patch of sky',
        'an unidentifiable region'],
    'derm7pt':['a skin lesion','a partial skin lesion','a patch of surrounding skin',
        'a hair crossing the skin','a ruler on skin','a shadow on skin',
        'a lighting reflection on skin','an unidentifiable region'],
}


def run(args):
    if not args.weights.is_file():
        raise FileNotFoundError('Provide an already downloaded OpenAI CLIP checkpoint.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable; use --device cpu.')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    weights_hash=sha256(args.weights)
    torch.set_num_threads(4)
    model,preprocess=clip.load(str(args.weights),device=args.device,jit=False)
    model.eval()
    banks={}
    for dataset in EXEMPLARS:
        for category,phrases in [('context',PARTS[dataset]),('shape',SHAPES)]:
            for view in ['neutral_crop','verified_contour']:
                template='a photo of {}.' if view=='neutral_crop' else 'the outline of {}.'
                tokens=clip.tokenize([template.format(p) for p in phrases]).to(args.device)
                with torch.inference_mode():
                    vectors=model.encode_text(tokens).float()
                    vectors=vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                banks[dataset,category,view]=(phrases,vectors,template)
    rows,inputs,panels=[],[],[]
    for dataset,(root,ids) in EXEMPLARS.items():
        for prototype in ids:
            folder=root/f'P{prototype:04d}'
            evidence_path=folder/'evidence.json'
            evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
            if evidence.get('geometry_status')!='verified_against_training_descriptor':
                raise ValueError(f'Unverified geometry: {folder}')
            if evidence['metadata'].get('source_split')!='train':
                raise ValueError('Naming uses training references exclusively.')
            inputs.append({'dataset':dataset,'prototype_id':prototype,
                'source_bank_root':str(root),'evidence_path':str(evidence_path),
                'evidence_sha256':sha256(evidence_path),'representative_key':evidence['representative_key'],
                'feature_columns':evidence['metadata']['feature_columns'],
                'raw_descriptor':evidence['metadata']['raw_descriptor']})
            for view in ['neutral_crop','verified_contour']:
                image_path=folder/(view+'.png')
                with Image.open(image_path) as source:
                    im=source.convert('RGB')
                tensor=preprocess(im).unsqueeze(0).to(args.device)
                with torch.inference_mode():
                    vector=model.encode_image(tensor).float()
                    vector=vector/vector.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                tops={}
                for category in ['context','shape']:
                    phrases,text,template=banks[dataset,category,view]
                    scores=(vector@text.T)[0].cpu().numpy()
                    order=np.argsort(-scores,kind='stable')
                    tops[category]=phrases[int(order[0])]
                    for rank,index in enumerate(order,start=1):
                        rows.append({'dataset':dataset,'prototype_id':prototype,
                            'view':view,'vocabulary':category,'rank':rank,
                            'candidate':phrases[int(index)],'cosine_similarity':float(scores[index]),
                            'top1_top2_gap':float(scores[order[0]]-scores[order[1]]),
                            'status':'unreviewed_exemplar_candidate','scope':'training_exemplar_only',
                            'prompt_template':template,'image_path':str(image_path),
                            'image_sha256':sha256(image_path)})
                panels.append((dataset,prototype,view,im.copy(),tops))
                print(f'[{dataset}/P{prototype}/{view}] {tops}',flush=True)
    write_csv(out/'candidate_rankings.csv',rows)
    write_json(out/'input_exemplars.json',inputs)
    write_json(out/'candidate_vocabulary.json',{'context':PARTS,'shape':SHAPES,
        'source':'Author-designed pilot candidates, not claimed as verbatim dataset annotations.',
        'diagnosis_or_class_names_included':False,
        'unknown_candidate_is_calibrated_abstention':False})
    for dataset in EXEMPLARS:
        (out/(dataset+'_context_vocab.txt')).write_text('\n'.join(PARTS[dataset])+'\n',encoding='utf-8')
    (out/'shape_vocab.txt').write_text('\n'.join(SHAPES)+'\n',encoding='utf-8')
    # A review sheet, not a scientific plot or endorsed set of labels.
    tile_w,tile_h=425,345
    sheet=Image.new('RGB',(3*tile_w,4*tile_h),'white')
    draw=ImageDraw.Draw(sheet)
    try:
        font=ImageFont.truetype('Arial.ttf',14)
    except OSError:
        try:
            font=ImageFont.truetype('DejaVuSans.ttf',14)
        except OSError:
            font=ImageFont.load_default()
    for i,(dataset,prototype,view,im,tops) in enumerate(panels):
        x=(i//2%3)*tile_w
        y=((i//6)*2+i%2)*tile_h
        thumb=ImageOps.contain(im,(400,220))
        sheet.paste(thumb,(x+(tile_w-thumb.width)//2,y+30+(220-thumb.height)//2))
        draw.text((x+10,y+5),f'{dataset} P{prototype}: {view}',font=font,fill='black')
        draw.text((x+10,y+260),'Context: '+tops['context'],font=font,fill='black')
        draw.text((x+10,y+283),'Shape: '+tops['shape'],font=font,fill='black')
        draw.text((x+10,y+308),'UNREVIEWED; training reference only',font=font,fill='#a02020')
    sheet.save(out/'candidate_review_sheet.png')
    unchanged=all(sha256(Path(r['evidence_path']))==r['evidence_sha256'] for r in inputs)
    unchanged=unchanged and all(sha256(Path(r['image_path']))==r['image_sha256'] for r in rows)
    assert unchanged and sha256(args.weights)==weights_hash
    write_json(out/'protocol.json',{'status':'complete','model':'OpenAI CLIP ViT-L/14',
        'weights':str(args.weights),'weights_sha256':weights_hash,'device':args.device,
        'n_exemplars':len(inputs),'n_image_views':len(panels),
        'selection':'Six existing qualitative figure references; a convenience sample, not naming validation.',
        'llm_called':False,'uploaded_images':False,'retrained_models':False,
        'existing_files_unchanged':unchanged,'classifier_predictions_affected':False,
        'score_type':'cosine similarity within candidate lists, not calibrated probability',
        'naming_scope':'Representative training exemplar, not a semantic guarantee for the prototype coordinate',
        'views_use_different_prompt_templates':True,'validated_abstention':False,
        'runner_sha256':sha256(Path(__file__))})
    print(json.dumps({'output':str(out),'status':'complete'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=Path.home()/'.cache/clip/ViT-L-14.pt')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
