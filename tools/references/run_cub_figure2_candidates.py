"""Find and preview real CUB examples for Figure 2's explanation readout.

Read-only with respect to models, datasets, manuscript and previous galleries.
Reuses the Figure 1 export validator and its trace assets. No training, mask
generation, coordinate substitution, or automatic prototype naming is involved.
Run with --source pointing to a completed run_cub_figure_candidates.py gallery.
"""
from __future__ import annotations

import argparse
import base64
from functools import lru_cache
import html
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.references.run_cub_figure_candidates import (
    SourceLedger, checked_decomposition, digest, fit_text, font, paste_fit,
    read_csv, read_json, write_csv, write_json,
)

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED, RULE = "#273640", "#65727B", "#D5DEE4"
PANEL_W, PANEL_H = 790, 820
REQUIRED_REFERENCE_FILES = (
    "crop.jpg", "mask.png", "neutral_crop.png", "verified_contour.png",
    "training_original.jpg", "training_context.png", "evidence.json",
    "region_provenance.json", "review.json",
)


@lru_cache(maxsize=None)
def contour_points(mask_path: str):
    """The current geo11 preprocessing, retaining the unsmoothed pixel boundary."""
    image = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {mask_path}")
    binary = (image > 127).astype(np.uint8) * 255
    binary = cv2.medianBlur(binary, 3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    outlines, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary)
    for outline in outlines:
        if cv2.contourArea(outline) > 100:
            cv2.drawContours(cleaned, [outline], 0, 255, -1)
    outlines, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not outlines:
        raise ValueError(f"Empty geo11 contour: {mask_path}")
    return max(outlines, key=cv2.contourArea).reshape(-1, 2).astype(float)


@lru_cache(maxsize=None)
def math_formula(latex, color):
    """Share the paper's STIX math typography between PNG and vector SVG."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    matplotlib.rcParams.update({"mathtext.fontset": "stix", "svg.fonttype": "path"})
    result = []
    for ext in ("png", "svg"):
        buffer = io.BytesIO()
        math_to_image(latex, buffer, prop=FontProperties(size=24), dpi=180,
                      format=ext, color=color)
        result.append(buffer.getvalue())
    return tuple(result)


class Panel:
    """Draw matching raster and editable SVG assets with shared coordinates."""

    def __init__(self):
        self.image = Image.new("RGB", (PANEL_W, PANEL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_W}" '
                    f'height="{PANEL_H}" viewBox="0 0 {PANEL_W} {PANEL_H}">',
                    '<rect width="100%" height="100%" fill="white"/>']

    def rect(self, box, fill="white", edge=RULE, width=1):
        x, y, w, h = box
        self.draw.rectangle((x, y, x+w, y+h), fill=fill, outline=edge, width=width)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                        f'fill="{fill}" stroke="{edge}" stroke-width="{width}"/>')

    def text(self, x, y, text, size=24, fill=INK, anchor="middle", bold=False, max_width=None):
        while max_width and self.draw.textlength(text, font=font(size, bold=bold)) > max_width:
            size -= 1
        if size < 15:
            raise ValueError(f"Text cannot fit panel: {text}")
        selected_font = font(size, bold=bold)
        anchor_pil = {"middle": "mm", "start": "lm", "end": "rm"}[anchor]
        bounds = self.draw.textbbox((x, y), text, font=selected_font, anchor=anchor_pil)
        if min(bounds[:2]) < 0 or bounds[2] > PANEL_W or bounds[3] > PANEL_H:
            raise ValueError(f"Text clipped: {text}: {bounds}")
        self.draw.text((x, y), text, font=selected_font, fill=fill, anchor=anchor_pil)
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'dominant-baseline="central" font-family="Times New Roman,serif" '
            f'font-size="{size}" font-weight="{"bold" if bold else "normal"}" '
            f'fill="{fill}">{html.escape(text)}</text>')

    def photo(self, path, box):
        x, y, w, h = box
        paste_fit(self.image, path, box, background="white")
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        self.svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
                        f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{data}"/>')

    def contour(self, path, box):
        x, y, w, h = box
        points = contour_points(str(path)).copy()
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high-low, 1)
        scale = min((w-20)/span[0], (h-16)/span[1])
        points = (points-low)*scale + [x+(w-span[0]*scale)/2, y+(h-span[1]*scale)/2]
        xy = [tuple(p) for p in points]
        self.draw.polygon(xy, fill="#EEF7FC")
        self.draw.line(xy+[xy[0]], fill=BLUE, width=3, joint="curve")
        values = " ".join(f"{xx:.3f},{yy:.3f}" for xx, yy in points)
        self.svg.append(f'<polygon points="{values}" fill="#EEF7FC" stroke="{BLUE}" '
                        'stroke-width="2.5" stroke-linejoin="round"/>')

    def math(self, x, y, latex, box=(650, 36), color=INK):
        png, svg = math_formula(latex, color)
        with Image.open(io.BytesIO(png)) as image:
            rgba = image.convert("RGBA")
        scale = min(box[0]/rgba.width, box[1]/rgba.height)
        width, height = round(rgba.width*scale), round(rgba.height*scale)
        rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
        self.image.paste(rgba, (round(x-width/2), round(y-height/2)), rgba)
        node = ET.fromstring(svg)
        node.set("x", str(x-width/2))
        node.set("y", str(y-height/2))
        node.set("width", str(width))
        node.set("height", str(height))
        self.svg.append(ET.tostring(node, encoding="unicode"))

    def save(self, path):
        self.image.save(path.with_suffix(".png"))
        path.with_suffix(".svg").write_text("\n".join(self.svg+["</svg>"]), encoding="utf-8")


def panel(case, output):
    p = Panel()
    p.rect((1, 1, PANEL_W-3, PANEL_H-3), edge=BLUE, width=2)
    p.text(395, 35, "Explanation: class f versus b", size=29, bold=True)
    p.text(395, 76, "f: "+case["sdrcbm_name"].replace("_", " "), size=24, max_width=740)
    p.text(395, 110, "b: "+case["dcbm_name"].replace("_", " "), size=24, max_width=740)
    p.math(395, 151, r"$q_k=\eta_s(W_{f,k}-W_{b,k})\hat a_{s,k}$", color=BLUE)
    p.text(395, 190, "Training prototype references", size=25, fill=MUTED)
    for i, row in enumerate(case["top"]):
        center = 138 + 257*i
        p.rect((center-109, 217, 218, 107))
        p.photo(row["reference"]/"neutral_crop.png", (center-105, 221, 210, 99))
        p.rect((center-109, 335, 218, 93))
        p.contour(row["reference"]/"mask.png", (center-109, 335, 218, 93))
        p.text(center, 450, "P"+row["prototype_id"], size=27, fill=BLUE)
        # 468 <= y <= 495 is intentionally left free for future reviewed names.
    p.text(395, 513, "Top-3 signed margin contributions", size=25)
    zero, span = 360, 246
    p.draw.line((zero, 540, zero, 650), fill=RULE, width=2)
    p.svg.append(f'<path d="M{zero},540 V650" stroke="{RULE}" stroke-width="2"/>')
    maxq = max(abs(float(r["signed_margin_contribution"])) for r in case["top"])
    for i, row in enumerate(case["top"]):
        y = 558 + 38*i
        q = float(row["signed_margin_contribution"])
        left = zero + min(0, q/maxq*span)
        color = BLUE if q >= 0 else ORANGE
        p.text(38, y, "P"+row["prototype_id"], size=24, anchor="start")
        p.rect((left, y-9, abs(q)/maxq*span, 18), fill=color, edge=color)
        p.text(751, y, f"{q:+.3f}", size=26, anchor="end", fill=color)
    p.math(395, 681, r"$m_{\mathrm{final}}=m_{\mathrm{sem}}+\sum_k q_k+q_{\mathrm{bias}}$")
    p.text(395, 721, f"Semantic {case['semantic_margin']:+.3f}     Other {case['other']:+.3f}",
           size=24, max_width=740)
    p.text(395, 758, f"Bias {case['bias']:+.3f}     Final {case['final_margin']:+.3f}",
           size=24, max_width=740)
    p.text(395, 800, "Training references; no test-image localization.", size=18, fill=MUTED)
    p.save(output/"panels"/case["candidate_id"])
    return p.image


def card(case, measured_panel, output):
    image = Image.new("RGB", (1755, 1050), "white")
    draw = ImageDraw.Draw(image)
    fit_text(draw, f"{case['candidate_id']} | {case['category']} | {case['true_name']}",
             (20, 15), 1705, 31)
    fit_text(draw, "Figure 2 readout candidate: exact Top-3, names left open", (20, 57), 1700, 22, MUTED)
    for x, width, title in ((20, 391, "Input and predictions"), (432, 483, "Training reference context / region / contour")):
        draw.rectangle((x, 99, x+width, 921), outline=RULE, width=2)
        fit_text(draw, title, (x+10, 111), width-20, 22)
    paste_fit(image, case["path"]/"input.jpg", (31, 160, 369, 325))
    fit_text(draw, "GT: "+case["true_name"], (36, 518), 355, 27)
    fit_text(draw, "DCBM: "+case["dcbm_name"], (36, 612), 355, 26)
    fit_text(draw, "SDR-CBM: "+case["sdrcbm_name"], (36, 706), 355, 26,
             GREEN if case["category"] == "correction" else ORANGE)
    fit_text(draw, case["run_label"]+"; test index "+str(case["index"]), (36, 839), 355, 22, MUTED)
    for i, row in enumerate(case["top"]):
        y = 161 + i*247
        ref = row["reference"]
        fit_text(draw, "P"+row["prototype_id"]+f"  q={float(row['signed_margin_contribution']):+.3f}",
                 (451, y-2), 445, 25, BLUE if float(row["signed_margin_contribution"]) >= 0 else ORANGE)
        paste_fit(image, ref/"training_context.png", (451, y+39, 204, 171))
        paste_fit(image, ref/"neutral_crop.png", (665, y+54, 112, 136))
        paste_fit(image, ref/"verified_contour.png", (786, y+54, 112, 136))
    image.paste(measured_panel, (945, 100))
    fit_text(draw, "Exact checkpoint example. Other contains the remaining 125 coordinates; values shown are rounded.",
             (20, 945), 1705, 23, MUTED)
    fit_text(draw, "Part names will be added only after a separate naming review. Positive contribution does not imply high similarity.",
             (20, 986), 1705, 22, MUTED)
    image.save(output/"cards"/(case["candidate_id"]+".png"))


def order_key(case):
    return (case["category"] != "correction", not case["mixed_signs"],
            -case["top3_bird_references"], -case["top3_usable_parts"],
            -case["top3_coverage"], case["candidate_id"])


def recommend(cases, limit):
    ordered = sorted((c for c in cases if c["category"] == "correction"), key=order_key)
    selected, seen_images, seen_refs = [], set(), set()

    def add(case):
        selected.append(case)
        seen_images.add(case["sample_key"])
        seen_refs.update(r["reference_id"] for r in case["top"] if r["usable_part"])

    # The strongest all-bird mixed-sign case and the already chosen Figure 1
    # example are both useful starting points, without mixing their model runs.
    for cid in ("R42-1425", "R44-0877"):
        match = next((c for c in ordered if c["candidate_id"] == cid), None)
        if match and len(selected) < limit:
            add(match)
    for case in ordered:
        new_parts = {r["reference_id"] for r in case["top"] if r["usable_part"]} - seen_refs
        if case["sample_key"] not in seen_images and new_parts and len(selected) < limit:
            add(case)
    for case in ordered:
        if case["sample_key"] not in seen_images and len(selected) < limit:
            add(case)
    return selected


def load_cases(source, output, ledger):
    references = {}
    cases, audit = [], []
    for record in read_csv(ledger.record(source/"candidate_ranking.csv")):
        cid = record["candidate_id"]
        folder = source/"cases"/cid
        candidate = read_json(ledger.record(folder/"candidate.json"))
        summary = read_json(ledger.record(folder/"case_summary.json"))
        coordinates = read_csv(ledger.record(folder/"shape_coordinates.csv"))
        classes = read_csv(ledger.record(folder/"class_logits.csv"))
        if len(coordinates) != 128 or len(classes) != 200:
            raise ValueError(f"Unexpected coordinate/class count: {cid}")
        top, other, closure = checked_decomposition(summary, coordinates, classes)
        if summary.get("context_margin_f_minus_b", 0):
            raise ValueError("This layout is for a shape-only residual; unexpected context branch.")
        ref_lookup = {int(ref["prototype_id"]): ref for ref in candidate["references"]}
        for row in top:
            meta = ref_lookup[int(row["prototype_id"])]
            identity = meta["reference_id"]
            source_ref = source/"references"/identity
            if identity not in references:
                review = read_json(ledger.record(source_ref/"review.json"))
                evidence = read_json(ledger.record(source_ref/"evidence.json"))
                if evidence["geometry_status"] != "verified_against_training_descriptor":
                    raise ValueError(f"Unverified geometry: {identity}")
                if evidence["metadata"]["source_split"] != "train":
                    raise ValueError(f"Nontraining reference: {identity}")
                references[identity] = {"review": review, "evidence": evidence, "source": source_ref}
            ref = references[identity]
            row.update(reference_id=identity, source_reference=source_ref,
                       usable_part=bool(ref["review"].get("usable_part", False)),
                       bird_reference=ref["review"].get("category") in ("bird_part", "whole_bird"))
        q = [float(r["signed_margin_contribution"]) for r in top]
        parts = sum(r["usable_part"] for r in top)
        birds = sum(r["bird_reference"] for r in top)
        qualifies = parts >= 1 or birds >= 2
        mixed = min(q) < 0 < max(q)
        full_abs = math.fsum(abs(float(r["signed_margin_contribution"])) for r in coordinates)
        case = {"candidate_id": cid, "run_label": candidate["run_label"], "index": candidate["index"],
                "sample_key": candidate["sample_key"], "true_name": summary["true_name"],
                "dcbm_name": summary["dcbm_name"], "sdrcbm_name": summary["sdrcbm_name"],
                "category": summary["category"], "top3_usable_parts": parts, "top3_bird_references": birds,
                "mixed_signs": mixed, "qualifies": qualifies,
                "top3_coverage": math.fsum(abs(x) for x in q)/full_abs if full_abs else 0,
                "semantic_margin": float(summary["semantic_margin_f_minus_b"]),
                "bias": float(summary["shape_bias_margin_f_minus_b"]),
                "final_margin": float(summary["final_margin_f_minus_b"]), "other": other,
                "closure_error": closure, "top3_ids": ";".join("P"+r["prototype_id"] for r in top),
                "top3_contributions": ";".join(str(x) for x in q),
                "path": folder, "top": top}
        audit.append({k: v for k, v in case.items() if k not in ("top", "path")})
        if not qualifies:
            continue
        target = output/"cases"/cid
        target.mkdir(parents=True)
        for name in ("candidate.json", "case_summary.json", "input.jpg", "shape_coordinates.csv",
                     "shape_prototypes.csv", "class_logits.csv", "waterfall.csv"):
            shutil.copy2(ledger.record(folder/name), target/name)
        case["path"] = target
        for row in top:
            dest = output/"references"/row["reference_id"]
            row["reference"] = dest
            if dest.exists():
                continue
            dest.mkdir(parents=True)
            for name in REQUIRED_REFERENCE_FILES:
                source_file = ledger.record(row["source_reference"]/name)
                shutil.copy2(source_file, dest/name)
            evidence = read_json(dest/"evidence.json")
            provenance = read_json(dest/"region_provenance.json")
            for name, key in (("crop.jpg", "crop_sha256"), ("mask.png", "mask_sha256")):
                assert digest(dest/name) == evidence["metadata"][key] == provenance["region"][key]
            assert digest(dest/"training_original.jpg") == provenance["source_sha256"]
            contour_points(str(dest/"mask.png"))
        cases.append(case)
    return sorted(cases, key=order_key), audit


def gallery(cases, recommended, counts, output):
    recommended_ids = {c["candidate_id"] for c in recommended}
    tiles = []
    for case in sorted(cases, key=lambda c: (c["candidate_id"] not in recommended_ids, order_key(c))):
        cid = case["candidate_id"]
        sign = "正负兼具" if case["mixed_signs"] else "所示三项同号"
        refs = " · ".join("P"+r["prototype_id"] for r in case["top"])
        text = " · ".join([cid, case["true_name"], case["dcbm_name"], case["sdrcbm_name"], refs])
        tile = f'''<article data-category="{case['category']}" data-mixed="{str(case['mixed_signs']).lower()}"
          data-short="{str(cid in recommended_ids).lower()}" data-run="{case['run_label']}"
          data-parts="{case['top3_usable_parts']}" data-birds="{case['top3_bird_references']}"
          data-search="{html.escape(text.lower(), quote=True)}">
          <h2>{cid} <small>{'优先查看' if cid in recommended_ids else ''}</small></h2>
          <p>{html.escape(case['dcbm_name'].replace('_',' '))} → {html.escape(case['sdrcbm_name'].replace('_',' '))}
          <strong class="{case['category']}">{'纠错' if case['category']=='correction' else '误改'}</strong></p>
          <p>{sign}；前三项清晰部件 {case['top3_usable_parts']}；鸟体参考 {case['top3_bird_references']}/3。
          {'与已选图1是同一案例。' if cid=='R44-0877' else ''}</p>
          <div class="pair"><a href="cases/{cid}/input.jpg"><img class="input" loading="lazy" src="cases/{cid}/input.jpg"></a>
          <a href="panels/{cid}.png"><img class="panel" loading="lazy" src="panels/{cid}.png"></a></div>
          <p><a href="cards/{cid}.png">完整核对卡</a> · <a href="panels/{cid}.svg">可编辑面板SVG</a> ·
          <a href="cases/{cid}/shape_coordinates.csv">全部128维</a> · <a href="cases/{cid}/class_logits.csv">类别分数</a>
          {' · <a href="in_context/'+cid+'.png">放入图2的效果</a>' if cid in recommended_ids else ''}</p></article>'''
        tiles.append(tile)
    source = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CUB 图2右上角候选</title>
    <style>*{box-sizing:border-box}body{margin:24px;background:#f3f6f8;color:#273640;font:16px 'Microsoft YaHei',sans-serif}
    main{max-width:1500px;margin:auto}header{background:#fff;padding:18px;border:1px solid #d5dee4;border-radius:8px}
    h1{font-size:27px;margin:0}h2{font-size:22px}p{line-height:1.7}a{color:#0072B2}small,.correction{color:#009E73}.harm{color:#ae6100}
    select,input,button{font:inherit;padding:6px;margin:5px}#grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
    article{background:white;padding:16px;border:1px solid #d5dee4;border-radius:6px}article[hidden]{display:none}
    .pair{display:grid;grid-template-columns:40% 58%;align-items:center;gap:2%;min-height:300px}.pair img{width:100%;height:auto}
    .input{max-height:300px;object-fit:contain}.panel{max-height:580px;object-fit:contain}#count{font-weight:bold}
    #filters{position:sticky;top:0;background:#fff;border-bottom:1px solid #d5dee4;padding:9px;z-index:2;margin:16px 0}
    @media(max-width:950px){#grid{grid-template-columns:1fr}body{margin:10px}#filters{position:static}}</style>
    <main><header><h1>CUB：图2右上角真实解释候选</h1>
    <p>审核全部 AUDITED 个已有运行—样例组合，筛出 TOTAL 个可辨鸟体/部件候选（纠错 CORRECT 个、误改 HARM 个）。
    初筛：真实 Top-3 至少有一个清晰部件，或至少有两个鸟体参考；不替换高贡献坐标。</p>
    <p>优先看纠错、正负贡献兼具的案例。只保留 P 编号，并预留命名空间。局部参考来自训练图像，不是在测试图上定位到的部件。
    各案例保留自身运行与完整贡献，不能把不同案例拼成同一次解释。</p>
    <p><a href="shortlist_overview.png">推荐总览</a> · <a href="candidate_ranking.csv">本次候选清单</a> ·
    <a href="all_cases_audit.csv">全部审核记录</a> · <a href="README_zh.md">范围与筛选说明</a></p></header>
    <div id="filters"><label><input id="short" type="checkbox" checked>只看优先候选</label>
    <select id="category"><option value="all">纠错与误改</option><option value="correction">仅纠错</option><option value="harm">仅误改</option></select>
    <label><input id="mixed" type="checkbox">正负兼具</label><label><input id="parts" type="checkbox">至少一个清晰部件</label>
    <label><input id="birds" type="checkbox">三项均为鸟体参考</label>
    <select id="run"><option value="all">全部运行</option><option>R42</option><option>R43</option><option>R44</option><option>T</option></select>
    <input id="search" placeholder="编号、物种或P编号"><span id="count"></span></div><div id="grid">ARTICLES</div></main>
    <script>const articles=[...document.querySelectorAll('article')];const get=id=>document.getElementById(id);
    function filter(){let n=0;for(const a of articles){const visible=(!get('short').checked||a.dataset.short==='true')&&
    (get('category').value==='all'||a.dataset.category===get('category').value)&&(!get('mixed').checked||a.dataset.mixed==='true')&&
    (!get('parts').checked||+a.dataset.parts>0)&&(!get('birds').checked||+a.dataset.birds===3)&&
    (get('run').value==='all'||a.dataset.run===get('run').value)&&a.dataset.search.includes(get('search').value.trim().toLowerCase());
    a.hidden=!visible;if(visible)n++;}get('count').textContent=`显示 ${n} / ${articles.length} 个`;}
    document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',filter));filter();</script></html>'''
    source = (source.replace("AUDITED", str(counts["audited"])).replace("TOTAL", str(len(cases)))
              .replace("CORRECT", str(counts["qualifying_corrections"]))
              .replace("HARM", str(counts["qualifying_harms"])).replace("ARTICLES", "\n".join(tiles)))
    (output/"index.html").write_text(source, encoding="utf-8")


def overview(recommended, output):
    cols, tw, th = 4, 500, 675
    height = 75+math.ceil(len(recommended)/cols)*th
    image = Image.new("RGB", (cols*tw, height), "white")
    draw = ImageDraw.Draw(image)
    fit_text(draw, "Figure 2: candidate shortlist (exact Top-3; prototype names not assigned)", (18, 14), cols*tw-36, 30)
    for i, case in enumerate(recommended):
        x, y = i%cols*tw, 75+i//cols*th
        draw.rectangle((x+5, y+3, x+tw-5, y+th-6), outline=RULE, width=2)
        fit_text(draw, case["candidate_id"]+" | "+case["true_name"], (x+15, y+12), tw-30, 24)
        paste_fit(image, case["path"]/"input.jpg", (x+100, y+65, 300, 140))
        paste_fit(image, output/"panels"/(case["candidate_id"]+".png"), (x+35, y+218, 430, 445))
    image.save(output/"shortlist_overview.png")


def context_previews(recommended, blueprint, output):
    """Update BOTH the inference image and its readout in review-only previews."""
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location("figure2_original_layout", blueprint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for case in recommended:
        module.INPUT = case["path"]/"input.jpg"
        diagram = module.figure_two()
        diagram.texts[-1].set_text(
            "Measured right-hand panel; references do not localize test-image evidence. Training icons are schematic.")
        diagram.check_text_bounds()
        buffer = io.BytesIO()
        diagram.fig.savefig(buffer, format="png", dpi=180)
        plt.close(diagram.fig)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        x0, y0 = round(11.24*180), round(.86*180)
        x1, y1 = round(15.19*180), round(4.96*180)
        with Image.open(output/"panels"/(case["candidate_id"]+".png")) as part:
            image.paste(part.resize((x1-x0, y1-y0), Image.Resampling.LANCZOS), (x0, y0))
        image.save(output/"in_context"/(case["candidate_id"]+".png"))


def render_existing(output, blueprint):
    """Refresh only derived previews after a layout edit, retaining all sources."""
    manifest = read_json(output/"manifest.json")
    cases = []
    for item in read_csv(output/"candidate_ranking.csv"):
        case = dict(item)
        for key in ("semantic_margin", "other", "bias", "final_margin"):
            case[key] = float(case[key])
        case["path"] = output/"cases"/case["candidate_id"]
        candidate = read_json(case["path"]/"candidate.json")
        summary = read_json(case["path"]/"case_summary.json")
        top, _, _ = checked_decomposition(summary, read_csv(case["path"]/"shape_coordinates.csv"),
                                           read_csv(case["path"]/"class_logits.csv"))
        identities = {str(ref["prototype_id"]): ref["reference_id"] for ref in candidate["references"]}
        for row in top:
            row["reference"] = output/"references"/identities[row["prototype_id"]]
        case["top"] = top
        cases.append(case)
    for i, case in enumerate(cases):
        card(case, panel(case, output), output)
        if (i+1) % 20 == 0 or i+1 == len(cases):
            print(f"Refreshed {i+1}/{len(cases)} previews", flush=True)
    lookup = {c["candidate_id"]: c for c in cases}
    recommended = [lookup[cid] for cid in manifest["recommended"]]
    overview(recommended, output)
    context_previews(recommended, blueprint, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, required=True,
                        help="Existing build_schematics.py, used only for review-only context previews.")
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--render-existing", action="store_true",
                        help="Refresh derived previews in a prior output after a layout edit.")
    args = parser.parse_args()
    if args.render_existing:
        render_existing(args.output, args.blueprint)
        return
    if args.output.exists():
        raise ValueError("Choose a new output directory; existing results will not be overwritten.")
    if args.shortlist < 1:
        raise ValueError("--shortlist must be positive")
    for name in ("cases", "references", "panels", "cards", "in_context"):
        (args.output/name).mkdir(parents=True, exist_ok=True)
    ledger = SourceLedger()
    ledger.record(args.blueprint)
    for name in ("fig2_explanation_preview.png", "fig2_explanation_preview.svg"):
        ledger.record(args.blueprint.parent/name)
    cases, audit = load_cases(args.source, args.output, ledger)
    recommended = recommend(cases, args.shortlist)
    recommendation_ids = {c["candidate_id"] for c in recommended}
    serial = [{**{k: v for k, v in c.items() if k not in ("path", "top")},
               "recommended": c["candidate_id"] in recommendation_ids} for c in cases]
    write_csv(args.output/"candidate_ranking.csv", serial)
    write_csv(args.output/"all_cases_audit.csv", audit)
    for i, case in enumerate(cases):
        rendered = panel(case, args.output)
        card(case, rendered, args.output)
        write_json(case["path"]/"figure2_readout.json", {
            "candidate_id": case["candidate_id"], "source_run": case["run_label"],
            "scope": "saved local single-checkpoint test example, not server averaged accuracy",
            "margin_definition": "fused predicted class f versus semantic predicted class b",
            "prototype_names_assigned": False,
            "name_slots_panel_pixels": [[29+257*j, 468, 218, 27] for j in range(3)],
            "top3": [{"prototype_id": int(r["prototype_id"]), "reference_id": r["reference_id"],
                      "activation_head_input": float(r["activation"]),
                      "weight_difference_with_eta": float(r["weight_difference_f_minus_b"]),
                      "contribution": float(r["signed_margin_contribution"])} for r in case["top"]],
            "other_125": case["other"], "bias": case["bias"],
            "semantic_margin": case["semantic_margin"], "final_margin": case["final_margin"],
            "closure_error": case["closure_error"], "all_128_coordinates_saved": True,
            "no_test_mask_for_inference": True,
        })
        if (i+1) % 20 == 0 or i+1 == len(cases):
            print(f"Rendered {i+1}/{len(cases)} candidates", flush=True)
    counts = {
        "audited": len(audit), "audited_unique_test_images": len({r["sample_key"] for r in audit}),
        "qualifying": len(cases), "qualifying_unique_test_images": len({c["sample_key"] for c in cases}),
        "qualifying_corrections": sum(c["category"] == "correction" for c in cases),
        "qualifying_harms": sum(c["category"] == "harm" for c in cases),
        "qualifying_mixed_corrections": sum(c["category"] == "correction" and c["mixed_signs"] for c in cases),
        "part_corrections": sum(c["category"] == "correction" and c["top3_usable_parts"] > 0 for c in cases),
        "recommended": [c["candidate_id"] for c in recommended],
        "maximum_closure_error": max(r["closure_error"] for r in audit),
    }
    gallery(cases, recommended, counts, args.output)
    overview(recommended, args.output)
    context_previews(recommended, args.blueprint, args.output)
    shutil.copy2(ledger.record(args.source/"runs.json"), args.output/"runs.json")
    write_json(args.output/"manifest.json", {**counts,
        "source_gallery": str(args.source.resolve()),
        "scope": "all changed-prediction examples in the four existing T/R42/R43/R44 trace exports; not all CUB images",
        "selection": "actual Top-3 has at least one reviewed clear part or at least two bird references",
        "preferred": "correction, mixed signs, more bird references, distinct test images and reference parts",
        "panel_pixels": [PANEL_W, PANEL_H], "target_layout_units": [3.95, 4.10],
        "names_added": False, "training_or_segmentation_run": False,
        "unchanged_sources": ledger.verify(),
    })
    readme = f"""# 图2右上角 CUB 候选

打开 index.html；默认显示 {len(recommended)} 个优先候选。取消“只看优先候选”可看全体。

本次完整审核已有 T、R42、R43、R44 四组解释导出的 {len(audit)} 个运行—样例组合，
对应 {counts['audited_unique_test_images']} 张不同测试图像。不是对 CUB 全部图像进行穷举。
这些导出覆盖各组已有的全部纠错/误改案例。其余预测未变的图像在本布局定义 f=b 时 margin 为零，
不适合用来展示 f-versus-b 的非零贡献。

初筛标准：保持同次预测的真实绝对贡献 Top-3，其中至少有一个可辨部件，或至少有两个
已审阅为鸟体/部件的训练参考。得到 {len(cases)} 个备选、{counts['qualifying_unique_test_images']} 张图，
其中纠错 {counts['qualifying_corrections']} 个（正负兼具 {counts['qualifying_mixed_corrections']} 个）、
误改 {counts['qualifying_harms']} 个。误改案例保留供审查，不默认推荐为成功纠错展示。
Top-3 至少有清晰部件的纠错案例为 {counts['part_corrections']} 个。

优先 R42-1425：三项训练参考均为鸟体，一项头部/喙、一项全鸟、一项前部身体，贡献两正一负。
R44-0877 是已选图1案例，若希望两图展示同一案例可选它；其前三项均为正贡献。
其他推荐优先覆盖不同图像和不同部件来源。名称仅用于既有人工审阅筛选，未写入面板；
所有面板只保留 P 编号并留出后续命名位置。没有使用CLIP或LLM新增名字。

panels/：符合原预留区域 3.95:4.10 比例的实测解释面板，PNG + 可编辑、内嵌素材的SVG。
cards/：输入、预测、训练原图位置、局部图、轮廓、面板的完整核对图。
in_context/：优先候选放入原图2预案的效果；左侧输入图也同步更换为该案例，避免解释张冠李戴。
cases/：原始输入、原始全128维坐标与完整类别分数、来源、贡献映射JSON。
references/：按参考身份ID分开保存训练原图、局部图、mask、轮廓和来源。
candidate_ranking.csv：本次全部备选；all_cases_audit.csv：包括未通过初筛的全部审核记录。
manifest.json：来源哈希、筛选口径、数量和闭合检查；runs.json：原始运行来源。

不同运行同一个P编号可能对应不同的参考，必须使用对应 reference_id 及完整同次数据。
面板贡献公式中的响应是分类头使用的标准化输入；导出权重差已含 eta，不再二次相乘。
Other包含其余125项，Bias单列；所有margin在舍入前闭合。正贡献不等于高相似度。
训练参考不是测试图像上的空间定位；一个参考图可辨部件不代表整个形状簇的解剖命名。
这些局部检查点的个案也不能代替服务器末15轮均值或总体性能结果。

仅新建筛选与预览输出；未训练、未分割，未修改论文、数据集、模型、原图1或原图2预案。
"""
    (args.output/"README_zh.md").write_text(readme, encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
