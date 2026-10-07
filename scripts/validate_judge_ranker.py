#!/usr/bin/env python3
"""Offline validation: can the internal Qwen2.5-VL judge RANK edits better than
the embedding scalar? (No training GPU spent on selection — pure evaluation.)

We already have two full ImgEdit candidate sets (r1, r2) for all 737 keys, each
GPT-scored. r1_mean == r2_mean, so ALL of the +0.183 oracle headroom is pure
SELECTION. This script runs the REAL internal VLM judge (InternalRubricCEPREvaluator,
the editor's own Qwen2.5-VL text encoder — no external model) over each (source,
r1, r2) triple and asks: if we select by the judge, how much of the oracle
headroom do we recover, and how does that compare to selecting by the weak
embedding scalar? Broad recovery => the judge is the fix for broad gains and is a
legitimate self-contained novel contribution.

Rankers compared per key (argmax over {r1, r2}):
  - embedding  : cepr_pre_vlm_raw_reward   (the weak scalar, expected ~random)
  - judge      : internal_vlm_judge_score  (structured Qwen2.5-VL verdict)
  - combined   : internal_vlm_judge_combined_raw_reward (0.45 emb + 0.55 judge blend)
Baselines: base=r1, random=mean(r1,r2), oracle=max(r1,r2).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

R1_DIR = ROOT / "outputs/benchmark_images/imgedit/imgedit_route_object_v3_r1_28_be"
R2_DIR = ROOT / "outputs/benchmark_images/imgedit/imgedit_route_object_v3_r2_28_ber"
R1_SCORES = ROOT / "outputs/scores/imgedit/imgedit_route_object_v3_r1_28_be_average_score.json"
R2_SCORES = ROOT / "outputs/scores/imgedit/imgedit_route_object_v3_r2_28_ber_average_score.json"
BASIC = ROOT / "data/processed/benchmark/imgedit/basic_edit.json"
SRC_BASE = ROOT / "data/downloads/benchmarks/imgedit/extracted/Benchmark/singleturn"

# ImgEdit family -> canonical rubric edit_type (only affects structured_edit hints;
# the judge is instruction-driven, so unmapped types fall back to generic).
FAMILY_MAP = {
    "remove": "object_removal",
    "add": "object_addition",
    "replace": "object_replacement",
    "extract": "object_extraction",
    "background": "background_change",
    "adjust": "attribute_change",
    "style": "style_change",
    "action": "action_change",
    "compose": "compose",
}


def build_proposal(key: str, edit_type: str, instruction: str):
    from qwen_edit_project.self_evolve.types import EditProposal, ProposalDefinition

    definition = ProposalDefinition(
        operation_id=f"judgeval_{key}",
        instruction=instruction,
        family=edit_type,
        difficulty=2,
        scope="local",
        metric="internal_prompt_gain",
        direction="increase",
        target=0.0,
        expected_changed_fraction=(0.03, 0.80),
        verifier="internal_cepr_plus",
    )
    return EditProposal(
        record_key=key,
        round_index=0,
        proposal_index=0,
        definition=definition,
        difficulty_level=2,
        instruction=instruction,
        structured_edit={"edit_type": edit_type, "instruction": instruction},
    )


def load_feature_pipe(model_id: str, device: str):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    # Skip the ~20B MMDiT transformer: judging only needs Qwen2.5-VL + processor.
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, transformer=None, torch_dtype=torch.bfloat16)
    if hasattr(pipe, "to"):
        pipe.to(device)
    if getattr(pipe, "torch_dtype", None) is None:
        try:
            pipe.torch_dtype = torch.bfloat16
        except Exception:
            pass
    if getattr(pipe, "processor", None) is None:
        from transformers import AutoProcessor

        pipe.processor = AutoProcessor.from_pretrained("Qwen/Qwen-Image-Edit")
    return pipe


def make_evaluator_config(config_path: str | None) -> dict[str, Any]:
    """Load a production evaluator block (judge enabled) and force score-every-candidate."""
    cfg: dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        import yaml

        loaded = yaml.safe_load(Path(config_path).read_text()) or {}
        cfg = dict(loaded.get("evaluator", {}))
        cfg.pop("backend", None)
    else:
        print(f"WARN: config '{config_path}' not found; using inline minimal judge config.", flush=True)
    # Force judge ON, score every candidate, never gate away the score.
    judge = dict(cfg.get("internal_vlm_judge", {})) if isinstance(cfg.get("internal_vlm_judge"), dict) else {}
    judge.update(
        {
            "enabled": True,
            "skip_infeasible": False,
            "require_for_feasible": False,
            "use_unreliable_scores": True,
        }
    )
    cfg["internal_vlm_judge"] = judge
    # Disable the heavy grounder / region gates: we read the judge score directly,
    # and this avoids loading extra detector models.
    cfg["object_detector_enabled"] = False
    cfg["conservative_region_reward_enabled"] = False
    return cfg


def src_path_for(rec: dict[str, Any]) -> Path:
    return SRC_BASE / rec["id"]


def recovered(picked: list[float], base: list[float], oracle: list[float]) -> dict[str, float]:
    pm, bm, om = statistics.mean(picked), statistics.mean(base), statistics.mean(oracle)
    denom = om - bm
    return {
        "picked_mean": round(pm, 4),
        "gain_over_base": round(pm - bm, 4),
        "recovered_frac": round((pm - bm) / denom, 3) if denom > 1e-9 else 0.0,
    }


_PAIRWISE_RE = None


def pairwise_choice(pipe, instruction: str, original: Image.Image, img_a: Image.Image, img_b: Image.Image) -> str | None:
    """Forced-choice: which candidate (A or B) better accomplishes the instruction?
    Uses the editor's own Qwen2.5-VL (self-contained). Returns 'A', 'B', or None."""
    import re
    import torch

    global _PAIRWISE_RE
    if _PAIRWISE_RE is None:
        _PAIRWISE_RE = re.compile(r'"?better"?\s*[:=]?\s*"?\s*([AB])', re.IGNORECASE)
    model, processor = pipe.text_encoder, pipe.processor
    prompt = (
        f'Two candidates (A and B) are edits of the original image for the instruction:\n"{instruction}"\n'
        "Pick the ONE candidate that better follows the instruction with correct, faithful content and "
        "good preservation of everything that should stay unchanged. You MUST choose; if nearly equal, pick "
        'the slightly better one. Return compact JSON only: {"better": "A" or "B"}.'
    )
    content = [
        {"type": "text", "text": "Original image:"}, {"type": "image", "image": "original"},
        {"type": "text", "text": "Candidate A:"}, {"type": "image", "image": "a"},
        {"type": "text", "text": "Candidate B:"}, {"type": "image", "image": "b"},
        {"type": "text", "text": prompt},
    ]
    messages = [
        {"role": "system", "content": "You are a strict image-editing evaluator. Return compact JSON only."},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = processor(text=[text], images=[original, img_a, img_b], padding=True, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
    m = _PAIRWISE_RE.search(decoded)
    if m:
        return m.group(1).upper()
    for ch in decoded:
        if ch in "AB":
            return ch
    return None


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 1e-9 and dy > 1e-9 else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    return _pearson(rank(xs), rank(ys))


_VERIFY_KEYS = ("realized", "correct", "preserved", "clean")


def verification_reward(pipe, instruction: str, original: Image.Image, candidate: Image.Image) -> dict[str, float]:
    """STRICT self-contained DECOMPOSED verification: ask the editor's OWN Qwen2.5-VL
    specific checkable conditions (realized/correct/preserved/clean) instead of a generic
    quality score. Plays to the VLM's discrimination strength. Returns the 4 component
    probabilities in [0,1] plus 'combined' (geometric mean)."""
    import re
    import torch

    model, processor = pipe.text_encoder, pipe.processor
    prompt = (
        f'Instruction that was requested: "{instruction}"\n'
        "Compare the ORIGINAL and EDITED images and verify the edit. For EACH item output a "
        "probability from 0.00 to 1.00 (1.00 = definitely true, 0.00 = definitely false):\n"
        "- realized: the requested change is clearly present in the edited image.\n"
        "- correct: the changed content matches exactly what was asked (right object/attribute/location).\n"
        "- preserved: everything the instruction did NOT ask to change is unchanged from the original.\n"
        "- clean: the edited image is free of artifacts, distortions, or implausible regions.\n"
        'Return compact JSON only: {"realized":x,"correct":x,"preserved":x,"clean":x}'
    )
    content = [
        {"type": "text", "text": "Original image:"}, {"type": "image", "image": "original"},
        {"type": "text", "text": "Edited image:"}, {"type": "image", "image": "edited"},
        {"type": "text", "text": prompt},
    ]
    messages = [
        {"role": "system", "content": "You are a strict image-editing verifier. Return compact JSON only."},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = processor(text=[text], images=[original, candidate], padding=True, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
    vals: dict[str, float] = {}
    for k in _VERIFY_KEYS:
        m = re.search(rf'"{k}"\s*:\s*([01](?:\.\d+)?)', decoded)
        vals[k] = max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    prod = 1.0
    for k in _VERIFY_KEYS:
        prod *= vals[k]
    vals["combined"] = prod ** 0.25
    return vals


_VERIFY_QUESTIONS = {
    "realized": "Was the change requested by the instruction actually applied in the edited image (clearly present, not missing)?",
    "correct": "Does the applied change match exactly what the instruction asked for (correct object, attribute, and location)?",
    "preserved": "Is everything that the instruction did NOT ask to change kept identical to the original image?",
    "clean": "Is the edited image free of visible artifacts, distortions, blur, or implausible regions?",
}
_YES_WORDS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
_NO_WORDS = ("No", " No", "no", " no", "NO", " NO")


def yesno_token_ids(processor):
    """First-token ids for Yes/No surface forms, used to read verification log-probs."""
    tok = processor.tokenizer

    def first_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    return first_ids(_YES_WORDS), first_ids(_NO_WORDS)


def verification_reward_logprob(pipe, instruction: str, original: Image.Image, candidate: Image.Image,
                                yes_ids, no_ids) -> dict[str, float]:
    """De-saturated decomposed verification. Verbalized probabilities collapse to 0.95/1.00
    (93% of candidate pairs tie), so instead of asking the model to SAY a number we read the
    Yes/No token LOG-PROBABILITY for each checkable condition. P(yes) is continuous and
    recovers the resolution that verbalization throws away."""
    import torch

    model, processor = pipe.text_encoder, pipe.processor
    device = getattr(model, "device", None) or next(model.parameters()).device
    yes_t = torch.tensor(yes_ids, device=device)
    no_t = torch.tensor(no_ids, device=device)
    vals: dict[str, float] = {}
    for k, q in _VERIFY_QUESTIONS.items():
        prompt = f'Instruction that was requested: "{instruction}"\n{q}\nAnswer with one word: Yes or No.'
        messages = [
            {"role": "system", "content": "You are a strict image-editing verifier. Answer only Yes or No."},
            {"role": "user", "content": [
                {"type": "text", "text": "Original image:"}, {"type": "image", "image": "original"},
                {"type": "text", "text": "Edited image:"}, {"type": "image", "image": "edited"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[original, candidate], padding=True, return_tensors="pt")
        inputs = {kk: (v.to(device) if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1, :].float()
        ly = torch.logsumexp(logits[yes_t], dim=0)
        ln = torch.logsumexp(logits[no_t], dim=0)
        vals[k] = float(torch.softmax(torch.stack([ly, ln]), dim=0)[0].item())
    prod = 1.0
    for k in _VERIFY_KEYS:
        prod *= vals[k]
    vals["combined"] = prod ** 0.25
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-Edit-2509")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="configs/self_evolve/qwen_edit_2509_v45_ranked_preference_caption_extract_keeponly.yaml")
    ap.add_argument("--per-family-limit", type=int, default=0, help="0 = all keys; else N keys per family (stratified).")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--out", default="outputs/analysis/judge_ranker_validation/report.json")
    ap.add_argument("--pairwise", action="store_true", help="Forced-choice both-orders pairwise judging (fixes saturation).")
    ap.add_argument("--verify", action="store_true", help="Decomposed internal-VLM verification reward; measure per-category alignment with GPT.")
    ap.add_argument("--logprob", action="store_true", help="With --verify: read Yes/No token log-probs instead of verbalized probs (de-saturates).")
    ap.add_argument("--dry-run", action="store_true", help="Validate data plumbing only; skip the model.")
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))

    basic = json.loads(BASIC.read_text())
    s1 = json.loads(R1_SCORES.read_text())
    s2 = json.loads(R2_SCORES.read_text())

    # Build the working set: keys present everywhere, with both candidate images.
    per_fam: dict[str, int] = defaultdict(int)
    work: list[dict[str, Any]] = []
    for key, rec in basic.items():
        if key not in s1 or key not in s2:
            continue
        r1img, r2img = R1_DIR / f"{key}.png", R2_DIR / f"{key}.png"
        srcimg = src_path_for(rec)
        if not (r1img.exists() and r2img.exists() and srcimg.exists()):
            continue
        fam = rec["edit_type"]
        if args.per_family_limit and per_fam[fam] >= args.per_family_limit:
            continue
        per_fam[fam] += 1
        work.append(
            {
                "key": key,
                "edit_type": fam,
                "instruction": rec["prompt"],
                "src": str(srcimg),
                "r1": str(r1img),
                "r2": str(r2img),
                "s1": float(s1[key]),
                "s2": float(s2[key]),
            }
        )

    print(f"working set: {len(work)} keys; per-family: {dict(per_fam)}", flush=True)
    if args.dry_run:
        print("dry-run OK (plumbing valid); exiting before model load.", flush=True)
        return 0

    pipe = load_feature_pipe(args.model, args.device)
    print("judge pipe ready (transformer skipped).", flush=True)

    def load_img_side(p: str) -> Image.Image:
        im = Image.open(p).convert("RGB")
        if args.max_side > 0:
            w, h = im.size
            sc = min(1.0, args.max_side / max(w, h))
            if sc < 1.0:
                im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.Resampling.LANCZOS)
        return im

    if args.verify:
        # STRICT self-contained decomposed verification reward. DECISIVE test: does the
        # verification signal CORRELATE with GPT quality WITHIN each family, and does
        # selecting by it recover oracle headroom broadly (agreement > 0.5)?
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comps = list(_VERIFY_KEYS) + ["combined"]
        fam_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: {c: [] for c in comps})
        fam_gpt: dict[str, list[float]] = defaultdict(list)
        base_s, orc_s, pick_s = [], [], []
        fam_base2: dict[str, list[float]] = defaultdict(list)
        fam_orc2: dict[str, list[float]] = defaultdict(list)
        fam_pick2: dict[str, list[float]] = defaultdict(list)
        agree2: list[int] = []
        pk_out: list[dict[str, Any]] = []
        yes_ids, no_ids = (yesno_token_ids(pipe.processor) if args.logprob else (None, None))
        if args.logprob:
            print(f"log-prob verification ON: yes_ids={yes_ids} no_ids={no_ids}", flush=True)
        for i, w in enumerate(work):
            src = load_img_side(w["src"]); r1 = load_img_side(w["r1"]); r2 = load_img_side(w["r2"])
            true = [w["s1"], w["s2"]]
            try:
                if args.logprob:
                    v1 = verification_reward_logprob(pipe, w["instruction"], src, r1, yes_ids, no_ids)
                    v2 = verification_reward_logprob(pipe, w["instruction"], src, r2, yes_ids, no_ids)
                else:
                    v1 = verification_reward(pipe, w["instruction"], src, r1)
                    v2 = verification_reward(pipe, w["instruction"], src, r2)
            except Exception as exc:
                print(f"  [{i}] {w['key']} verify error: {exc}", flush=True)
                continue
            fam = w["edit_type"]
            for vv, tt in ((v1, true[0]), (v2, true[1])):
                for c in comps:
                    fam_vals[fam][c].append(vv[c])
                fam_gpt[fam].append(tt)
            base_s.append(true[0]); orc_s.append(max(true))
            fam_base2[fam].append(true[0]); fam_orc2[fam].append(max(true))
            pick = 0 if v1["combined"] >= v2["combined"] else 1
            pick_s.append(true[pick]); fam_pick2[fam].append(true[pick])
            if abs(true[0] - true[1]) > 1e-6:
                better = 0 if true[0] > true[1] else 1
                agree2.append(1 if pick == better else 0)
            pk_out.append({"key": w["key"], "edit_type": fam, "s1": true[0], "s2": true[1],
                           "v1": {k: round(v1[k], 3) for k in comps}, "v2": {k: round(v2[k], 3) for k in comps}, "pick": pick})
            if (i + 1) % 20 == 0 or i + 1 == len(work):
                allv = [v for f in fam_vals for v in fam_vals[f]["combined"]]
                allg = [g for f in fam_gpt for g in fam_gpt[f]]
                print(f"[{i+1}/{len(work)}] pooled corr(combined,gpt)={_pearson(allv, allg):+.3f} "
                      f"sel_recovered={recovered(pick_s, base_s, orc_s)['recovered_frac']} "
                      f"agree={round(statistics.mean(agree2), 3) if agree2 else None}", flush=True)
                json.dump({"partial": True, "n": len(base_s)}, open(out_path, "w"))
        fam_report = {}
        for fam in sorted(fam_gpt):
            g = fam_gpt[fam]
            row: dict[str, Any] = {"n_candidates": len(g),
                                   "gpt_std": round(statistics.pstdev(g), 4) if len(g) > 1 else 0.0}
            for c in comps:
                row[f"pearson_{c}"] = round(_pearson(fam_vals[fam][c], g), 3)
                row[f"spearman_{c}"] = round(_spearman(fam_vals[fam][c], g), 3)
            if fam_pick2[fam]:
                row["selection"] = recovered(fam_pick2[fam], fam_base2[fam], fam_orc2[fam])
            fam_report[fam] = row
        allv = [v for f in fam_vals for v in fam_vals[f]["combined"]]
        allg = [g for f in fam_gpt for g in fam_gpt[f]]
        overall = {
            "n_candidates": len(allg),
            "pooled_pearson_combined": round(_pearson(allv, allg), 3),
            "median_within_family_pearson_combined": round(statistics.median(
                [fam_report[f]["pearson_combined"] for f in fam_report]), 3) if fam_report else 0.0,
            "selection": recovered(pick_s, base_s, orc_s),
            "true_better_agreement": round(statistics.mean(agree2), 3) if agree2 else None,
        }
        json.dump({"overall": overall, "per_family": fam_report, "per_key": pk_out},
                  open(out_path, "w"), indent=2)
        print("\n=== VERIFICATION REWARD ALIGNMENT (per family) ===", flush=True)
        print(f"{'family':10s} {'n':>4s} {'gptSD':>6s} {'r_comb':>7s} {'r_real':>7s} {'r_corr':>7s} "
              f"{'r_pres':>7s} {'r_cln':>7s} {'selrec':>7s}", flush=True)
        for fam in sorted(fam_report, key=lambda f: -fam_report[f]["pearson_combined"]):
            r = fam_report[fam]
            sel = r.get("selection", {}).get("recovered_frac", 0.0)
            print(f"{fam:10s} {r['n_candidates']:4d} {r['gpt_std']:6.3f} "
                  f"{r['pearson_combined']:+7.3f} {r['pearson_realized']:+7.3f} {r['pearson_correct']:+7.3f} "
                  f"{r['pearson_preserved']:+7.3f} {r['pearson_clean']:+7.3f} {sel:+7.3f}", flush=True)
        print(f"\nOVERALL pooled_corr={overall['pooled_pearson_combined']:+.3f} "
              f"median_within_fam_corr={overall['median_within_family_pearson_combined']:+.3f} "
              f"sel_recovered={overall['selection']['recovered_frac']} "
              f"agree={overall['true_better_agreement']}", flush=True)
        print(f"report -> {out_path}\n=== VERIFICATION VALIDATION COMPLETE ===", flush=True)
        return 0

    if args.pairwise:
        # Forced-choice both-orders pairwise judging. Directly attacks absolute-score
        # saturation: the judge MUST commit to A or B; running both orders detects
        # position bias (inconsistent -> treated as a tie / no signal).
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base_s, orc_s, pick_s = [], [], []
        fam_base2: dict[str, list[float]] = defaultdict(list)
        fam_orc2: dict[str, list[float]] = defaultdict(list)
        fam_pick2: dict[str, list[float]] = defaultdict(list)
        agree2: list[int] = []
        consistent_n = 0
        pk_out: list[dict[str, Any]] = []
        for i, w in enumerate(work):
            src = load_img_side(w["src"])
            r1, r2 = load_img_side(w["r1"]), load_img_side(w["r2"])
            true = [w["s1"], w["s2"]]
            base_s.append(true[0]); orc_s.append(max(true))
            fam_base2[w["edit_type"]].append(true[0]); fam_orc2[w["edit_type"]].append(max(true))
            try:
                fwd = pairwise_choice(pipe, w["instruction"], src, r1, r2)   # A=r1 B=r2
                rev = pairwise_choice(pipe, w["instruction"], src, r2, r1)   # A=r2 B=r1
            except Exception as exc:
                print(f"  [{i}] {w['key']} pairwise error: {exc}", flush=True)
                pick_s.append(sum(true) / 2.0); fam_pick2[w["edit_type"]].append(sum(true) / 2.0)
                continue
            fwd_idx = None if fwd is None else (0 if fwd == "A" else 1)             # index into [r1,r2]
            rev_idx = None if rev is None else (1 if rev == "A" else 0)             # A=r2 -> r2 is idx1
            if fwd_idx is not None and fwd_idx == rev_idx:
                pick = fwd_idx; consistent = True; consistent_n += 1
            else:
                pick = None; consistent = False
            if pick is None:
                pscore = sum(true) / 2.0  # no reliable signal -> random-equivalent
            else:
                pscore = true[pick]
                if abs(true[0] - true[1]) > 1e-6:
                    better = 0 if true[0] > true[1] else 1
                    agree2.append(1 if pick == better else 0)
            pick_s.append(pscore); fam_pick2[w["edit_type"]].append(pscore)
            pk_out.append({"key": w["key"], "edit_type": w["edit_type"], "s1": true[0], "s2": true[1],
                           "fwd": fwd, "rev": rev, "consistent": consistent,
                           "pick": pick, "picked_true": pscore})
            if (i + 1) % 20 == 0 or i + 1 == len(work):
                rep = {
                    "mode": "pairwise_both_orders", "n": len(base_s),
                    "base_mean": round(statistics.mean(base_s), 4),
                    "oracle_mean": round(statistics.mean(orc_s), 4),
                    "pairwise": {**recovered(pick_s, base_s, orc_s),
                                 "true_better_agreement": round(statistics.mean(agree2), 3) if agree2 else None,
                                 "consistency_rate": round(consistent_n / len(base_s), 3),
                                 "agreement_n": len(agree2)},
                }
                fam_rep = {}
                for fam in fam_base2:
                    fr = {"n": len(fam_base2[fam]), "base": round(statistics.mean(fam_base2[fam]), 3),
                          "oracle": round(statistics.mean(fam_orc2[fam]), 3)}
                    if fam_pick2[fam]:
                        fr["pairwise"] = recovered(fam_pick2[fam], fam_base2[fam], fam_orc2[fam])
                    fam_rep[fam] = fr
                json.dump({"summary": rep, "per_family": fam_rep, "per_key": pk_out}, open(out_path, "w"), indent=2)
                pw = rep["pairwise"]
                print(f"[{i+1}/{len(work)}] base={rep['base_mean']} oracle={rep['oracle_mean']} "
                      f"pairwise=+{pw['gain_over_base']}({pw['recovered_frac']}) "
                      f"agree={pw['true_better_agreement']} consist={pw['consistency_rate']}", flush=True)
        print("\n=== PER-FAMILY (pairwise recovered_frac) ===", flush=True)
        final = json.load(open(out_path))
        for fam, row in sorted(final["per_family"].items(),
                               key=lambda x: -(x[1].get("pairwise", {}).get("recovered_frac", 0))):
            p = row.get("pairwise", {})
            print(f"  {fam:12s} n={row['n']:3d} base={row['base']:.2f} oracle={row['oracle']:.2f} "
                  f"pairwise=+{p.get('gain_over_base',0):.3f} ({p.get('recovered_frac',0)})", flush=True)
        print(f"\nreport -> {out_path}\n=== PAIRWISE JUDGE VALIDATION COMPLETE ===", flush=True)
        return 0

    from qwen_edit_project.self_evolve.backends import InternalRubricCEPREvaluator

    evaluator = InternalRubricCEPREvaluator(make_evaluator_config(args.config))
    evaluator._get_internal_pipe = lambda _editor=None: pipe  # type: ignore[assignment]

    def load_img(p: str) -> Image.Image:
        im = Image.open(p).convert("RGB")
        if args.max_side > 0:
            w, h = im.size
            sc = min(1.0, args.max_side / max(w, h))
            if sc < 1.0:
                im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.Resampling.LANCZOS)
        return im

    RANKERS = {
        "embedding": "cepr_pre_vlm_raw_reward",
        "judge": "internal_vlm_judge_score",
        "combined": "internal_vlm_judge_combined_raw_reward",
    }
    picks: dict[str, list[float]] = defaultdict(list)  # ranker -> picked true scores
    fam_picks: dict[tuple[str, str], list[float]] = defaultdict(list)
    base_s: list[float] = []
    orc_s: list[float] = []
    rnd_s: list[float] = []
    fam_base: dict[str, list[float]] = defaultdict(list)
    fam_orc: dict[str, list[float]] = defaultdict(list)
    agree: dict[str, list[int]] = defaultdict(list)  # ranker -> 1 if picked true-better (differing keys)
    per_key_out: list[dict[str, Any]] = []
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, w in enumerate(work):
        proposal = build_proposal(w["key"], FAMILY_MAP.get(w["edit_type"], w["edit_type"]), w["instruction"])
        source = load_img(w["src"])
        cands = [load_img(w["r1"]), load_img(w["r2"])]
        true = [w["s1"], w["s2"]]
        try:
            results = evaluator.score_group(proposal, source, cands, editor=None)
        except Exception as exc:
            print(f"  [{i}] {w['key']} score error: {exc}", flush=True)
            continue
        by_ci = {}
        for r in results:
            ci = int(r.signals.get("candidate_index", -1))
            by_ci[ci] = r
        if 0 not in by_ci or 1 not in by_ci:
            continue
        comp = {ci: by_ci[ci].component_scores for ci in (0, 1)}
        base_s.append(true[0])
        orc_s.append(max(true))
        rnd_s.append(sum(true) / 2.0)
        fam_base[w["edit_type"]].append(true[0])
        fam_orc[w["edit_type"]].append(max(true))
        row_rank = {}
        for rname, ckey in RANKERS.items():
            v0 = float(comp[0].get(ckey, 0.0) or 0.0)
            v1 = float(comp[1].get(ckey, 0.0) or 0.0)
            pick = 1 if v1 > v0 else 0
            picks[rname].append(true[pick])
            fam_picks[(rname, w["edit_type"])].append(true[pick])
            row_rank[rname] = {"v0": round(v0, 4), "v1": round(v1, 4), "pick": pick, "picked_true": true[pick]}
            if abs(true[0] - true[1]) > 1e-6:
                better = 0 if true[0] > true[1] else 1
                agree[rname].append(1 if pick == better else 0)
        per_key_out.append(
            {"key": w["key"], "edit_type": w["edit_type"], "s1": true[0], "s2": true[1], "rankers": row_rank}
        )
        if (i + 1) % 20 == 0 or i + 1 == len(work):
            summary = {
                "n": len(base_s),
                "base_mean": round(statistics.mean(base_s), 4),
                "random_mean": round(statistics.mean(rnd_s), 4),
                "oracle_mean": round(statistics.mean(orc_s), 4),
                "rankers": {
                    rn: {
                        **recovered(picks[rn], base_s, orc_s),
                        "true_better_agreement": round(statistics.mean(agree[rn]), 3) if agree[rn] else None,
                    }
                    for rn in RANKERS
                },
            }
            json.dump({"summary": summary, "per_key": per_key_out}, open(out_path, "w"), indent=2)
            print(
                f"[{i+1}/{len(work)}] base={summary['base_mean']} oracle={summary['oracle_mean']} "
                + " ".join(
                    f"{rn}:+{summary['rankers'][rn]['gain_over_base']}({summary['rankers'][rn]['recovered_frac']})"
                    for rn in RANKERS
                ),
                flush=True,
            )

    # Final per-family recovery for the judge.
    fam_report = {}
    for fam in fam_base:
        row = {"n": len(fam_base[fam]), "base": round(statistics.mean(fam_base[fam]), 3),
               "oracle": round(statistics.mean(fam_orc[fam]), 3)}
        for rn in RANKERS:
            if fam_picks[(rn, fam)]:
                row[rn] = recovered(fam_picks[(rn, fam)], fam_base[fam], fam_orc[fam])
        fam_report[fam] = row
    final = json.load(open(out_path))
    final["per_family"] = fam_report
    json.dump(final, open(out_path, "w"), indent=2)
    print("\n=== PER-FAMILY (judge recovered_frac) ===", flush=True)
    for fam, row in sorted(fam_report.items(), key=lambda x: -(x[1].get("judge", {}).get("recovered_frac", 0))):
        j = row.get("judge", {})
        print(f"  {fam:12s} n={row['n']:3d} base={row['base']:.2f} oracle={row['oracle']:.2f} "
              f"judge=+{j.get('gain_over_base',0):.3f} ({j.get('recovered_frac',0)})", flush=True)
    print(f"\nreport -> {out_path}\n=== JUDGE RANKER VALIDATION COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
