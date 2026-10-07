"""Reward-verified best-of-N self-distillation pool builder — MULTI-FAMILY.

Generalizes the proven extract recipe (build_extract_selfdistill.py) to the other
starved ImgEdit families with cheap, internal, eval-aligned rewards:

  extract    : background whiteness x object-present x sharpness            (proven +0.59)
  remove     : object-gone (detector) x outside-region preserved x inside-changed
  background : subject-region preserved x background-region changed
  add        : new-object present (detector) x rest preserved
  adjust     : attribute (color/material) changed x object-region changed x background preserved (VLM)

Each reward measures "did the rubric-requested edit happen, cleanly" from features
we already compute (GroundingDINO boxes + pixel region stats) — no external judge.
Source images are COCO (disjoint from the ImgEdit test set). We sample the BASE
editor N times, keep its own best clean edit by the family reward, and emit an SFT
target. Combine families into one manifest to train a single multi-family LoRA.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit

CLASSES = [
    "person", "man", "woman", "child", "dog", "cat", "bird", "horse", "cow",
    "sheep", "elephant", "bear", "zebra", "giraffe", "car", "truck", "bus",
    "motorcycle", "bicycle", "boat", "airplane", "train", "chair", "couch",
    "potted plant", "bottle", "wine glass", "cup", "bowl", "laptop", "keyboard",
    "cell phone", "book", "clock", "vase", "teddy bear", "backpack", "umbrella",
    "handbag", "suitcase", "skateboard", "surfboard", "guitar", "toy", "flower",
    "cake", "pizza", "sports ball", "kite", "bench", "traffic light", "fire hydrant",
]
BACKGROUNDS = [
    "a sandy beach at sunset", "a snowy mountain landscape", "a lush green forest",
    "a busy city street at night", "a plain studio backdrop", "a desert with dunes",
    "an underwater coral reef", "a starry night sky", "a blooming flower field",
]
ACTIONS = [
    "raising one arm", "jumping in the air", "turning to look to the side",
    "sitting down", "running forward", "opening its mouth wide", "lying down", "stretching",
]
ANIMATE = {
    "person", "man", "woman", "child", "dog", "cat", "bird", "horse", "cow",
    "sheep", "elephant", "bear", "zebra", "giraffe",
}
COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "white", "black",
]
MATERIALS = ["wood", "metal", "glass", "gold", "marble", "plastic"]
# Yes/No surface forms for reading verification log-probs from the editor's own Qwen2.5-VL.
_YES_WORDS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
_NO_WORDS = ("No", " No", "no", " no", "NO", " NO")
_VERIFY_QUESTIONS = {
    "realized": "Was the change requested by the instruction actually applied in the edited image (clearly present, not missing)?",
    "correct": "Does the applied change match exactly what the instruction asked for (correct object, attribute, and location)?",
}


# ---------------- cheap region stats ----------------
def to_arr(img: Image.Image, size) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize(size, Image.BILINEAR)).astype(np.float32)


def region_sim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 16:
        return 1.0
    return float(1.0 - np.abs(a[mask] - b[mask]).mean() / 255.0)


def box_mask(size, box) -> np.ndarray:
    w, h = size
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def extract_reward(cand: Image.Image):
    a = np.asarray(cand.convert("RGB")).astype(np.float32)
    nw = (a > 244).all(-1)
    h, w = nw.shape
    b = max(2, int(0.08 * min(h, w)))
    border = np.concatenate([nw[:b].ravel(), nw[-b:].ravel(), nw[:, :b].ravel(), nw[:, -b:].ravel()])
    bw = float(border.mean()); obj = 1.0 - float(nw.mean())
    present = 1.0 if 0.02 < obj < 0.92 else 0.0
    g = a.mean(-1)
    lap = np.abs(np.gradient(np.gradient(g, axis=0), axis=0) + np.gradient(np.gradient(g, axis=1), axis=1))
    sharp = min(float(lap[~nw].mean()) / 8.0, 1.0) if (~nw).any() else 0.0
    r = bw * present * (0.5 + 0.5 * sharp)
    return r, {"border_white": round(bw, 3), "obj": round(obj, 3)}


class Builder:
    def __init__(self, device: str, family: str, reward_mode: str = "detector"):
        self.device = device
        self.family = family
        self.reward_mode = reward_mode
        self.proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.det = (AutoModelForZeroShotObjectDetection
                    .from_pretrained("IDEA-Research/grounding-dino-tiny").to(device).eval())
        self.classes_text = ". ".join(CLASSES) + " ."
        # VLM verifier (attached lazily from the editor pipeline's own Qwen2.5-VL).
        self.vmodel = None
        self.vproc = None
        self.yes_ids = None
        self.no_ids = None

    def attach_verifier(self, pipe) -> None:
        """Reuse the editor pipeline's Qwen2.5-VL text_encoder + processor for log-prob
        verification (no second model load). Validated as a strong did-edit-happen filter."""
        self.vmodel = pipe.text_encoder
        self.vproc = pipe.processor
        tok = self.vproc.tokenizer

        def first_ids(words):
            ids = set()
            for w in words:
                enc = tok.encode(w, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            return sorted(ids)

        self.yes_ids, self.no_ids = first_ids(_YES_WORDS), first_ids(_NO_WORDS)
        print(f"[verifier] attached; yes_ids={self.yes_ids} no_ids={self.no_ids}", flush=True)

    def vlm_success(self, instruction: str, source: Image.Image, candidate: Image.Image) -> float:
        """P(realized) * P(correct) read from Yes/No token log-probs. Continuous did-edit-happen
        signal (verbalized probs saturate; log-probs de-saturate -> validated filter)."""
        yes_t = torch.tensor(self.yes_ids, device=self.vmodel.device if hasattr(self.vmodel, "device")
                             else next(self.vmodel.parameters()).device)
        no_t = torch.tensor(self.no_ids, device=yes_t.device)
        ps = []
        for q in _VERIFY_QUESTIONS.values():
            prompt = f'Instruction that was requested: "{instruction}"\n{q}\nAnswer with one word: Yes or No.'
            messages = [
                {"role": "system", "content": "You are a strict image-editing verifier. Answer only Yes or No."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Original image:"}, {"type": "image", "image": "o"},
                    {"type": "text", "text": "Edited image:"}, {"type": "image", "image": "e"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            text = self.vproc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.vproc(text=[text], images=[source, candidate], padding=True, return_tensors="pt")
            inputs = {k: (v.to(yes_t.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.vmodel(**inputs)
            logits = out.logits[0, -1, :].float()
            ly = torch.logsumexp(logits[yes_t], dim=0)
            ln = torch.logsumexp(logits[no_t], dim=0)
            ps.append(float(torch.softmax(torch.stack([ly, ln]), dim=0)[0].item()))
        return (ps[0] * ps[1]) ** 0.5

    def detect(self, image: Image.Image, text: str, box_thr=0.30, text_thr=0.25):
        inp = self.proc(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.det(**inp)
        try:
            res = self.proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=box_thr, text_threshold=text_thr,
                target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = self.proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=box_thr, target_sizes=[image.size[::-1]])[0]
        return res

    def name_dominant(self, image: Image.Image):
        res = self.detect(image, self.classes_text)
        if len(res["boxes"]) == 0:
            return None, None, 0.0
        W, H = image.size
        best_i, best = -1, -1.0
        labels = res.get("text_labels") or res.get("labels")
        for i, box in enumerate(res["boxes"].tolist()):
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0) / (W * H)
            if 0.03 < area < 0.9 and area * float(res["scores"][i]) > best:
                best, best_i = area * float(res["scores"][i]), i
        if best_i < 0:
            return None, None, 0.0
        lab = str(labels[best_i] if labels is not None else "object").strip().strip(".") or "object"
        return lab, res["boxes"][best_i].tolist(), float(res["scores"][best_i])

    def detect_score(self, image: Image.Image, query: str) -> float:
        res = self.detect(image, query.lower().strip() + " .", box_thr=0.25, text_thr=0.2)
        if len(res["scores"]) == 0:
            return 0.0
        return float(res["scores"].max())

    def prompt_and_setup(self, image, rng):
        """Return (prompt, aux) or (None, None) if unsuitable."""
        obj, box, det = self.name_dominant(image)
        if obj is None:
            return None, None
        if self.family == "extract":
            return f"Extract the {obj}.", {"obj": obj, "box": box}
        if self.family == "remove":
            return f"Remove the {obj} from the image.", {"obj": obj, "box": box}
        if self.family == "add":
            new = rng.choice([c for c in CLASSES if c != obj])
            return f"Add a {new} to the image.", {"obj": new, "box": box}
        if self.family == "background":
            bg = rng.choice(BACKGROUNDS)
            return f"Change the background to {bg}.", {"obj": obj, "box": box}
        if self.family == "replace":
            new = rng.choice([c for c in CLASSES if c != obj])
            return f"Replace the {obj} with a {new}.", {"obj": obj, "new": new, "box": box}
        if self.family == "action":
            if obj not in ANIMATE:
                return None, None
            act = rng.choice(ACTIONS)
            return f"Make the {obj} appear to be {act}.", {"obj": obj, "box": box, "act": act}
        if self.family == "adjust":
            if rng.random() < 0.5:
                val = rng.choice(COLORS)
                return f"Change the color of the {obj} to {val}.", {"obj": obj, "box": box, "attr": "color", "val": val}
            val = rng.choice(MATERIALS)
            return f"Make the {obj} look like it is made of {val}.", {"obj": obj, "box": box, "attr": "material", "val": val}
        raise ValueError(self.family)

    def reward(self, image, cand, aux, instruction=None):
        if self.family == "extract":
            return extract_reward(cand)
        size = image.size
        src_a = to_arr(image, size)
        cand_a = to_arr(cand, size)
        m_in = box_mask(size, aux["box"])
        m_out = ~m_in
        if self.family == "remove":
            out_pres = region_sim(src_a, cand_a, m_out)
            in_changed = 1.0 - region_sim(src_a, cand_a, m_in)
            if self.reward_mode == "edit_realization":
                # Detector-free: the VLM did-remove-happen signal drives selection; the
                # grounder is used ONLY for the localization box, never as the reward term.
                if self.vmodel is None or not instruction:
                    raise ValueError("edit_realization reward requires --use-vlm")
                v = self.vlm_success(instruction, image, cand)
                ok = out_pres > 0.85 and in_changed > 0.06 and v > 0.4
                r = v * out_pres * min(in_changed * 3, 1.0) if ok else 0.0
                return r, {"out_pres": round(out_pres, 3), "in_chg": round(in_changed, 3),
                           "vlm": round(v, 3), "mode": "ER"}
            gone = 1.0 - self.detect_score(cand, aux["obj"])
            ok = out_pres > 0.85 and gone > 0.4 and in_changed > 0.06
            r = gone * out_pres * min(in_changed * 3, 1.0) if ok else 0.0
            meta = {"out_pres": round(out_pres, 3), "in_chg": round(in_changed, 3), "gone": round(gone, 3)}
            if r > 0 and self.vmodel is not None and instruction:
                v = self.vlm_success(instruction, image, cand)
                meta["vlm"] = round(v, 3)
                r = (r * v) ** 0.5  # require detector AND VLM to agree
            return r, meta
        if self.family == "background":
            subj_pres = region_sim(src_a, cand_a, m_in)
            bg_changed = 1.0 - region_sim(src_a, cand_a, m_out)
            ok = subj_pres > 0.72 and bg_changed > 0.12
            r = subj_pres * min(bg_changed * 2, 1.0) if ok else 0.0
            return r, {"subj_pres": round(subj_pres, 3), "bg_chg": round(bg_changed, 3)}
        if self.family == "add":
            rest = region_sim(src_a, cand_a, m_out)  # areas away from source subject stay-ish
            if self.reward_mode == "edit_realization":
                if self.vmodel is None or not instruction:
                    raise ValueError("edit_realization reward requires --use-vlm")
                v = self.vlm_success(instruction, image, cand)
                ok = v > 0.4
                r = v * (0.5 + 0.5 * rest) if ok else 0.0
                return r, {"rest": round(rest, 3), "vlm": round(v, 3), "mode": "ER"}
            present = self.detect_score(cand, aux["obj"])
            ok = present > 0.35
            r = present * (0.5 + 0.5 * rest) if ok else 0.0
            return r, {"present": round(present, 3), "rest": round(rest, 3)}
        if self.family == "replace":
            bg_pres = region_sim(src_a, cand_a, m_out)
            if self.reward_mode == "edit_realization":
                if self.vmodel is None or not instruction:
                    raise ValueError("edit_realization reward requires --use-vlm")
                v = self.vlm_success(instruction, image, cand)
                ok = bg_pres > 0.70 and v > 0.4
                r = v * (0.5 + 0.5 * bg_pres) if ok else 0.0
                return r, {"bg_pres": round(bg_pres, 3), "vlm": round(v, 3), "mode": "ER"}
            old_gone = 1.0 - self.detect_score(cand, aux["obj"])
            new_present = self.detect_score(cand, aux["new"])
            ok = new_present > 0.35 and old_gone > 0.30 and bg_pres > 0.70
            r = new_present * old_gone * (0.5 + 0.5 * bg_pres) if ok else 0.0
            meta = {"old_gone": round(old_gone, 3), "new_present": round(new_present, 3), "bg_pres": round(bg_pres, 3)}
            if r > 0 and self.vmodel is not None and instruction:
                v = self.vlm_success(instruction, image, cand)
                meta["vlm"] = round(v, 3)
                r = (r * v) ** 0.5
            return r, meta
        if self.family == "adjust":
            # Attribute change (color/material): the object stays in place, its attribute
            # changes, and the background is preserved. VLM-family (no programmatic proxy).
            if self.vmodel is None or not instruction:
                raise ValueError("family adjust requires --use-vlm")
            in_changed = 1.0 - region_sim(src_a, cand_a, m_in)
            bg_pres = region_sim(src_a, cand_a, m_out)
            v = self.vlm_success(instruction, image, cand)
            ok = bg_pres > 0.6 and in_changed > 0.06 and v > 0.4
            r = v * (0.5 + 0.5 * bg_pres) if ok else 0.0
            return r, {"in_chg": round(in_changed, 3), "bg_pres": round(bg_pres, 3),
                       "vlm": round(v, 3), "mode": "adjust"}
        if self.family in ("action", "compose"):
            # No reliable programmatic proxy -> use the validated VLM did-edit-happen filter,
            # gated by background preservation (edit should change the subject, keep the scene).
            if self.vmodel is None or not instruction:
                raise ValueError(f"family {self.family} requires --use-vlm")
            v = self.vlm_success(instruction, image, cand)
            bg_pres = region_sim(src_a, cand_a, m_out)
            ok = v > 0.4 and bg_pres > 0.5
            r = v * (0.5 + 0.5 * bg_pres) if ok else 0.0
            return r, {"vlm": round(v, 3), "bg_pres": round(bg_pres, 3)}
        raise ValueError(self.family)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["extract", "remove", "background", "add", "replace", "action", "compose", "adjust"])
    ap.add_argument("--use-vlm", action="store_true",
                    help="Attach the editor's Qwen2.5-VL as a log-prob did-edit-happen verifier (hybrid reward).")
    ap.add_argument("--reward-mode", choices=["detector", "edit_realization"], default="detector",
                    help="detector = original detector-anchored reward; edit_realization = detector-free "
                         "VLM did-edit-happen reward (localization box still from the grounder). Needs --use-vlm.")
    ap.add_argument("--sources", default="data/unlabeled/raw/coco2017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--process", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--reward-gate", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out); (out / "targets").mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in Path(args.sources).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[args.offset: args.offset + args.process]
    print(f"[{args.family}] pool candidates: {len(imgs)} (offset {args.offset})")

    b = Builder(args.device, args.family, reward_mode=args.reward_mode)
    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths="Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors",
        checkpoint_path=None, model_type="base", device=args.device,
        torch_dtype="bfloat16", backend="official_diffusers", base_model="Qwen/Qwen-Image-Edit-2509")
    gen = {"num_inference_steps": args.steps, "true_cfg_scale": 4.0, "guidance_scale": 1.0,
           "negative_prompt": " ", "num_images_per_prompt": 1}
    if args.use_vlm or args.family in ("action", "compose", "adjust") or args.reward_mode == "edit_realization":
        b.attach_verifier(pipe)
    rng = random.Random(args.seed + 1)

    manifest = []
    accepted = 0

    def flush():
        json.dump([{"prompt": m["prompt"], "image": m["target"], "edit_image": m["source"],
                    "sample_weight": 1.0, "family": args.family, "record_key": m["key"]} for m in manifest],
                  open(out / "sft_train_manifest.json", "w"), indent=2)
        json.dump(manifest, open(out / "manifest.json", "w"), indent=2)

    for idx, src in enumerate(imgs, start=args.offset):
        if accepted >= args.target:
            break
        try:
            image = Image.open(src).convert("RGB")
        except Exception:
            continue
        prompt, aux = b.prompt_and_setup(image, rng)
        if prompt is None:
            continue
        best = None; rewards = []
        for s in range(args.n_samples):
            try:
                o = render_edit(pipe, prompt, [src], {**gen, "seed": 1000 + idx * 100 + s})
                cand = o.images[0] if hasattr(o, "images") else o
            except Exception as exc:
                print(f"[gen-fail] {src.name} s{s}: {exc}"); continue
            r, meta = b.reward(image, cand, aux, instruction=prompt)
            rewards.append(round(r, 4))
            if best is None or r > best[0]:
                best = (r, cand, meta)
        if best is None:
            continue
        r, cand, meta = best
        status = "ACCEPT" if r >= args.reward_gate else "reject"
        if r >= args.reward_gate:
            tp = out / "targets" / f"{idx:05d}.png"; cand.save(tp)
            manifest.append({"key": f"{args.family}_{idx:05d}", "source": str(src), "prompt": prompt,
                             "target": str(tp), "best_reward": round(r, 4), "meta": meta, "all_rewards": rewards})
            accepted += 1; flush()
        print(f"[{args.family}][{idx+1}] {status} p={prompt[:40]!r} bestR={r:.3f} {meta} rewards={rewards} acc={accepted}", flush=True)

    flush()
    if manifest:
        print(f"\n[{args.family}] accepted {accepted} -> {out}/sft_train_manifest.json")


if __name__ == "__main__":
    main()
