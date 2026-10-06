"""Reward-verified best-of-N self-distillation pool for the `extract` family.

For each disjoint source image (COCO / editing_datasets, NOT the ImgEdit test set):
  1. Name the dominant object with GroundingDINO -> instruction "Extract the {obj}."
  2. Sample the BASE Qwen-Image-Edit-2509 model N times (different seeds).
  3. Score each sample with a cheap no-GPT extract reward
     (background/border whiteness x object-present x sharpness) — the same signal
     shown to separate base extraction success (0.208) from failure (0.014).
  4. Keep the single best sample as the SFT target if it clears the reward gate.

Output: an SFT manifest (source, "Extract the {obj}.", best_target) that folds the
best-of-N extraction capability into supervised targets = the model's OWN best
outputs, not inferior segmentation cutouts.
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

# Common, concrete object classes GroundingDINO can name reliably.
CLASSES = [
    "person", "man", "woman", "child", "dog", "cat", "bird", "horse", "cow",
    "sheep", "elephant", "bear", "zebra", "giraffe", "car", "truck", "bus",
    "motorcycle", "bicycle", "boat", "airplane", "train", "chair", "couch",
    "potted plant", "bottle", "wine glass", "cup", "bowl", "laptop", "keyboard",
    "cell phone", "book", "clock", "vase", "teddy bear", "backpack", "umbrella",
    "handbag", "suitcase", "skateboard", "surfboard", "tennis racket", "guitar",
    "toy", "sculpture", "flower", "cake", "pizza", "sports ball",
]


def extract_reward(img: Image.Image) -> tuple[float, dict]:
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    near_white = (a > 244).all(-1)
    h, w = near_white.shape
    b = max(2, int(0.08 * min(h, w)))
    border = np.concatenate([
        near_white[:b].ravel(), near_white[-b:].ravel(),
        near_white[:, :b].ravel(), near_white[:, -b:].ravel(),
    ])
    border_white = float(border.mean())
    white_frac = float(near_white.mean())
    obj_frac = 1.0 - white_frac
    present = 1.0 if 0.02 < obj_frac < 0.92 else 0.0
    # sharpness proxy: laplacian variance on the object (non-white) region
    gray = a.mean(-1)
    lap = np.abs(np.gradient(np.gradient(gray, axis=0), axis=0) +
                 np.gradient(np.gradient(gray, axis=1), axis=1))
    obj_mask = ~near_white
    sharp = float(lap[obj_mask].mean()) if obj_mask.any() else 0.0
    sharp_n = min(sharp / 8.0, 1.0)
    reward = border_white * present * (0.5 + 0.5 * sharp_n)
    return reward, {"border_white": round(border_white, 3), "obj_frac": round(obj_frac, 3),
                    "sharp": round(sharp_n, 3), "reward": round(reward, 4)}


class ObjectNamer:
    def __init__(self, device: str):
        self.device = device
        self.proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.det = (AutoModelForZeroShotObjectDetection
                    .from_pretrained("IDEA-Research/grounding-dino-tiny").to(device).eval())
        self.text = ". ".join(CLASSES) + " ."

    def name(self, image: Image.Image):
        inp = self.proc(images=image, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.det(**inp)
        try:
            res = self.proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=0.35, text_threshold=0.30,
                target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = self.proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=0.35, target_sizes=[image.size[::-1]])[0]
        if len(res["boxes"]) == 0:
            return None, 0.0
        labels = res.get("text_labels") or res.get("labels")
        W, H = image.size
        best_i, best_area = -1, -1.0
        for i, box in enumerate(res["boxes"].tolist()):
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0) / (W * H)
            if 0.03 < area < 0.85 and area * float(res["scores"][i]) > best_area:
                best_area, best_i = area * float(res["scores"][i]), i
        if best_i < 0:
            return None, 0.0
        lab = labels[best_i] if labels is not None else "object"
        lab = str(lab).strip().strip(".") or "object"
        return lab, float(res["scores"][best_i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="data/unlabeled/raw/coco2017")
    ap.add_argument("--out", default="outputs/edit_pairs/extract_selfdistill")
    ap.add_argument("--process", type=int, default=200, help="pool images to attempt")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--target", type=int, default=120, help="stop after this many accepted")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--reward-gate", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offset", type=int, default=0, help="skip this many of the shuffled pool (for sharding)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.sources)
    imgs = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[args.offset : args.offset + args.process]
    print(f"pool candidates: {len(imgs)} from {src_dir} (offset {args.offset})")

    namer = ObjectNamer(args.device)
    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths="Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors",
        checkpoint_path=None, model_type="base", device=args.device,
        torch_dtype="bfloat16", backend="official_diffusers",
        base_model="Qwen/Qwen-Image-Edit-2509",
    )
    gen_base = {"num_inference_steps": args.steps, "true_cfg_scale": 4.0,
                "guidance_scale": 1.0, "negative_prompt": " ", "num_images_per_prompt": 1}

    manifest = []
    accepted = 0

    def flush_manifests():
        json.dump(manifest, open(out / "sft_manifest.json", "w"), indent=2)
        train = [{"prompt": m["prompt"], "image": m["target"], "edit_image": m["source"],
                  "sample_weight": 1.0, "family": "extract", "record_key": m["key"]}
                 for m in manifest]
        json.dump(train, open(out / "sft_train_manifest.json", "w"), indent=2)

    for idx, src in enumerate(imgs, start=args.offset):
        if accepted >= args.target:
            break
        try:
            image = Image.open(src).convert("RGB")
        except Exception:
            continue
        obj, det = namer.name(image)
        if obj is None:
            continue
        prompt = f"Extract the {obj}."
        best = None
        rewards = []
        for s in range(args.n_samples):
            try:
                output = render_edit(pipe, prompt, [src], {**gen_base, "seed": 1000 + idx * 100 + s})
                cand = output.images[0] if hasattr(output, "images") else output
            except Exception as exc:
                print(f"[gen-fail] {src.name} seed{s}: {exc}")
                continue
            r, meta = extract_reward(cand)
            rewards.append(round(r, 4))
            if best is None or r > best[0]:
                best = (r, cand, meta)
        if best is None:
            continue
        r, cand, meta = best
        status = "ACCEPT" if r >= args.reward_gate else "reject"
        if r >= args.reward_gate:
            tgt_path = out / "targets" / f"{idx:05d}.png"
            cand.save(tgt_path)
            manifest.append({
                "key": f"{idx:05d}", "source": str(src), "prompt": prompt, "object": obj,
                "det_score": round(det, 3), "target": str(tgt_path),
                "best_reward": round(r, 4), "reward_meta": meta,
                "all_rewards": rewards, "n_samples": args.n_samples,
            })
            accepted += 1
            flush_manifests()
        print(f"[{idx+1}/{len(imgs)}] {status} obj={obj!r} bestR={r:.3f} "
              f"bw={meta['border_white']} rewards={rewards} acc={accepted}", flush=True)

    flush_manifests()
    if manifest:
        singles = [m["all_rewards"][0] for m in manifest]
        bests = [m["best_reward"] for m in manifest]
        print(f"\nAccepted {accepted} targets -> {out}/sft_manifest.json")
        print(f"training-pool best-of-{args.n_samples} reward mean={np.mean(bests):.3f} "
              f"vs single-shot(seed0) mean={np.mean(singles):.3f} (lift {np.mean(bests)-np.mean(singles):+.3f})")


if __name__ == "__main__":
    main()
