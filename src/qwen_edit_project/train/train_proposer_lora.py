from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_edit_project.train.training_safety import (
    require_completed_steps,
    resolve_training_schedule,
    resume_position,
    seed_for_process,
)


def _load_records(path: Path, min_reward: float) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("use_for_sft", False):
                continue
            if float(record.get("reward", 0.0)) < min_reward:
                continue
            if not record.get("source_image") or not record.get("target_json"):
                continue
            records.append(record)
    if not records:
        raise ValueError(f"No proposer SFT records selected from {path}")
    return records


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def _checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        step = _checkpoint_step(path)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _rotate_checkpoints(output_dir: Path, total_limit: int | None) -> None:
    if not total_limit or total_limit <= 0:
        return
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))
    checkpoints = sorted(checkpoints, key=lambda item: item[0])
    if len(checkpoints) < total_limit:
        return
    for _, path in checkpoints[: len(checkpoints) - total_limit + 1]:
        shutil.rmtree(path)


class ProposerDataset:
    def __init__(self, records: list[dict[str, Any]], base_dir: Path):
        self.records = records
        self.base_dir = base_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = _resolve(record["source_image"], self.base_dir)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        return {
            "image": image,
            "prompt": record["prompt"],
            "target": record["target_json"],
            "reward": float(record.get("reward", 1.0)),
        }


class ProposerCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def _chat_text(self, prompt: str, target: str | None) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a research proposer for image-editing self-training.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        if target is not None:
            messages.append({"role": "assistant", "content": target})
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        images = [item["image"] for item in batch]
        full_texts = [self._chat_text(item["prompt"], item["target"]) for item in batch]
        prompt_texts = [self._chat_text(item["prompt"], None) for item in batch]
        inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100
        inputs["labels"] = labels
        inputs["reward_weights"] = torch.tensor(
            [float(item.get("reward", 1.0)) for item in batch],
            dtype=torch.float32,
        )
        return inputs


def _model_class(model_class: str):
    if model_class == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    if model_class != "auto":
        raise ValueError(f"Unsupported proposer model_class: {model_class}")
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Qwen-Image-Edit VLM proposer LoRA from self-evolve rewards.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_subfolder", default=None)
    parser.add_argument("--processor_subfolder", default=None)
    parser.add_argument("--model_class", default="auto", choices=["auto", "qwen2_5_vl"])
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min_reward", type=float, default=0.35)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=0)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--reward_weighting", action="store_true")
    parser.add_argument("--min_reward_weight", type=float, default=0.25)
    parser.add_argument("--max_reward_weight", type=float, default=1.50)
    args = parser.parse_args()

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be positive.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive when provided.")

    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, get_scheduler, set_seed

    from qwen_edit_project.utils.device import resolve_torch_dtype
    from qwen_edit_project.utils.paths import ensure_dir

    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    base_dir = Path(args.dataset_base_path).resolve()
    records = _load_records(Path(args.train_jsonl).resolve(), args.min_reward)
    processor_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.processor_subfolder:
        processor_kwargs["subfolder"] = args.processor_subfolder
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, **processor_kwargs)
    dtype = resolve_torch_dtype(torch, args.torch_dtype, accelerator.device)
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.model_subfolder:
        model_kwargs["subfolder"] = args.model_subfolder
    model = _model_class(args.model_class).from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[part.strip() for part in args.lora_target_modules.split(",") if part.strip()],
        )
        model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    output_dir = ensure_dir(Path(args.output_dir))
    dataset = ProposerDataset(records, base_dir)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=ProposerCollator(processor),
        generator=dataloader_generator,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    requested_max_train_steps = args.max_train_steps
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    schedule = resolve_training_schedule(
        batches_per_epoch=len(dataloader),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        requested_num_train_epochs=args.num_train_epochs,
        requested_max_train_steps=requested_max_train_steps,
    )
    max_steps = schedule.max_train_steps
    num_train_epochs = schedule.num_train_epochs
    scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    scheduler = accelerator.prepare_scheduler(scheduler)
    if accelerator.num_processes > 1:
        # The DataLoader has its own common generator, so reseeding global RNG
        # here only makes dropout and other model stochasticity rank-specific.
        set_seed(seed_for_process(args.seed, accelerator.process_index))

    global_step = 0
    starting_epoch = 0
    resume_step = 0
    resume_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            resume_path = _latest_checkpoint(output_dir)
        else:
            resume_path = Path(args.resume_from_checkpoint)
            if not resume_path.is_absolute():
                resume_path = output_dir / resume_path
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(
                f"Requested resume checkpoint '{args.resume_from_checkpoint}' does not exist."
            )
        if not resume_path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint is not a directory: {resume_path}")
        checkpoint_step = _checkpoint_step(resume_path)
        if checkpoint_step is None:
            raise ValueError(f"Resume checkpoint directory must end in an integer step: {resume_path}")
        if checkpoint_step > max_steps:
            raise ValueError(
                f"Resume checkpoint step {checkpoint_step} exceeds requested max_train_steps={max_steps}."
            )
        accelerator.print(f"Resuming proposer LoRA training from {resume_path}")
        accelerator.load_state(str(resume_path))
        global_step = checkpoint_step
        position = resume_position(
            global_step=global_step,
            batches_per_epoch=len(dataloader),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        starting_epoch = position.first_epoch
        resume_step = position.batches_to_skip

    model.train()
    for epoch in range(starting_epoch, num_train_epochs):
        active_dataloader = dataloader
        if resume_step:
            if hasattr(accelerator, "skip_first_batches"):
                active_dataloader = accelerator.skip_first_batches(dataloader, resume_step)
            else:
                active_dataloader = itertools.islice(dataloader, resume_step, None)
            resume_step = 0
        for batch in active_dataloader:
            with accelerator.accumulate(model):
                reward_weights = batch.pop("reward_weights", None)
                outputs = model(**batch)
                if args.reward_weighting and reward_weights is not None:
                    labels = batch["labels"]
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    token_mask = shift_labels.ne(-100)
                    safe_labels = shift_labels.masked_fill(~token_mask, 0)
                    token_losses = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        safe_labels.view(-1),
                        reduction="none",
                    ).view_as(shift_labels)
                    example_losses = (token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
                    weights = reward_weights.to(example_losses.device, dtype=example_losses.dtype)
                    weights = weights.clamp(min=args.min_reward_weight, max=args.max_reward_weight)
                    weights = weights / weights.mean().clamp_min(1e-6)
                    loss = (example_losses * weights).mean()
                else:
                    loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if args.logging_steps and global_step % args.logging_steps == 0:
                    accelerator.print(
                        json.dumps(
                            {
                                "event": "proposer_train_step",
                                "epoch": epoch,
                                "global_step": global_step,
                                "max_steps": max_steps,
                                "loss": float(loss.detach().item()),
                                "lr": float(scheduler.get_last_lr()[0]),
                            },
                            ensure_ascii=True,
                        )
                    )
                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        _rotate_checkpoints(output_dir, args.checkpoints_total_limit)
                    accelerator.wait_for_everyone()
                    checkpoint_dir = ensure_dir(output_dir / f"checkpoint-{global_step}")
                    accelerator.save_state(str(checkpoint_dir))
                    unwrapped = accelerator.unwrap_model(model)
                    if accelerator.is_main_process:
                        unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True)
                        processor.save_pretrained(checkpoint_dir)
                        accelerator.print(f"Saved proposer state checkpoint to {checkpoint_dir}")
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    require_completed_steps(
        global_step=global_step,
        requested_steps=max_steps,
        trainer_name="Proposer LoRA trainer",
    )
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        processor.save_pretrained(output_dir)
        metadata = {
            "model_name_or_path": args.model_name_or_path,
            "model_subfolder": args.model_subfolder,
            "processor_subfolder": args.processor_subfolder,
            "model_class": args.model_class,
            "train_jsonl": str(Path(args.train_jsonl).resolve()),
            "records": len(records),
            "global_step": global_step,
            "max_steps": max_steps,
            "requested_max_train_steps": requested_max_train_steps,
            "num_train_epochs": num_train_epochs,
            "prepared_batches_per_epoch": schedule.batches_per_epoch,
            "update_steps_per_epoch": schedule.update_steps_per_epoch,
            "world_size": accelerator.num_processes,
            "resume_from_checkpoint": str(resume_path) if resume_path is not None else None,
            "min_reward": args.min_reward,
            "reward_weighting": bool(args.reward_weighting),
            "min_reward_weight": args.min_reward_weight,
            "max_reward_weight": args.max_reward_weight,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        }
        with (output_dir / "proposer_training_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
