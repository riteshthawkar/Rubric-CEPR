from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    offload_models,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from qwen_edit_project.train.training_safety import (
    require_completed_steps,
    require_full_lora_key_coverage,
    resolve_training_schedule,
    resume_position,
    seed_for_process,
)

try:
    from diffusers.training_utils import _collate_lora_metadata
except ImportError:  # pragma: no cover - compatibility with older Diffusers

    def _collate_lora_metadata(_: dict[str, Any]) -> dict[str, Any]:
        return {}


if is_torch_npu_available():  # pragma: no cover - NPU-only runtime path
    torch.npu.config.allow_internal_format = False


logger = get_logger(__name__)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def retrieve_latents(encoder_output: Any, generator: torch.Generator | None = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from VAE encoder output")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Training manifest must be a JSON list or JSONL records: {path}")
    return records


def resolve_record_path(base_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_path / path


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        image = exif_transpose(handle)
        if image.mode != "RGB":
            image = image.convert("RGB")
        else:
            image = image.copy()
    return image


class EditManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        base_path: Path,
        image_key: str,
        condition_image_key: str,
        prompt_key: str,
        repeats: int,
    ) -> None:
        records = load_manifest(manifest_path)
        if repeats < 1:
            raise ValueError("--dataset_repeat must be >= 1")
        self.examples: list[dict[str, Any]] = []
        for record in records:
            prompt = record.get(prompt_key)
            target_image = record.get(image_key)
            source_image = record.get(condition_image_key)
            if not prompt or not target_image or not source_image:
                continue
            sample_weight = float(record.get("sample_weight", 1.0))
            if not math.isfinite(sample_weight) or sample_weight < 0:
                raise ValueError(f"Invalid sample_weight={sample_weight!r} in {manifest_path}")
            self.examples.append(
                {
                    "prompt": str(prompt),
                    "target_path": resolve_record_path(base_path, str(target_image)),
                    "source_path": resolve_record_path(base_path, str(source_image)),
                    "sample_weight": sample_weight,
                }
            )
        if not self.examples:
            raise ValueError(f"No trainable edit examples found in {manifest_path}")
        missing_paths = []
        for example in self.examples:
            for key in ("target_path", "source_path"):
                if not example[key].exists():
                    missing_paths.append(str(example[key]))
        if missing_paths:
            preview = ", ".join(missing_paths[:5])
            suffix = "" if len(missing_paths) <= 5 else f", ... ({len(missing_paths)} missing total)"
            raise FileNotFoundError(f"Training manifest references missing image paths: {preview}{suffix}")
        self.examples = self.examples * repeats

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class PreferenceManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        base_path: Path,
        chosen_image_key: str,
        rejected_image_key: str,
        condition_image_key: str,
        prompt_key: str,
        repeats: int,
    ) -> None:
        records = load_manifest(manifest_path)
        if repeats < 1:
            raise ValueError("--dataset_repeat must be >= 1")
        self.examples: list[dict[str, Any]] = []
        for record in records:
            prompt = record.get(prompt_key)
            chosen_image = record.get(chosen_image_key)
            rejected_image = record.get(rejected_image_key)
            source_image = record.get(condition_image_key)
            if not prompt or not chosen_image or not rejected_image or not source_image:
                continue
            sample_weight = float(record.get("sample_weight", 1.0))
            if not math.isfinite(sample_weight) or sample_weight < 0:
                raise ValueError(f"Invalid sample_weight={sample_weight!r} in {manifest_path}")
            preference_sft_weight = float(record.get("preference_sft_weight", math.nan))
            if not math.isfinite(preference_sft_weight):
                preference_sft_weight = math.nan
            elif preference_sft_weight < 0:
                raise ValueError(f"Invalid preference_sft_weight={preference_sft_weight!r} in {manifest_path}")
            self.examples.append(
                {
                    "prompt": str(prompt),
                    "chosen_path": resolve_record_path(base_path, str(chosen_image)),
                    "rejected_path": resolve_record_path(base_path, str(rejected_image)),
                    "source_path": resolve_record_path(base_path, str(source_image)),
                    "sample_weight": sample_weight,
                    "preference_sft_weight": preference_sft_weight,
                    "family": str(record.get("family", "")),
                    "preference_source": str(record.get("preference_source", "")),
                }
            )
        if not self.examples:
            raise ValueError(f"No trainable preference examples found in {manifest_path}")
        missing_paths = []
        for example in self.examples:
            for key in ("chosen_path", "rejected_path", "source_path"):
                if not example[key].exists():
                    missing_paths.append(str(example[key]))
        if missing_paths:
            preview = ", ".join(missing_paths[:5])
            suffix = "" if len(missing_paths) <= 5 else f", ... ({len(missing_paths)} missing total)"
            raise FileNotFoundError(f"Preference manifest references missing image paths: {preview}{suffix}")
        self.examples = self.examples * repeats

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_single_example(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return examples


@dataclass
class PreparedEditBatch:
    model_input: torch.Tensor
    image_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    img_shapes: list[list[tuple[int, int, int]]]
    txt_seq_lens: list[int] | None
    target_height: int
    target_width: int


def choose_training_sizes(args: argparse.Namespace, source_image: Image.Image) -> tuple[int, int, int, int, int, int]:
    if args.preserve_aspect_ratio:
        ratio = source_image.size[0] / source_image.size[1]
        target_width, target_height = calculate_dimensions(args.max_pixels, ratio)
        condition_width, condition_height = calculate_dimensions(args.condition_pixels, ratio)
        return target_width, target_height, target_width, target_height, condition_width, condition_height
    return (
        args.resolution,
        args.resolution,
        args.resolution,
        args.resolution,
        args.condition_resolution,
        args.condition_resolution,
    )


def encode_edit_example(
    args: argparse.Namespace,
    conditioning_pipe: QwenImageEditPlusPipeline,
    vae: AutoencoderKLQwenImage,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    example: dict[str, Any],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    latents_mean: torch.Tensor,
    latents_std_scale: torch.Tensor,
    generator: torch.Generator | None = None,
) -> PreparedEditBatch:
    source_image = load_rgb(example["source_path"])
    target_image = load_rgb(example["target_path"])
    target_width, target_height, vae_width, vae_height, condition_width, condition_height = choose_training_sizes(
        args, source_image
    )

    condition_image = conditioning_pipe.image_processor.resize(source_image, condition_height, condition_width)
    target_tensor = conditioning_pipe.image_processor.preprocess(target_image, target_height, target_width).unsqueeze(2)
    source_tensor = conditioning_pipe.image_processor.preprocess(source_image, vae_height, vae_width).unsqueeze(2)

    with torch.no_grad():
        with offload_models(text_encoder, device=accelerator.device, offload=args.offload):
            prompt_embeds, prompt_embeds_mask = conditioning_pipe.encode_prompt(
                image=[condition_image],
                prompt=[example["prompt"]],
                device=accelerator.device,
                max_sequence_length=args.max_sequence_length,
            )

        with offload_models(vae, device=accelerator.device, offload=args.offload):
            target_tensor = target_tensor.to(device=accelerator.device, dtype=vae.dtype)
            source_tensor = source_tensor.to(device=accelerator.device, dtype=vae.dtype)
            target_latents = retrieve_latents(
                vae.encode(target_tensor), generator=generator, sample_mode="sample"
            )
            source_latents = retrieve_latents(vae.encode(source_tensor), sample_mode="argmax")

    model_input = (target_latents - latents_mean) * latents_std_scale
    model_input = model_input.to(device=accelerator.device, dtype=weight_dtype)
    source_latents = (source_latents - latents_mean) * latents_std_scale
    source_latents = source_latents.to(device=accelerator.device, dtype=weight_dtype)

    image_latent_height, image_latent_width = source_latents.shape[3:]
    image_latents = QwenImageEditPlusPipeline._pack_latents(
        source_latents,
        batch_size=1,
        num_channels_latents=model_input.shape[1],
        height=image_latent_height,
        width=image_latent_width,
    )

    vae_scale_factor = conditioning_pipe.vae_scale_factor
    img_shapes = [
        [
            (1, target_height // vae_scale_factor // 2, target_width // vae_scale_factor // 2),
            (1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2),
        ]
    ]
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(device=accelerator.device)
        txt_seq_lens = [int(value) for value in prompt_embeds_mask.sum(dim=1).tolist()]
    else:
        txt_seq_lens = None

    return PreparedEditBatch(
        model_input=model_input,
        image_latents=image_latents,
        prompt_embeds=prompt_embeds.to(device=accelerator.device, dtype=weight_dtype),
        prompt_embeds_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        target_height=target_height,
        target_width=target_width,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen-Image-Edit LoRA with native Diffusers components.")
    parser.add_argument("--pretrained_model_name_or_path", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--chosen_image_key", default="chosen_image")
    parser.add_argument("--rejected_image_key", default="rejected_image")
    parser.add_argument("--condition_image_key", default="edit_image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--condition_resolution", type=int, default=384)
    parser.add_argument("--max_pixels", type=int, default=VAE_IMAGE_SIZE)
    parser.add_argument("--condition_pixels", type=int, default=CONDITION_IMAGE_SIZE)
    parser.add_argument("--preserve_aspect_ratio", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument(
        "--weighting_scheme",
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--scheduler_shift", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--training_objective",
        default="sft",
        choices=["sft", "pairwise_dpo", "pairwise_sdpo", "pairwise_linear_dpo", "pairwise_linear_sdpo"],
        help=(
            "Use standard target SFT, diffusion pairwise preference losses, or safeguarded "
            "pairwise preference updates that preserve the chosen branch."
        ),
    )
    parser.add_argument("--preference_beta", type=float, default=10.0)
    parser.add_argument("--preference_margin", type=float, default=0.0)
    parser.add_argument("--preference_sft_weight", type=float, default=0.0)
    parser.add_argument(
        "--preference_sdpo_epsilon",
        type=float,
        default=1.0e-12,
        help="Numerical epsilon for the safeguarded pairwise_sdpo rejected-gradient scale.",
    )
    parser.add_argument(
        "--preference_reference_mode",
        default="none",
        choices=["none", "initial_lora"],
        help=(
            "For pairwise_dpo, compare the current preference delta against a frozen copy of the "
            "initial LoRA delta. This gives a reference-subtracted diffusion preference objective "
            "without loading a second transformer."
        ),
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--lora_checkpoint", default=None)
    parser.add_argument(
        "--lora_reference_l2_weight",
        type=float,
        default=0.0,
        help=(
            "Optional trust-region penalty on the relative squared LoRA parameter drift from the "
            "initial trainable adapter state."
        ),
    )
    parser.add_argument(
        "--lora_reference_max_relative_delta",
        type=float,
        default=0.0,
        help=(
            "Optionally project LoRA parameters back to this maximum relative L2 distance from "
            "the initial trainable adapter state after each optimizer step. Disabled at 0."
        ),
    )
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--output_dir", default="outputs/checkpoints/Qwen-Image-Edit-2509_lora_diffusers")
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--upcast_before_saving", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.train_batch_size != 1:
        raise ValueError("Qwen edit LoRA training currently requires --train_batch_size=1.")
    if args.max_sequence_length > 1024:
        raise ValueError("--max_sequence_length cannot exceed 1024 for Qwen-Image-Edit.")
    if args.lora_reference_l2_weight < 0:
        raise ValueError("--lora_reference_l2_weight must be non-negative.")
    if args.lora_reference_max_relative_delta < 0:
        raise ValueError("--lora_reference_max_relative_delta must be non-negative.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be positive.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive when provided.")
    return args


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def load_lora_into_transformer(transformer: QwenImageTransformer2DModel, checkpoint: str) -> None:
    lora_state_dict = QwenImageEditPlusPipeline.lora_state_dict(checkpoint)
    transformer_state_dict = {
        key.replace("transformer.", ""): value
        for key, value in lora_state_dict.items()
        if key.startswith("transformer.")
    }
    if not transformer_state_dict:
        transformer_state_dict = dict(lora_state_dict)
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    expected_state_dict = get_peft_model_state_dict(transformer)
    coverage = require_full_lora_key_coverage(
        expected_state_dict.keys(),
        transformer_state_dict.keys(),
        checkpoint=checkpoint,
    )
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    if unexpected_keys:
        raise ValueError(f"Unexpected LoRA keys while loading '{checkpoint}': {unexpected_keys}")
    missing_lora_keys = [
        key for key in (getattr(incompatible_keys, "missing_keys", None) or []) if ".lora_" in key
    ]
    if missing_lora_keys:
        raise ValueError(f"Missing LoRA tensors while loading '{checkpoint}': {missing_lora_keys[:10]}")
    logger.info(
        "Loaded LoRA checkpoint %s with complete key coverage (%s/%s).",
        checkpoint,
        coverage.matched_count,
        coverage.expected_count,
    )


def clone_lora_state(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in get_peft_model_state_dict(transformer).items()}


def clone_trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters() if param.requires_grad}


def restore_lora_state(transformer: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> None:
    incompatible_keys = set_peft_model_state_dict(transformer, state, adapter_name="default")
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    if unexpected_keys:
        logger.warning("Unexpected LoRA keys while restoring %s state: %s", label, unexpected_keys)


def trainable_relative_l2(module: torch.nn.Module, reference_state: dict[str, torch.Tensor]) -> torch.Tensor:
    total_diff = None
    total_ref = None
    for name, param in module.named_parameters():
        if not param.requires_grad or name not in reference_state:
            continue
        ref = reference_state[name].to(device=param.device, dtype=param.dtype)
        diff = torch.sum((param.float() - ref.float()) ** 2)
        ref_norm = torch.sum(ref.float() ** 2)
        total_diff = diff if total_diff is None else total_diff + diff
        total_ref = ref_norm if total_ref is None else total_ref + ref_norm
    if total_diff is None or total_ref is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return total_diff / torch.clamp(total_ref, min=1.0e-12)


@torch.no_grad()
def project_trainable_state_to_relative_delta(
    module: torch.nn.Module,
    reference_state: dict[str, torch.Tensor],
    max_relative_delta: float,
) -> torch.Tensor:
    if max_relative_delta <= 0:
        return torch.zeros((), device=next(module.parameters()).device)
    relative_l2 = trainable_relative_l2(module, reference_state)
    relative_delta = torch.sqrt(relative_l2)
    if relative_delta <= max_relative_delta:
        return relative_delta.detach()
    scale = max_relative_delta / torch.clamp(relative_delta, min=1.0e-12)
    for name, param in module.named_parameters():
        if not param.requires_grad or name not in reference_state:
            continue
        ref = reference_state[name].to(device=param.device, dtype=param.dtype)
        param.copy_(ref + (param - ref) * scale.to(device=param.device, dtype=param.dtype))
    return relative_delta.detach()


def main() -> None:
    args = parse_args()

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("MPS does not support bf16 mixed precision. Use fp16 or no mixed precision.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    report_to = None if args.report_to in {"none", "None", ""} else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    is_pairwise_objective = args.training_objective in {
        "pairwise_dpo",
        "pairwise_sdpo",
        "pairwise_linear_dpo",
        "pairwise_linear_sdpo",
    }
    if is_pairwise_objective:
        train_dataset = PreferenceManifestDataset(
            manifest_path=Path(args.dataset_metadata_path),
            base_path=Path(args.dataset_base_path),
            chosen_image_key=args.chosen_image_key,
            rejected_image_key=args.rejected_image_key,
            condition_image_key=args.condition_image_key,
            prompt_key=args.prompt_key,
            repeats=args.dataset_repeat,
        )
    else:
        train_dataset = EditManifestDataset(
            manifest_path=Path(args.dataset_metadata_path),
            base_path=Path(args.dataset_base_path),
            image_key=args.image_key,
            condition_image_key=args.condition_image_key,
            prompt_key=args.prompt_key,
            repeats=args.dataset_repeat,
        )
    dataloader_generator = None
    if args.seed is not None and accelerator.num_processes > 1:
        # Every rank must construct the same shuffle order before Accelerate
        # shards it. Runtime stochasticity is reseeded per rank after prepare.
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_single_example,
        num_workers=args.dataloader_num_workers,
        generator=dataloader_generator,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="processor",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
        shift=args.scheduler_shift,
        local_files_only=args.local_files_only,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    if args.offload:
        vae.to(dtype=weight_dtype)
        text_encoder.to(dtype=weight_dtype)
    else:
        vae.to(device=accelerator.device, dtype=weight_dtype)
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
    transformer.to(device=accelerator.device, dtype=weight_dtype)

    conditioning_pipe = QwenImageEditPlusPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=None,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = [layer.strip() for layer in args.lora_layers.split(",") if layer.strip()]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.lora_checkpoint:
        load_lora_into_transformer(transformer, args.lora_checkpoint)

    def save_model_hook(models: list[torch.nn.Module], weights: list[Any], output_dir: str) -> None:
        if not accelerator.is_main_process:
            return
        transformer_lora_layers_to_save = None
        modules_to_save = {}
        for model in models:
            unwrapped = unwrap_model(accelerator, model)
            if isinstance(unwrapped, type(unwrap_model(accelerator, transformer))):
                transformer_lora_layers_to_save = get_peft_model_state_dict(unwrapped)
                modules_to_save["transformer"] = unwrapped
            else:
                raise ValueError(f"Unexpected model during save: {model.__class__}")
            if weights:
                weights.pop()
        QwenImageEditPlusPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
            **_collate_lora_metadata(modules_to_save),
        )

    def load_model_hook(models: list[torch.nn.Module], input_dir: str) -> None:
        transformer_to_load = None
        while models:
            model = models.pop()
            unwrapped = unwrap_model(accelerator, model)
            if isinstance(unwrapped, type(unwrap_model(accelerator, transformer))):
                transformer_to_load = unwrapped
            else:
                raise ValueError(f"Unexpected model during load: {model.__class__}")
        if transformer_to_load is None:
            return
        load_lora_into_transformer(transformer_to_load, input_dir)
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_to_load], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(
            "No trainable LoRA parameters were created. Check --lora_layers against the Qwen transformer module names."
        )
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    logger.info("Trainable LoRA parameters = %s", trainable_parameter_count)

    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError("Install bitsandbytes to use --use_8bit_adam.") from exc
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            [{"params": trainable_parameters}],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        raise ValueError(f"Unsupported optimizer for Diffusers Qwen edit LoRA training: {args.optimizer}")

    requested_max_train_steps = args.max_train_steps
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    schedule = resolve_training_schedule(
        batches_per_epoch=len(train_dataloader),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        requested_num_train_epochs=args.num_train_epochs,
        requested_max_train_steps=requested_max_train_steps,
    )
    args.max_train_steps = schedule.max_train_steps
    args.num_train_epochs = schedule.num_train_epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    if args.seed is not None and accelerator.num_processes > 1:
        # Adapter initialization and shuffle order use the common base seed;
        # stochastic latents, diffusion noise, and dropout are rank-specific.
        set_seed(seed_for_process(args.seed, accelerator.process_index))
    reference_lora_state: dict[str, torch.Tensor] | None = None
    if is_pairwise_objective and args.preference_reference_mode == "initial_lora":
        reference_lora_state = clone_lora_state(unwrap_model(accelerator, transformer))
        logger.info("Captured initial LoRA state for reference-subtracted pairwise preference loss.")
    trainable_reference_state: dict[str, torch.Tensor] | None = None
    if args.lora_reference_l2_weight > 0 or args.lora_reference_max_relative_delta > 0:
        trainable_reference_state = clone_trainable_state(unwrap_model(accelerator, transformer))
        logger.info(
            "Captured initial trainable LoRA state for trust-region regularization "
            "(l2_weight=%s, max_relative_delta=%s).",
            args.lora_reference_l2_weight,
            args.lora_reference_max_relative_delta,
        )

    vae_scale_factor = conditioning_pipe.vae_scale_factor
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    latents_std_scale = (
        1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    )

    if accelerator.is_main_process:
        tracker_config = vars(args).copy()
        tracker_config.update(
            {
                "requested_max_train_steps": requested_max_train_steps,
                "prepared_batches_per_epoch": schedule.batches_per_epoch,
                "update_steps_per_epoch": schedule.update_steps_per_epoch,
                "world_size": accelerator.num_processes,
            }
        )
        accelerator.init_trackers("qwen-edit-diffusers-lora", config=tracker_config)
        with open(Path(args.output_dir) / "training_args.json", "w", encoding="utf-8") as handle:
            json.dump(tracker_config, handle, indent=2, ensure_ascii=True)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running Diffusers-native Qwen edit LoRA training *****")
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  Num batches each epoch = %s", len(train_dataloader))
    logger.info("  Num epochs = %s", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %s", args.train_batch_size)
    logger.info("  Total train batch size = %s", total_batch_size)
    logger.info("  Total optimization steps = %s", args.max_train_steps)

    global_step = 0
    first_epoch = 0
    resume_step = 0
    resume_path: Path | None = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = []
            for name in os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []:
                if not name.startswith("checkpoint-"):
                    continue
                try:
                    dirs.append((int(name.split("-")[-1]), name))
                except ValueError:
                    continue
            dirs = sorted(dirs, key=lambda item: item[0])
            resume_path = Path(args.output_dir) / dirs[-1][1] if dirs else None
        else:
            requested_path = Path(args.resume_from_checkpoint).expanduser()
            resume_path = requested_path if requested_path.is_absolute() else Path(args.output_dir) / requested_path
        if resume_path is None or not resume_path.is_dir():
            raise FileNotFoundError(
                f"Requested resume checkpoint '{args.resume_from_checkpoint}' does not exist as a directory."
            )
        try:
            global_step = int(resume_path.name.split("-")[-1])
        except ValueError as exc:
            raise ValueError(
                f"Resume checkpoint directory must end in an integer step: {resume_path}"
            ) from exc
        if global_step > args.max_train_steps:
            raise ValueError(
                f"Resume checkpoint step {global_step} exceeds requested max_train_steps={args.max_train_steps}."
            )
        accelerator.print(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(str(resume_path))
        position = resume_position(
            global_step=global_step,
            batches_per_epoch=len(train_dataloader),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        first_epoch = position.first_epoch
        resume_step = position.batches_to_skip

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps: torch.Tensor, n_dim: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    guidance_embeds = bool(getattr(unwrap_model(accelerator, transformer).config, "guidance_embeds", False))
    if guidance_embeds and args.guidance_scale is None:
        raise ValueError("--guidance_scale is required for guidance-distilled Qwen transformers.")

    def compute_flow_matching_loss_vector(
        prepared: PreparedEditBatch,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        model_input = prepared.model_input
        bsz = model_input.shape[0]
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
            noisy_model_input.permute(0, 2, 1, 3, 4).contiguous(),
            batch_size=bsz,
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[3],
            width=model_input.shape[4],
        )
        latent_model_input = torch.cat([packed_noisy_model_input, prepared.image_latents], dim=1)
        guidance = None
        if guidance_embeds:
            guidance = torch.full([bsz], args.guidance_scale, device=model_input.device, dtype=torch.float32)

        cache_context = getattr(transformer, "cache_context", None)
        context = cache_context("cond") if cache_context is not None else nullcontext()
        with context:
            model_pred = transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prepared.prompt_embeds,
                encoder_hidden_states_mask=prepared.prompt_embeds_mask,
                timestep=timesteps / 1000,
                img_shapes=prepared.img_shapes,
                txt_seq_lens=prepared.txt_seq_lens,
                guidance=guidance,
                return_dict=False,
            )[0]
        model_pred = model_pred[:, : packed_noisy_model_input.shape[1]]
        model_pred = QwenImageEditPlusPipeline._unpack_latents(
            model_pred,
            prepared.target_height,
            prepared.target_width,
            vae_scale_factor,
        )
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        return torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        )

    def sample_noise_timestep(model_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        return noise, timesteps, sigmas

    def compute_sdpo_rejected_scale(
        chosen_loss: torch.Tensor,
        rejected_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_grads = torch.autograd.grad(
            chosen_loss,
            trainable_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        rejected_grads = torch.autograd.grad(
            rejected_loss,
            trainable_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        chosen_norm_sq = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        chosen_rejected_dot = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        for chosen_grad, rejected_grad in zip(chosen_grads, rejected_grads, strict=True):
            if chosen_grad is None or rejected_grad is None:
                continue
            chosen_grad = chosen_grad.detach().float()
            rejected_grad = rejected_grad.detach().float()
            chosen_norm_sq = chosen_norm_sq + torch.sum(chosen_grad * chosen_grad)
            chosen_rejected_dot = chosen_rejected_dot + torch.sum(chosen_grad * rejected_grad)
        if accelerator.num_processes > 1:
            chosen_norm_sq = accelerator.reduce(chosen_norm_sq, reduction="sum")
            chosen_rejected_dot = accelerator.reduce(chosen_rejected_dot, reduction="sum")
        scale_if_conflicting = torch.clamp(
            chosen_norm_sq / (chosen_rejected_dot + args.preference_sdpo_epsilon),
            min=0.0,
            max=1.0,
        )
        rejected_scale = torch.where(
            chosen_rejected_dot > 0,
            scale_if_conflicting,
            torch.ones_like(scale_if_conflicting),
        )
        return rejected_scale.detach(), chosen_norm_sq.detach(), chosen_rejected_dot.detach()

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        active_dataloader = train_dataloader
        if resume_step:
            accelerator.print(f"Skipping {resume_step} already-processed dataloader batch(es) in epoch {epoch}.")
            if hasattr(accelerator, "skip_first_batches"):
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = itertools.islice(train_dataloader, resume_step, None)
            resume_step = 0
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(transformer):
                if len(batch) != 1:
                    raise ValueError("Diffusers edit training currently expects exactly one example per batch.")
                example = batch[0]
                sample_weight = float(example.get("sample_weight", 1.0))
                example_preference_sft_weight = args.preference_sft_weight
                if is_pairwise_objective:
                    raw_preference_sft_weight = float(example.get("preference_sft_weight", math.nan))
                    if math.isfinite(raw_preference_sft_weight):
                        example_preference_sft_weight = raw_preference_sft_weight
                logs_extra: dict[str, float] = {}
                if is_pairwise_objective:
                    chosen_prepared = encode_edit_example(
                        args=args,
                        conditioning_pipe=conditioning_pipe,
                        vae=vae,
                        text_encoder=text_encoder,
                        example={
                            "prompt": example["prompt"],
                            "target_path": example["chosen_path"],
                            "source_path": example["source_path"],
                        },
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        latents_mean=latents_mean,
                        latents_std_scale=latents_std_scale,
                    )
                    rejected_prepared = encode_edit_example(
                        args=args,
                        conditioning_pipe=conditioning_pipe,
                        vae=vae,
                        text_encoder=text_encoder,
                        example={
                            "prompt": example["prompt"],
                            "target_path": example["rejected_path"],
                            "source_path": example["source_path"],
                        },
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        latents_mean=latents_mean,
                        latents_std_scale=latents_std_scale,
                    )
                    if chosen_prepared.model_input.shape != rejected_prepared.model_input.shape:
                        raise ValueError(
                            "Chosen and rejected preference latents must have matching shapes; "
                            f"got {chosen_prepared.model_input.shape} and {rejected_prepared.model_input.shape}."
                        )
                    noise, timesteps, sigmas = sample_noise_timestep(chosen_prepared.model_input)
                    reference_delta = None
                    if reference_lora_state is not None:
                        unwrapped_transformer = unwrap_model(accelerator, transformer)
                        current_lora_state = clone_lora_state(unwrapped_transformer)
                        was_training = transformer.training
                        restore_lora_state(unwrapped_transformer, reference_lora_state, "reference")
                        transformer.eval()
                        with torch.no_grad():
                            reference_chosen_loss_vector = compute_flow_matching_loss_vector(
                                chosen_prepared, noise, timesteps, sigmas
                            )
                            reference_rejected_loss_vector = compute_flow_matching_loss_vector(
                                rejected_prepared, noise, timesteps, sigmas
                            )
                        restore_lora_state(unwrapped_transformer, current_lora_state, "current")
                        if was_training:
                            transformer.train()
                        reference_delta = reference_rejected_loss_vector - reference_chosen_loss_vector
                        del current_lora_state

                    chosen_loss_vector = compute_flow_matching_loss_vector(chosen_prepared, noise, timesteps, sigmas)
                    rejected_loss_vector = compute_flow_matching_loss_vector(
                        rejected_prepared, noise, timesteps, sigmas
                    )
                    preference_delta = rejected_loss_vector - chosen_loss_vector
                    optimized_delta = (
                        preference_delta - reference_delta if reference_delta is not None else preference_delta
                    )
                    preference_logits = args.preference_beta * (optimized_delta - args.preference_margin)
                    preference_loss = F.softplus(-preference_logits).mean()
                    chosen_sft_loss = chosen_loss_vector.mean()
                    rejected_loss = rejected_loss_vector.mean()
                    if args.training_objective in {"pairwise_sdpo", "pairwise_linear_sdpo"}:
                        rejected_scale, chosen_grad_norm_sq, chosen_rejected_grad_dot = compute_sdpo_rejected_scale(
                            chosen_sft_loss,
                            rejected_loss,
                        )
                        if args.training_objective == "pairwise_linear_sdpo":
                            preference_weight = torch.full(
                                (),
                                args.preference_beta,
                                device=accelerator.device,
                                dtype=chosen_sft_loss.dtype,
                            )
                        else:
                            preference_weight = (
                                args.preference_beta * torch.sigmoid(-preference_logits)
                            ).detach().mean()
                        preference_surrogate = preference_weight * (chosen_sft_loss - rejected_scale * rejected_loss)
                        loss = (
                            preference_surrogate + example_preference_sft_weight * chosen_sft_loss
                        ) * sample_weight
                        logs_extra = {
                            "chosen_mse": chosen_sft_loss.detach().item(),
                            "rejected_mse": rejected_loss.detach().item(),
                            "preference_delta": preference_delta.mean().detach().item(),
                            "optimized_preference_delta": optimized_delta.mean().detach().item(),
                            "preference_loss": preference_loss.detach().item(),
                            "preference_weight": preference_weight.detach().item(),
                            "preference_sft_weight": float(example_preference_sft_weight),
                            "sdpo_rejected_scale": rejected_scale.detach().item(),
                            "sdpo_chosen_grad_norm_sq": chosen_grad_norm_sq.detach().item(),
                            "sdpo_chosen_rejected_grad_dot": chosen_rejected_grad_dot.detach().item(),
                        }
                    elif args.training_objective == "pairwise_linear_dpo":
                        linear_preference_loss = -args.preference_beta * (
                            optimized_delta - args.preference_margin
                        ).mean()
                        loss = (
                            linear_preference_loss + example_preference_sft_weight * chosen_sft_loss
                        ) * sample_weight
                        logs_extra = {
                            "chosen_mse": chosen_sft_loss.detach().item(),
                            "rejected_mse": rejected_loss.detach().item(),
                            "preference_delta": preference_delta.mean().detach().item(),
                            "optimized_preference_delta": optimized_delta.mean().detach().item(),
                            "preference_loss": preference_loss.detach().item(),
                            "linear_preference_loss": linear_preference_loss.detach().item(),
                            "preference_sft_weight": float(example_preference_sft_weight),
                        }
                    else:
                        loss = (
                            preference_loss + example_preference_sft_weight * chosen_sft_loss
                        ) * sample_weight
                        logs_extra = {
                            "chosen_mse": chosen_sft_loss.detach().item(),
                            "rejected_mse": rejected_loss.detach().item(),
                            "preference_delta": preference_delta.mean().detach().item(),
                            "optimized_preference_delta": optimized_delta.mean().detach().item(),
                            "preference_loss": preference_loss.detach().item(),
                            "preference_sft_weight": float(example_preference_sft_weight),
                        }
                    if reference_delta is not None:
                        logs_extra["reference_preference_delta"] = reference_delta.mean().detach().item()
                else:
                    prepared = encode_edit_example(
                        args=args,
                        conditioning_pipe=conditioning_pipe,
                        vae=vae,
                        text_encoder=text_encoder,
                        example=example,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        latents_mean=latents_mean,
                        latents_std_scale=latents_std_scale,
                    )
                    noise, timesteps, sigmas = sample_noise_timestep(prepared.model_input)
                    loss = compute_flow_matching_loss_vector(prepared, noise, timesteps, sigmas).mean() * sample_weight

                if trainable_reference_state is not None and args.lora_reference_l2_weight > 0:
                    lora_reference_l2 = trainable_relative_l2(
                        unwrap_model(accelerator, transformer),
                        trainable_reference_state,
                    )
                    loss = loss + args.lora_reference_l2_weight * lora_reference_l2
                    logs_extra["lora_reference_l2"] = lora_reference_l2.detach().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                if trainable_reference_state is not None and args.lora_reference_max_relative_delta > 0:
                    projected_relative_delta = project_trainable_state_to_relative_delta(
                        unwrap_model(accelerator, transformer),
                        trainable_reference_state,
                        args.lora_reference_max_relative_delta,
                    )
                    logs_extra["lora_reference_relative_delta_pre_projection"] = projected_relative_delta.item()
                lr_scheduler.step()
                optimizer.zero_grad()
                free_memory()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = []
                        for name in os.listdir(args.output_dir):
                            if not name.startswith("checkpoint-"):
                                continue
                            try:
                                checkpoints.append((int(name.split("-")[-1]), name))
                            except ValueError:
                                continue
                        checkpoints = [name for _, name in sorted(checkpoints, key=lambda item: item[0])]
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint))
                    # Distributed optimizer/RNG state is rank-specific.  Every
                    # process must participate in save_state; only directory
                    # retention is a main-process responsibility.
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info("Saved state to %s", save_path)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "sample_weight": sample_weight,
            }
            logs.update(logs_extra)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    progress_bar.close()
    accelerator.wait_for_everyone()
    require_completed_steps(
        global_step=global_step,
        requested_steps=args.max_train_steps,
        trainer_name="Diffusers Qwen edit LoRA trainer",
    )
    if accelerator.is_main_process:
        unwrapped_transformer = unwrap_model(accelerator, transformer)
        if args.upcast_before_saving:
            unwrapped_transformer.to(torch.float32)
        else:
            unwrapped_transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(unwrapped_transformer)
        QwenImageEditPlusPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata({"transformer": unwrapped_transformer}),
        )
        completion = {
            "schema_version": 1,
            "status": "complete",
            "global_step": global_step,
            "max_train_steps": args.max_train_steps,
            "requested_max_train_steps": requested_max_train_steps,
            "num_train_epochs": args.num_train_epochs,
            "prepared_batches_per_epoch": schedule.batches_per_epoch,
            "update_steps_per_epoch": schedule.update_steps_per_epoch,
            "world_size": accelerator.num_processes,
            "seed": args.seed,
            "resume_from_checkpoint": str(resume_path) if resume_path is not None else None,
            "lora_tensor_count": len(transformer_lora_layers),
            "weights": "pytorch_lora_weights.safetensors",
        }
        with (Path(args.output_dir) / "training_completion.json").open("w", encoding="utf-8") as handle:
            json.dump(completion, handle, indent=2, ensure_ascii=True)
        logger.info("Saved final Diffusers LoRA weights to %s", args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
