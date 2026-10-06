from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from shutil import which
from typing import Any

from qwen_edit_project.utils.commands import run_and_tee, shell_join
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def build_accelerate_command(executable: Any | None = None) -> list[str]:
    value = str(executable or "accelerate")
    if value in {"accelerate", "accelerate.exe"}:
        env_executable = Path(sys.executable).with_name(value)
        if env_executable.exists():
            return [str(env_executable)]
        resolved = which(value)
        if resolved:
            return [resolved]
    return [value]


def _build_diffusers_train_command(config: dict[str, Any]) -> tuple[list[str], Path]:
    runtime = config["runtime"]
    mode = config["mode"]
    if mode != "lora":
        raise ValueError(f"Diffusers-native Qwen edit training only supports LoRA mode, got: {mode}")

    working_dir = resolve_path(runtime.get("working_dir", "."))
    if working_dir is None:
        raise ValueError("runtime.working_dir is required")
    train_script = resolve_path(runtime.get("train_script", "src/qwen_edit_project/train/diffusers_qwen_edit_lora.py"))
    if train_script is None:
        raise ValueError("runtime.train_script could not be resolved")

    command = [*build_accelerate_command(runtime.get("accelerate_executable")), "launch"]
    accelerate_config = runtime.get("accelerate_config_file")
    if accelerate_config:
        accelerate_config_path = Path(accelerate_config)
        if not accelerate_config_path.is_absolute():
            accelerate_config_path = working_dir / accelerate_config_path
        command.extend(["--config_file", str(accelerate_config_path)])
    command.append(str(train_script))

    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    output = config["output"]
    lora = config["lora"]

    _append_flag(command, "--pretrained_model_name_or_path", model["pretrained_model_name_or_path"])
    _append_flag(command, "--revision", model.get("revision"))
    _append_flag(command, "--variant", model.get("variant"))
    _append_flag(command, "--local_files_only", model.get("local_files_only", False))

    _append_flag(command, "--dataset_base_path", resolve_path(dataset["dataset_base_path"]))
    _append_flag(command, "--dataset_metadata_path", resolve_path(dataset["dataset_metadata_path"]))
    _append_flag(command, "--image_key", dataset.get("image_key", "image"))
    _append_flag(command, "--chosen_image_key", dataset.get("chosen_image_key"))
    _append_flag(command, "--rejected_image_key", dataset.get("rejected_image_key"))
    _append_flag(command, "--condition_image_key", dataset.get("condition_image_key", "edit_image"))
    _append_flag(command, "--prompt_key", dataset.get("prompt_key", "prompt"))
    _append_flag(command, "--dataset_repeat", dataset.get("dataset_repeat"))
    _append_flag(command, "--dataloader_num_workers", dataset.get("dataset_num_workers"))
    _append_flag(command, "--resolution", training.get("resolution"))
    _append_flag(command, "--condition_resolution", training.get("condition_resolution"))
    _append_flag(command, "--max_pixels", training.get("max_pixels"))
    _append_flag(command, "--condition_pixels", training.get("condition_pixels"))
    _append_flag(command, "--preserve_aspect_ratio", training.get("preserve_aspect_ratio", False))

    _append_flag(command, "--train_batch_size", training.get("train_batch_size"))
    _append_flag(command, "--num_train_epochs", training.get("num_epochs"))
    _append_flag(command, "--max_train_steps", training.get("max_train_steps"))
    _append_flag(command, "--checkpointing_steps", training.get("checkpointing_steps"))
    _append_flag(command, "--checkpoints_total_limit", training.get("checkpoints_total_limit"))
    _append_flag(command, "--resume_from_checkpoint", training.get("resume_from_checkpoint"))
    _append_flag(command, "--gradient_accumulation_steps", training.get("gradient_accumulation_steps"))
    _append_flag(command, "--gradient_checkpointing", training.get("use_gradient_checkpointing", False))
    _append_flag(command, "--learning_rate", training.get("learning_rate"))
    _append_flag(command, "--scale_lr", training.get("scale_lr", False))
    _append_flag(command, "--lr_scheduler", training.get("lr_scheduler"))
    _append_flag(command, "--lr_warmup_steps", training.get("lr_warmup_steps"))
    _append_flag(command, "--lr_num_cycles", training.get("lr_num_cycles"))
    _append_flag(command, "--lr_power", training.get("lr_power"))
    _append_flag(command, "--weighting_scheme", training.get("weighting_scheme"))
    _append_flag(command, "--logit_mean", training.get("logit_mean"))
    _append_flag(command, "--logit_std", training.get("logit_std"))
    _append_flag(command, "--mode_scale", training.get("mode_scale"))
    _append_flag(command, "--max_sequence_length", training.get("max_sequence_length"))
    _append_flag(command, "--scheduler_shift", training.get("scheduler_shift"))
    _append_flag(command, "--guidance_scale", training.get("guidance_scale"))
    _append_flag(command, "--training_objective", training.get("training_objective"))
    _append_flag(command, "--preference_beta", training.get("preference_beta"))
    _append_flag(command, "--preference_margin", training.get("preference_margin"))
    _append_flag(command, "--preference_sft_weight", training.get("preference_sft_weight"))
    _append_flag(command, "--preference_sdpo_epsilon", training.get("preference_sdpo_epsilon"))
    _append_flag(command, "--preference_reference_mode", training.get("preference_reference_mode"))
    _append_flag(command, "--allow_tf32", training.get("allow_tf32", False))
    _append_flag(command, "--mixed_precision", training.get("mixed_precision"))
    _append_flag(command, "--offload", training.get("offload", False))
    _append_flag(command, "--optimizer", training.get("optimizer"))
    _append_flag(command, "--use_8bit_adam", training.get("use_8bit_adam", False))
    _append_flag(command, "--adam_beta1", training.get("adam_beta1"))
    _append_flag(command, "--adam_beta2", training.get("adam_beta2"))
    _append_flag(command, "--adam_weight_decay", training.get("adam_weight_decay"))
    _append_flag(command, "--adam_epsilon", training.get("adam_epsilon"))
    _append_flag(command, "--max_grad_norm", training.get("max_grad_norm"))
    _append_flag(command, "--report_to", training.get("report_to"))
    _append_flag(command, "--seed", training.get("seed"))
    _append_flag(command, "--upcast_before_saving", training.get("upcast_before_saving", False))

    _append_flag(command, "--rank", lora.get("lora_rank", lora.get("rank")))
    _append_flag(command, "--lora_alpha", lora.get("lora_alpha", lora.get("alpha")))
    _append_flag(command, "--lora_dropout", lora.get("lora_dropout", lora.get("dropout")))
    _append_flag(command, "--lora_layers", lora.get("lora_target_modules", lora.get("target_modules")))
    _append_flag(command, "--lora_checkpoint", lora.get("lora_checkpoint") or lora.get("checkpoint_path"))
    _append_flag(command, "--lora_reference_l2_weight", lora.get("lora_reference_l2_weight"))
    _append_flag(command, "--lora_reference_max_relative_delta", lora.get("lora_reference_max_relative_delta"))

    _append_flag(command, "--output_dir", resolve_path(output["output_path"]))
    _append_flag(command, "--logging_dir", output.get("logging_dir", "logs"))

    if config.get("resume", {}).get("enabled"):
        for value in config["resume"].get("extra_args", []):
            command.append(str(value))

    for value in config.get("extra_args", []):
        command.append(str(value))

    return command, working_dir


def build_train_command(config: dict[str, Any]) -> tuple[list[str], Path]:
    runtime = config["runtime"]
    backend = str(runtime.get("backend", runtime.get("training_backend", "diffsynth")))
    if backend in {"diffusers", "official_diffusers", "diffusers_native"}:
        return _build_diffusers_train_command(config)

    mode = config["mode"]
    working_dir = resolve_path(runtime["working_dir"])
    if working_dir is None:
        raise ValueError("runtime.working_dir is required")
    train_script = working_dir / "examples/qwen_image/model_training/train.py"
    command = [*build_accelerate_command(runtime.get("accelerate_executable")), "launch"]
    accelerate_config = runtime.get("accelerate_config_file")
    if accelerate_config:
        command.extend(["--config_file", str(working_dir / accelerate_config)])
    command.append(str(train_script))

    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    output = config["output"]

    _append_flag(command, "--dataset_base_path", resolve_path(dataset["dataset_base_path"]))
    _append_flag(command, "--dataset_metadata_path", resolve_path(dataset["dataset_metadata_path"]))
    _append_flag(command, "--data_file_keys", dataset["data_file_keys"])
    _append_flag(command, "--extra_inputs", dataset.get("extra_inputs"))
    _append_flag(command, "--max_pixels", dataset.get("max_pixels"))
    _append_flag(command, "--dataset_repeat", dataset.get("dataset_repeat"))
    _append_flag(command, "--dataset_num_workers", dataset.get("dataset_num_workers"))
    _append_flag(command, "--height", training.get("height"))
    _append_flag(command, "--width", training.get("width"))
    _append_flag(command, "--gradient_accumulation_steps", training.get("gradient_accumulation_steps"))

    _append_flag(command, "--model_id_with_origin_paths", model["model_id_with_origin_paths"])
    _append_flag(command, "--tokenizer_path", model.get("tokenizer_path"))
    _append_flag(command, "--processor_path", model.get("processor_path"))
    _append_flag(command, "--zero_cond_t", model.get("zero_cond_t", False))
    _append_flag(command, "--remove_prefix_in_ckpt", model.get("remove_prefix_in_ckpt"))

    _append_flag(command, "--learning_rate", training.get("learning_rate"))
    _append_flag(command, "--num_epochs", training.get("num_epochs"))
    _append_flag(command, "--use_gradient_checkpointing", training.get("use_gradient_checkpointing", False))
    _append_flag(command, "--find_unused_parameters", training.get("find_unused_parameters", False))
    _append_flag(command, "--output_path", resolve_path(output["output_path"]))

    if mode == "lora":
        lora = config["lora"]
        _append_flag(command, "--lora_base_model", lora["lora_base_model"])
        _append_flag(command, "--lora_target_modules", lora["lora_target_modules"])
        _append_flag(command, "--lora_rank", lora["lora_rank"])
        _append_flag(command, "--lora_checkpoint", lora.get("lora_checkpoint"))
        _append_flag(command, "--preset_lora_path", lora.get("preset_lora_path"))
        _append_flag(command, "--preset_lora_model", lora.get("preset_lora_model"))
    elif mode == "full":
        _append_flag(command, "--trainable_models", training.get("trainable_models"))
    else:
        raise ValueError(f"Unsupported train mode: {mode}")

    if config.get("resume", {}).get("enabled"):
        for value in config["resume"].get("extra_args", []):
            command.append(str(value))

    for value in config.get("extra_args", []):
        command.append(str(value))

    return command, working_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DiffSynth training from YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="Override config using dotted.key=value")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    command, working_dir = build_train_command(config)
    output = config["output"]
    command_file = resolve_path(output["command_file"])
    log_dir = ensure_dir(resolve_path(output["log_dir"]))
    if command_file is None:
        raise ValueError("output.command_file is required")
    ensure_dir(command_file.parent)
    command_file.write_text(shell_join(command) + "\n", encoding="utf-8")

    timestamp = utc_timestamp()
    log_path = log_dir / f"{config['name']}_{timestamp}.log"
    metadata_path = log_dir / f"{config['name']}_{timestamp}.json"
    metadata = base_run_metadata()
    metadata.update(
        {
            "config_path": config["_config_path"],
            "command": command,
            "command_shell": shell_join(command),
            "working_dir": str(working_dir),
            "log_path": str(log_path),
            "mode": config["mode"],
            "output_path": str(resolve_path(output["output_path"])),
        }
    )
    save_json(metadata, metadata_path)

    print(shell_join(command))
    if args.dry_run or config["runtime"].get("dry_run", False):
        print(f"Dry run. Command file written to {command_file}")
        return

    return_code = run_and_tee(command, cwd=working_dir, log_path=log_path)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
