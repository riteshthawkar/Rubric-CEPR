"""Release commands; all model/scoring imports happen after scheduler checks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from .checks import (
    ROOT, check_code, check_completion, check_data, check_disjointness,
    digest, environment_report, identity, safe_relative_path,
)
from .runtime import child_environment, require_gpu_step


def training_command(data_root: Path, output: Path) -> list[str]:
    template = json.loads((ROOT / "configs/train/extraction_v1.json").read_text())["command_template"]
    values = {"python": sys.executable, "root": str(ROOT), "data_root": str(data_root.resolve()), "output": str(output.resolve())}
    return [token.format(**values) for token in template]


def install_artifacts(bundle: Path, data_root: Path) -> dict:
    check_code()
    if digest(bundle) != identity()["bundle_sha256"]:
        raise ValueError("Input bundle hash differs from the verified extraction bundle")
    if data_root.exists():
        raise FileExistsError("Use a new data root; artifact installation never overwrites a directory")
    inventory = json.loads((ROOT / "reproducibility/manifests/extraction_v1_artifacts.json").read_text())
    expected = {item["path"]: item for item in inventory["artifacts"]}
    prefix = identity()["bundle_member_prefix"]
    # Validate the entire archive topology before creating a destination.
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        by_name = {}
        for member in members:
            if member.name in by_name or not member.isfile():
                raise ValueError("Bundle contains duplicate names or non-file members")
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("Bundle contains an unsafe path")
            by_name[member.name] = member
        for name, item in expected.items():
            member = by_name.get(prefix + name)
            if member is None or member.size != item["bytes"]:
                raise ValueError(f"Bundle lacks an expected artifact: {name}")
        data_root.mkdir(parents=True)
        for name, item in expected.items():
            destination = safe_relative_path(data_root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle, archive.extractfile(by_name[prefix + name]) as source:
                shutil.copyfileobj(source, handle)
            if digest(destination) != item["sha256"]:
                raise ValueError(f"Installed artifact differs: {name}")
    return check_data(data_root)


def delegate(module: str, arguments: list[str]) -> None:
    saved = sys.argv
    try:
        sys.argv = [module, *arguments]
        importlib.import_module(module).main()
    finally:
        sys.argv = saved


def patch_inference_processor() -> None:
    """Use the same official processor components as the training compatibility shim."""
    from qwen_edit_project.train.diffusers_qwen_edit_lora_reproduction_v1 import ProcessorFolderCompatibilityLoader
    from qwen_edit_project.utils import qwen_pipeline

    def load(pipeline_cls, model_id, dtype, local_files_only, revision=None):
        processor = ProcessorFolderCompatibilityLoader.from_pretrained(
            model_id, subfolder="processor", revision=revision, local_files_only=local_files_only,
        )
        return pipeline_cls.from_pretrained(
            model_id, torch_dtype=dtype, processor=processor,
            local_files_only=local_files_only, revision=revision,
        )
    qwen_pipeline._from_pretrained_with_dtype = load


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rubric-CEPR: image editing, self-distillation and benchmark evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="CPU-only code, data and provenance checks")
    check.add_argument("--data-root", type=Path)
    check.add_argument("--benchmark-json", type=Path)
    check.add_argument("--benchmark-images", type=Path)
    check.add_argument("--checkpoint", type=Path)
    check.add_argument("--training-output", type=Path)
    check.add_argument("--environment", action="store_true")
    check.add_argument("--strict-environment", action="store_true")
    install = commands.add_parser("install-artifacts", help="Install the exact 128 images from the verified companion bundle")
    install.add_argument("--bundle", required=True, type=Path)
    install.add_argument("--data-root", required=True, type=Path)
    train = commands.add_parser("train", help="Train the fixed 400-step rank-16 extraction recipe")
    train.add_argument("--data-root", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--benchmark-json", type=Path, default=ROOT / "data/processed/benchmark/imgedit/basic_edit.json")
    train.add_argument("--benchmark-images", type=Path, default=ROOT / "data/processed/benchmark/imgedit/original_images")
    train.add_argument("--dry-run", action="store_true", help="Print the exact command; does not check missing external data or load a model")
    infer = commands.add_parser("infer", help="Edit one image with the reference or retrained adapter")
    infer.add_argument("--image", required=True, type=Path)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--allow-reconstructed-checkpoint", action="store_true")
    infer.add_argument("--dry-run", action="store_true")
    for name in ("export", "score"):
        sub = commands.add_parser(name, help=f"{name.title()} a benchmark using the release rerun protocol")
        sub.add_argument("--benchmark", required=True, choices=["imgedit", "gedit", "complex_edit"])
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--set", action="append", default=[])
        sub.add_argument("--dry-run", action="store_true")
    framework = commands.add_parser("framework", help="Run the Planner–Editor–Critic reference framework")
    framework.add_argument("--config", type=Path, default=ROOT / "configs/self_evolve/qwen_edit_2509_internal_cepr_rubric_trainable_proposer.yaml")
    framework.add_argument("--dry-run", action="store_true")
    framework.add_argument("--set", action="append", default=[])
    commands.add_parser("results", help="Display recorded benchmark results; no API calls")
    args = parser.parse_args(argv)
    # Caller-relative paths must be resolved before moving into the checkout.
    for field, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, field, value.expanduser().resolve())
    os.chdir(ROOT)
    report = check_code()
    if args.command == "check":
        if args.data_root:
            report.update(check_data(args.data_root.resolve()))
        if args.benchmark_json or args.benchmark_images:
            if not (args.data_root and args.benchmark_json and args.benchmark_images):
                parser.error("Overlap validation needs --data-root, --benchmark-json and --benchmark-images")
            report.update(check_disjointness(args.data_root.resolve(), args.benchmark_json.resolve(), args.benchmark_images.resolve()))
        if args.checkpoint:
            if digest(args.checkpoint) != identity()["checkpoint_sha256"]:
                raise ValueError("Reference checkpoint hash mismatch")
            report["reference_checkpoint_verified"] = True
        if args.training_output:
            report["training_output"] = check_completion(args.training_output)
        if args.environment or args.strict_environment:
            report["environment"] = environment_report(args.strict_environment)
        print(json.dumps(report, indent=2))
        return
    if args.command == "install-artifacts":
        print(json.dumps(install_artifacts(args.bundle.resolve(), args.data_root.resolve()), indent=2))
        return
    if args.command == "results":
        print((ROOT / "reproducibility/results/results.json").read_text())
        return
    if args.command == "train":
        command = training_command(args.data_root, args.output)
        if args.dry_run:
            print(shlex.join(command))
            return
        report.update(check_data(args.data_root.resolve()))
        report.update(check_disjointness(args.data_root.resolve(), args.benchmark_json.resolve(), args.benchmark_images.resolve()))
        report["environment"] = environment_report(strict=True)
        if args.output.exists():
            raise FileExistsError("Use a new output directory; v1 never silently resumes or overwrites")
        require_gpu_step()
        args.output.mkdir(parents=True)
        (args.output / "release_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "training_command.json").write_text(json.dumps(command, indent=2) + "\n")
        subprocess.run(command, cwd=ROOT, env=child_environment(), check=True)
        result = check_completion(args.output)
        (args.output / "release_completion.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.command == "infer":
        if args.dry_run:
            print(json.dumps({"model": "Qwen/Qwen-Image-Edit-2509", "revision": "d3968ef930e841f4c73640fb8afa3b306a78167e", "steps": 40, "seed": 42, "true_cfg_scale": 4.0, "checkpoint": str(args.checkpoint), "output": str(args.output)}, indent=2))
            return
        if not args.allow_reconstructed_checkpoint and digest(args.checkpoint) != identity()["checkpoint_sha256"]:
            raise ValueError("Expected reference checkpoint; use --allow-reconstructed-checkpoint for retrained weights")
        if args.output.exists() or not args.image.is_file():
            raise ValueError("The input image must exist and the output must be a new file")
        require_gpu_step()
        patch_inference_processor()
        from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
        pipe = load_qwen_edit_pipeline(
            model_id_with_origin_paths="Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors",
            base_model="Qwen/Qwen-Image-Edit-2509", revision="d3968ef930e841f4c73640fb8afa3b306a78167e",
            checkpoint_path=str(args.checkpoint.resolve()), model_type="lora", device="cuda",
            torch_dtype="bfloat16", backend="official_diffusers",
        )
        output = render_edit(pipe, args.prompt, [args.image.resolve()], {"seed": 42, "num_inference_steps": 40, "true_cfg_scale": 4.0, "guidance_scale": 1.0, "negative_prompt": " ", "num_images_per_prompt": 1})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.images[0].save(args.output)
        return
    if args.command in {"export", "score"}:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        if cfg["benchmark"] != args.benchmark:
            raise ValueError("The config benchmark does not match --benchmark")
        module = f"qwen_edit_project.eval.{('export_' if args.command == 'export' else 'run_')}{args.benchmark}{('_score' if args.command == 'score' else '')}"
        arguments = ["--config", str(args.config.resolve())]
        if args.command == "export":
            arguments += ["--device", "cuda"]
        # Paths vary across installations; judge/protocol changes are explicitly new reruns.
        for value in [f"runtime.python_executable={sys.executable}", *args.set]:
            arguments += ["--set", value]
        if args.dry_run:
            print(shlex.join([sys.executable, "-m", module, *arguments]))
            return
        require_gpu_step()
        if args.command == "export":
            patch_inference_processor()
        delegate(module, arguments)
        return
    if args.command == "framework":
        arguments = ["--config", str(args.config.resolve())]
        for value in args.set:
            arguments += ["--set", value]
        # Print a plan only: historical runner construction can load real backends.
        if args.dry_run:
            print(shlex.join([sys.executable, "-m", "qwen_edit_project.self_evolve.run_loop", *arguments]))
            return
        require_gpu_step()
        patch_inference_processor()
        delegate("qwen_edit_project.self_evolve.run_loop", arguments)


if __name__ == "__main__":
    main()
