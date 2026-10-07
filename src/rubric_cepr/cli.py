"""Public internal-CEPR commands; model imports follow Slurm verification."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import sys
from pathlib import Path

from .checks import (
    ROOT, DEFAULT_CONFIG, BASELINE_BRANCH, check_code, check_completion,
    check_data, check_disjointness, digest, environment_report, validate_internal_config,
)
from .runtime import require_gpu_step


def delegate(module: str, arguments: list[str]) -> None:
    saved = sys.argv
    try:
        sys.argv = [module, *arguments]
        importlib.import_module(module).main()
    finally:
        sys.argv = saved


def patch_inference_processor() -> None:
    from qwen_edit_project.train.diffusers_qwen_edit_lora_compat import ProcessorFolderCompatibilityLoader
    from qwen_edit_project.utils import qwen_pipeline
    def load(pipeline_cls, model_id, dtype, local_files_only, revision=None):
        processor = ProcessorFolderCompatibilityLoader.from_pretrained(
            model_id, subfolder='processor', revision=revision, local_files_only=local_files_only,
        )
        return pipeline_cls.from_pretrained(
            model_id, torch_dtype=dtype, processor=processor,
            local_files_only=local_files_only, revision=revision,
        )
    qwen_pipeline._from_pretrained_with_dtype = load


def workflow_plan(args) -> tuple[dict, list[str]]:
    from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override
    config = load_yaml_config(args.config)
    overrides = []
    if args.command == 'train':
        overrides += ['training.trigger=launch', 'proposer.training.trigger=launch']
    if args.manifest:
        overrides += [f'dataset.manifest_jsonl={args.manifest}']
    if args.output:
        overrides += [f'output.root_dir={args.output}']
    overrides += args.set
    for value in overrides:
        key, parsed = parse_override(value)
        config = merge_override(config, key, parsed)
    validate_internal_config(config)
    arguments = ['--config', str(args.config)]
    for value in overrides:
        arguments += ['--set', value]
    return config, arguments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Rubric-CEPR internal self-evolution and benchmark evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='CPU-only internal-method, data and environment checks')
    check.add_argument('--manifest', type=Path)
    check.add_argument('--benchmark-json', type=Path)
    check.add_argument('--benchmark-images', type=Path)
    check.add_argument('--checkpoint', type=Path)
    check.add_argument('--training-output', type=Path)
    check.add_argument('--environment', action='store_true')
    check.add_argument('--strict-environment', action='store_true')
    check.add_argument('--data-root', type=Path, help=argparse.SUPPRESS)
    install = commands.add_parser('install-artifacts', help=f'Legacy extraction bundle: use branch {BASELINE_BRANCH}')
    install.add_argument('--bundle', type=Path)
    install.add_argument('--data-root', type=Path)
    for name in ('train', 'framework'):
        sub = commands.add_parser(name, help='Run internal CEPR with training' if name == 'train' else 'Run internal CEPR candidate generation and verification')
        sub.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        sub.add_argument('--manifest', type=Path)
        sub.add_argument('--output', type=Path)
        sub.add_argument('--benchmark-json', type=Path, default=ROOT / 'data/processed/benchmark/imgedit/basic_edit.json')
        sub.add_argument('--benchmark-images', type=Path, default=ROOT / 'data/processed/benchmark/imgedit/original_images')
        sub.add_argument('--set', action='append', default=[])
        sub.add_argument('--dry-run', action='store_true')
        sub.add_argument('--data-root', type=Path, help=argparse.SUPPRESS)
    infer = commands.add_parser('infer', help='Edit one image using Qwen and a supplied adapter')
    infer.add_argument('--image', required=True, type=Path)
    infer.add_argument('--prompt', required=True)
    infer.add_argument('--checkpoint', required=True, type=Path)
    infer.add_argument('--expected-checkpoint-sha256')
    infer.add_argument('--output', required=True, type=Path)
    infer.add_argument('--allow-reconstructed-checkpoint', action='store_true', help=argparse.SUPPRESS)
    infer.add_argument('--dry-run', action='store_true')
    for name in ('export', 'score'):
        sub = commands.add_parser(name, help=f'{name.title()} a benchmark with a matched evaluation protocol')
        sub.add_argument('--benchmark', required=True, choices=['imgedit', 'gedit', 'complex_edit'])
        sub.add_argument('--config', required=True, type=Path)
        sub.add_argument('--set', action='append', default=[])
        sub.add_argument('--dry-run', action='store_true')
    commands.add_parser('results', help='Display available evidence and branch locations')
    args = parser.parse_args(argv)
    for field, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, field, value.expanduser().resolve())
    if args.command == 'install-artifacts' or getattr(args, 'data_root', None):
        parser.error(f'The fixed extraction bundle and --data-root recipe are on {BASELINE_BRANCH}. On main use train --manifest sources.jsonl; see docs/TRAINING.md.')
    os.chdir(ROOT)
    report = check_code()
    if args.command == 'check':
        if args.manifest:
            report.update(check_data(args.manifest))
        if args.benchmark_json or args.benchmark_images:
            if not (args.manifest and args.benchmark_json and args.benchmark_images):
                parser.error('Overlap validation needs --manifest, --benchmark-json and --benchmark-images')
            report.update(check_disjointness(args.manifest, args.benchmark_json, args.benchmark_images))
        if args.checkpoint:
            report['checkpoint_sha256'] = digest(args.checkpoint)
        if args.training_output:
            report['training_output'] = check_completion(args.training_output)
        if args.environment or args.strict_environment:
            report['environment'] = environment_report(args.strict_environment)
        print(json.dumps(report, indent=2))
        return
    if args.command == 'results':
        print((ROOT / 'reproducibility/results/results.json').read_text())
        return
    if args.command in {'train', 'framework'}:
        config, arguments = workflow_plan(args)
        if args.dry_run:
            print(shlex.join([sys.executable, '-m', 'qwen_edit_project.self_evolve.run_loop', *arguments]))
            return
        if config.get('dataset', {}).get('source') != 'jsonl':
            raise ValueError('The public workflow requires an audited JSONL source manifest')
        manifest = Path(config['dataset']['manifest_jsonl'])
        manifest = manifest if manifest.is_absolute() else ROOT / manifest
        check_data(manifest)
        check_disjointness(manifest, args.benchmark_json, args.benchmark_images)
        environment_report(strict=True)
        require_gpu_step()
        patch_inference_processor()
        delegate('qwen_edit_project.self_evolve.run_loop', arguments)
        return
    if args.command == 'infer':
        settings = {'model': 'Qwen/Qwen-Image-Edit-2509', 'revision': 'd3968ef930e841f4c73640fb8afa3b306a78167e', 'seed': 42, 'num_inference_steps': 40, 'true_cfg_scale': 4.0, 'guidance_scale': 1.0, 'negative_prompt': ' ', 'num_images_per_prompt': 1}
        if args.dry_run:
            print(json.dumps({**settings, 'checkpoint': str(args.checkpoint), 'output': str(args.output)}, indent=2))
            return
        checkpoint_sha = digest(args.checkpoint)
        if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
            raise ValueError('Supplied checkpoint hash mismatch')
        if args.output.exists() or not args.image.is_file():
            raise ValueError('The input image must exist and output must be a new file')
        require_gpu_step()
        patch_inference_processor()
        from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
        pipe = load_qwen_edit_pipeline(
            model_id_with_origin_paths='Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors',
            base_model=settings['model'], revision=settings['revision'],
            checkpoint_path=str(args.checkpoint), model_type='lora', device='cuda',
            torch_dtype='bfloat16', backend='official_diffusers',
        )
        output = render_edit(pipe, args.prompt, [args.image], {key: value for key, value in settings.items() if key not in {'model', 'revision'}})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.images[0].save(args.output)
        return
    if args.command in {'export', 'score'}:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        if cfg['benchmark'] != args.benchmark:
            raise ValueError('The config benchmark does not match --benchmark')
        module = f"qwen_edit_project.eval.{('export_' if args.command == 'export' else 'run_')}{args.benchmark}{('_score' if args.command == 'score' else '')}"
        arguments = ['--config', str(args.config)]
        if args.command == 'export':
            arguments += ['--device', 'cuda']
        for value in [f'runtime.python_executable={sys.executable}', *args.set]:
            arguments += ['--set', value]
        if args.dry_run:
            print(shlex.join([sys.executable, '-m', module, *arguments]))
            return
        require_gpu_step()
        if args.command == 'export':
            patch_inference_processor()
        delegate(module, arguments)


if __name__ == '__main__':
    main()
