"""CPU-only validation for the internal Planner–Editor–Critic workflow."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/self_evolve/qwen_edit_2509_internal_cepr_rubric_trainable_proposer.yaml'
BASELINE_BRANCH = 'groundingdino-extraction'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_internal_config(config: dict) -> dict:
    evaluator = config.get('evaluator', config.get('solver', {}))
    if evaluator.get('backend') not in {'internal_cepr', 'contrastive_edit_preservation', 'internal_cepr_rubric', 'rubric_cepr', 'internal_rubric_cepr'}:
        raise ValueError('Main requires an internal CEPR evaluator')
    if evaluator.get('counterfactual_backend', 'internal') != 'internal':
        raise ValueError('CEPR counterfactual scoring must use internal features')
    if evaluator.get('require_internal_components', True) is not True:
        raise ValueError('Main requires internal scoring components; proxy fallback is disabled')
    if evaluator.get('object_grounder', 'qwen_vl_internal') != 'qwen_vl_internal':
        raise ValueError(f'External detector methods belong on {BASELINE_BRANCH}')
    proposer = config.get('proposer', {})
    if proposer.get('backend') not in {'trainable_qwen_image_edit', 'qwen_image_edit_lora', 'trainable_qwen_vl', 'qwen_vl_lora', 'internal_qwen'}:
        raise ValueError('Main requires an editor-side Qwen Planner')
    editor = config.get('editor', {})
    if editor.get('backend') != 'qwen_edit':
        raise ValueError('Main requires the Qwen editor')
    model_id = editor.get('model', {}).get('base_model')
    planner_id = proposer.get('model_name_or_path', model_id)
    if model_id != 'Qwen/Qwen-Image-Edit-2509' or planner_id != model_id:
        raise ValueError('Planner and Editor must use the same Qwen-Image-Edit backbone')
    if proposer.get('revision') != editor.get('model', {}).get('revision'):
        raise ValueError('Planner and Editor must use the same base-model revision')
    if evaluator.get('top_m', 1) != 1:
        raise ValueError('Main retains one selected target per proposal')
    training = config.get('training', {})
    weights = training.get('weighted_sft', {})
    if weights.get('enabled', False) and weights.get('include_rejected', True):
        raise ValueError('Main trains on selected gate-passing targets; rejected-target SFT is excluded')
    if weights.get('include_feasible_ranked_positives', False):
        raise ValueError('Main retains one selected target per proposal')
    if training.get('preference', {}).get('enabled', False):
        raise ValueError('Main uses accepted-target SFT; preference experiments use a separate recipe')
    from qwen_edit_project.self_evolve.training_weights import accepted_target_weight
    accepted_target_weight(1.0, scale=weights.get('accepted_weight', 1.0),
                           mode=str(weights.get('accepted_weight_mode', 'uniform')))
    return {'method': 'internal_cepr', 'external_training_models': False}


def check_code() -> dict:
    import yaml
    identity = json.loads((ROOT / 'reproducibility/framework.json').read_text())
    for name, expected in identity['pinned_files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Pinned framework file differs: {name}')
    report = validate_internal_config(yaml.safe_load(DEFAULT_CONFIG.read_text()))
    return {**report, 'pinned_files': len(identity['pinned_files']), 'code_verified': True}


def source_paths(manifest: Path) -> list[Path]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('The unlabeled source manifest is empty')
    keys, paths = set(), []
    for row in rows:
        if not isinstance(row, dict) or not row.get('key') or not row.get('image'):
            raise ValueError('Each source needs key and image fields')
        if row['key'] in keys:
            raise ValueError(f"Duplicate source key: {row['key']}")
        if row.get('caption') or any(row.get(field) for field in ('target', 'edit_image', 'chosen_image', 'rejected_image')):
            raise ValueError('Main takes unlabeled source images, not external training pairs or captions')
        keys.add(row['key'])
        path = Path(row['image']).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'Missing source image: {path}')
        paths.append(path.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate source image paths')
    if len({digest(path) for path in paths}) != len(paths):
        raise ValueError('Duplicate source image bytes')
    return paths


def check_data(manifest: Path) -> dict:
    paths = source_paths(manifest)
    return {'source_rows': len(paths), 'manifest_sha256': digest(manifest), 'data_verified': True}


def check_disjointness(manifest: Path, benchmark_json: Path, image_root: Path) -> dict:
    def no_duplicate_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate benchmark JSON key: {key}')
            result[key] = value
        return result
    records = json.loads(benchmark_json.read_text(), object_pairs_hook=no_duplicate_keys)
    if len(records) != 737:
        raise ValueError(f'Expected complete 737-row ImgEdit benchmark, found {len(records)}')
    sources = source_paths(manifest)
    images = []
    for row in records.values():
        relative = Path(row['id'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe benchmark image path')
        path = (image_root / relative).resolve()
        if not path.is_relative_to(image_root.resolve()) or not path.is_file():
            raise ValueError('Missing or out-of-root ImgEdit source image')
        images.append(path)
    if {path.name for path in sources} & {path.name for path in images}:
        raise ValueError('Training and benchmark source basenames overlap')
    if {digest(path) for path in sources} & {digest(path) for path in images}:
        raise ValueError('Training and benchmark source content overlaps')
    return {'benchmark_rows': len(records), 'basename_overlap': 0, 'content_overlap': 0}


def environment_report(strict: bool = False) -> dict:
    found, mismatches = {}, []
    for line in (ROOT / 'requirements-model.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, _, target = line.partition('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        found[name] = actual
        if actual is None or (target and actual.split('+')[0] != target):
            mismatches.append(name)
    if platform.python_version_tuple()[:2] != ('3', '11'):
        mismatches.append('python')
    report = {'python': platform.python_version(), 'packages': found, 'environment_mismatches': mismatches}
    if strict and mismatches:
        raise ValueError(f"Model environment mismatch: {', '.join(mismatches)}")
    return report


def check_completion(output: Path) -> dict:
    receipt = json.loads((output / 'training_completion.json').read_text())
    steps = receipt.get('global_step')
    requested = receipt.get('requested_max_train_steps') or receipt.get('max_train_steps')
    if receipt.get('status') != 'complete' or not isinstance(steps, int) or steps <= 0 or steps != requested:
        raise ValueError('Training did not complete its requested optimizer steps')
    weights = output / 'pytorch_lora_weights.safetensors'
    if not weights.is_file():
        raise ValueError('Completed training is missing its adapter')
    return {'complete': True, 'optimizer_steps': steps, 'sha256': digest(weights)}
