"""CPU regression checks for branch boundaries, inputs and model launch guards."""
import json
import sys
from pathlib import Path

import pytest
import yaml

from rubric_cepr import checks, cli, runtime

@pytest.mark.parametrize("step", ["", "batch", "extern"])
def test_login_or_batch_context_is_rejected_before_scheduler_query(monkeypatch, step):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", step)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw: pytest.fail("Scheduler should not be queried for an invalid step"))
    with pytest.raises(RuntimeError, match="numeric"):
        runtime.require_gpu_step()



@pytest.mark.parametrize("record", [
    "UserId=another_user(42) JobState=RUNNING AllocTRES=gres/gpu=1",
    "UserId=test_user(42) JobState=PENDING AllocTRES=gres/gpu=1",
    "UserId=test_user(42) JobState=RUNNING AllocTRES=cpu=4,mem=32G",
])
def test_unowned_pending_or_cpu_only_jobs_are_rejected(monkeypatch, record):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/fake/scontrol")
    from types import SimpleNamespace
    monkeypatch.setattr(runtime.pwd, "getpwuid", lambda _: SimpleNamespace(pw_name="test_user"))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw: record)
    with pytest.raises(RuntimeError):
        runtime.require_gpu_step()



def test_complex_edit_config_keeps_strict_dataset_and_judge():
    from qwen_edit_project.eval.complex_edit_contract import ComplexEditContractError, validate_hardened_config
    config = yaml.safe_load((checks.ROOT / "configs/eval/complex_edit_c4_hardened.yaml").read_text())
    config["runtime"]["python_executable"] = sys.executable
    validate_hardened_config(config)
    config["dataset"]["complexities"] = [1]
    with pytest.raises(ComplexEditContractError):
        validate_hardened_config(config)



def test_inference_compatibility_loader_keeps_model_revision(monkeypatch):
    from types import SimpleNamespace
    from qwen_edit_project.utils import qwen_pipeline
    calls = []
    class Loader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return "official-processor-components"
    class Pipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return "pipeline"
    name = "qwen_edit_project.train.diffusers_qwen_edit_lora_compat"
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(ProcessorFolderCompatibilityLoader=Loader))
    monkeypatch.setattr(qwen_pipeline, "_from_pretrained_with_dtype", qwen_pipeline._from_pretrained_with_dtype)
    cli.patch_inference_processor()
    result = qwen_pipeline._from_pretrained_with_dtype(Pipeline, "official-model", "bf16", True, "pinned-revision")
    assert result == "pipeline"
    assert calls[0] == ("official-model", {"subfolder": "processor", "revision": "pinned-revision", "local_files_only": True})
    assert calls[1][1]["processor"] == "official-processor-components"
    assert calls[1][1]["revision"] == "pinned-revision"



def config():
    return yaml.safe_load(checks.DEFAULT_CONFIG.read_text())


def test_main_pins_internal_framework_without_legacy_pairs():
    report = checks.check_code()
    assert report['method'] == 'internal_cepr'
    assert report['external_training_models'] is False
    assert not (checks.ROOT / 'scripts/build_extract_selfdistill.py').exists()
    assert not (checks.ROOT / 'reproducibility/manifests/extraction_v1.json').exists()
    source = (checks.ROOT / 'src/qwen_edit_project/self_evolve/backends.py').read_text()
    assert 'GroundingDinoForObjectDetection' not in source
    assert 'grounding-dino-tiny' not in source


@pytest.mark.parametrize('override', [
    'evaluator.object_grounder=grounding_dino',
    'evaluator.counterfactual_backend=proxy',
    'evaluator.backend=hybrid',
    'evaluator.require_internal_components=false',
    'proposer.model_name_or_path=external-model',
])
def test_external_scoring_overrides_fail_before_model_execution(override, monkeypatch):
    monkeypatch.setattr(cli, 'require_gpu_step', lambda: pytest.fail('Scheduler should not be called'))
    monkeypatch.setattr(cli, 'delegate', lambda *args: pytest.fail('A model should not be imported'))
    with pytest.raises(ValueError):
        cli.main(['framework', '--set', override, '--dry-run'])


def test_direct_evaluator_cannot_enable_external_grounding():
    from qwen_edit_project.self_evolve.backends import InternalRubricCEPREvaluator
    with pytest.raises(ValueError, match='External detector variants'):
        InternalRubricCEPREvaluator({'object_grounder': 'grounding_dino'})


def test_source_caption_or_edited_target_is_rejected(tmp_path):
    image = tmp_path / 'source.jpg'
    image.write_bytes(b'source bytes')
    manifest = tmp_path / 'sources.jsonl'
    for extra in ({'caption': 'external caption'}, {'target': 'target.png'}, {'chosen_image': 'chosen.png'}):
        manifest.write_text(json.dumps({'key': 'one', 'image': str(image), **extra}) + '\n')
        with pytest.raises(ValueError, match='unlabeled'):
            checks.check_data(manifest)


def test_duplicate_source_bytes_are_rejected(tmp_path):
    first, second = tmp_path / 'a.jpg', tmp_path / 'b.jpg'
    first.write_bytes(b'same bytes')
    second.write_bytes(b'same bytes')
    manifest = tmp_path / 'sources.jsonl'
    manifest.write_text('\n'.join(json.dumps({'key': str(i), 'image': str(path)}) for i, path in enumerate([first, second])))
    with pytest.raises(ValueError, match='Duplicate source image bytes'):
        checks.check_data(manifest)


def test_renamed_source_content_overlap_is_rejected(tmp_path):
    image = tmp_path / 'source.jpg'
    image.write_bytes(b'same bytes')
    manifest = tmp_path / 'sources.jsonl'
    manifest.write_text(json.dumps({'key': 'one', 'image': str(image)}) + '\n')
    image_root = tmp_path / 'benchmark'
    image_root.mkdir()
    (image_root / 'renamed.jpg').write_bytes(b'same bytes')
    benchmark = tmp_path / 'benchmark.json'
    benchmark.write_text(json.dumps({str(i): {'id': 'renamed.jpg'} for i in range(737)}))
    with pytest.raises(ValueError, match='content overlaps'):
        checks.check_disjointness(manifest, benchmark, image_root)


def test_partial_benchmark_cannot_disable_overlap_guard(tmp_path):
    benchmark = tmp_path / 'partial.json'
    benchmark.write_text('{}')
    with pytest.raises(ValueError, match='complete 737-row'):
        checks.check_disjointness(tmp_path / 'missing.jsonl', benchmark, tmp_path)


def test_dry_runs_use_internal_loop_without_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'require_gpu_step', lambda: pytest.fail('GPU guard called in dry run'))
    monkeypatch.setattr(cli, 'delegate', lambda *args: pytest.fail('Model code imported in dry run'))
    cli.main(['train', '--manifest', str(tmp_path / 'sources.jsonl'), '--output', str(tmp_path / 'unused'), '--dry-run'])
    train = capsys.readouterr().out
    assert 'self_evolve.run_loop' in train
    assert 'training.trigger=launch' in train
    assert 'proposer.training.trigger=launch' in train
    assert 'extraction_v1' not in train
    cli.main(['framework', '--dry-run'])
    assert 'training.trigger=launch' not in capsys.readouterr().out
    cli.main(['export', '--benchmark', 'imgedit', '--config', str(checks.ROOT / 'configs/eval/imgedit_hardened.yaml'), '--dry-run'])
    assert 'eval.export_imgedit' in capsys.readouterr().out
    assert not (tmp_path / 'unused').exists()


def test_legacy_bundle_commands_direct_users_to_baseline_branch(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['train', '--data-root', str(tmp_path), '--dry-run'])
    assert exc.value.code == 2
    assert 'groundingdino-extraction' in capsys.readouterr().err


def test_gpu_guard_precedes_model_delegate(monkeypatch):
    monkeypatch.setattr(cli, 'check_data', lambda *args: {})
    monkeypatch.setattr(cli, 'check_disjointness', lambda *args: {})
    monkeypatch.setattr(cli, 'environment_report', lambda **kwargs: {})
    def refuse():
        raise RuntimeError('no numeric GPU step')
    monkeypatch.setattr(cli, 'require_gpu_step', refuse)
    monkeypatch.setattr(cli, 'patch_inference_processor', lambda: pytest.fail('Model module imported'))
    monkeypatch.setattr(cli, 'delegate', lambda *args: pytest.fail('Model module imported'))
    with pytest.raises(RuntimeError, match='numeric GPU step'):
        cli.main(['framework'])


@pytest.mark.parametrize('steps', [0, 15, 17])
def test_completion_requires_the_actual_requested_steps(tmp_path, steps):
    (tmp_path / 'training_completion.json').write_text(json.dumps({
        'status': 'complete', 'global_step': steps, 'requested_max_train_steps': 16,
    }))
    with pytest.raises(ValueError, match='requested optimizer steps'):
        checks.check_completion(tmp_path)


def test_results_do_not_assign_baseline_scores_to_main():
    report = json.loads((checks.ROOT / 'reproducibility/results/results.json').read_text())
    assert report['method'] == 'internal_cepr'
    assert report['provenance_document'] == 'docs/PAPER_PROVENANCE.md'
    assert (checks.ROOT / report['provenance_document']).is_file()
    assert report['detector_assisted_baseline']['branch'] == 'groundingdino-extraction'
    assert 'imgedit_737' not in report


def test_cli_resolves_caller_relative_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / 'adapter.bin'
    checkpoint.write_bytes(b'test-adapter')
    cli.main(['check', '--checkpoint', 'adapter.bin'])
    report = json.loads(capsys.readouterr().out)
    assert report['checkpoint_sha256'] == checks.digest(checkpoint)
