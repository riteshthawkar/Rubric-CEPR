"""CPU-only checks for portability, unsafe inputs and evidence boundaries."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from rubric_cepr import checks, cli, runtime


def test_original_trainer_and_manifest_are_pinned():
    assert checks.check_code()["rows"] == 64
    assert checks.digest(checks.ROOT / "src/qwen_edit_project/train/diffusers_qwen_edit_lora.py") == "0d033ef25ec85d4b80fdd00f594badb4223ec7ff481d9cd32c1109d1f6783b84"


def test_portable_command_uses_exact_scientific_arguments(tmp_path):
    command = cli.training_command(tmp_path / "inputs", tmp_path / "output")
    # Argument boundaries matter more than shell spelling or site-local paths.
    def flag(name):
        return command[command.index(name) + 1]
    assert flag("--max_train_steps") == "400"
    assert flag("--learning_rate") == "1.0e-4"
    assert flag("--rank") == flag("--lora_alpha") == "16"
    assert flag("--seed") == "123"
    assert flag("--training_objective") == "sft"
    assert flag("--dataset_base_path") == str((tmp_path / "inputs").resolve())
    assert flag("--dataset_metadata_path").endswith("extraction_v1.json")
    assert "--local_files_only" in command
    assert "--resume_from_checkpoint" not in command
    assert flag("--lora_reference_l2_weight") == flag("--lora_reference_max_relative_delta") == "0.0"


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "a/../../outside"])
def test_artifact_path_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        checks.safe_relative_path(tmp_path, name)


def test_artifact_ancestor_symlink_is_rejected(tmp_path):
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        checks.safe_relative_path(tmp_path, "alias/file")


def test_content_overlap_is_rejected_even_after_source_rename(tmp_path, monkeypatch):
    root = tmp_path / "release"
    manifest_dir = root / "reproducibility/manifests"
    manifest_dir.mkdir(parents=True)
    data = tmp_path / "training"
    images = tmp_path / "benchmark"
    data.mkdir()
    images.mkdir()
    (data / "source.jpg").write_bytes(b"same image content")
    (images / "renamed.jpg").write_bytes(b"same image content")
    (manifest_dir / "extraction_v1.json").write_text(json.dumps([{"edit_image": "source.jpg"}]))
    records = {str(i): {"id": "renamed.jpg"} for i in range(737)}
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(records))
    monkeypatch.setattr(checks, "ROOT", root)
    with pytest.raises(ValueError, match="content overlaps"):
        checks.check_disjointness(data, benchmark, images)


def test_source_basename_overlap_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "release"
    manifest_dir = root / "reproducibility/manifests"
    manifest_dir.mkdir(parents=True)
    data = tmp_path / "training"
    images = tmp_path / "benchmark"
    data.mkdir()
    images.mkdir()
    (data / "source.jpg").write_bytes(b"training image")
    (images / "source.jpg").write_bytes(b"different bytes")
    (manifest_dir / "extraction_v1.json").write_text(json.dumps([{"edit_image": "source.jpg"}]))
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({str(i): {"id": "source.jpg"} for i in range(737)}))
    monkeypatch.setattr(checks, "ROOT", root)
    with pytest.raises(ValueError, match="basenames overlap"):
        checks.check_disjointness(data, benchmark, images)


def test_corrupted_bundle_is_rejected_before_destination_exists(tmp_path):
    bundle = tmp_path / "corrupt.tar.gz"
    bundle.write_bytes(b"not the historical bundle")
    destination = tmp_path / "data"
    with pytest.raises(ValueError, match="hash"):
        cli.install_artifacts(bundle, destination)
    assert not destination.exists()


def test_archive_link_is_rejected_even_with_matching_container_hash(tmp_path, monkeypatch):
    archive = tmp_path / "linked.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("extraction_v1/repository/file")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/outside"
        handle.addfile(member)
    monkeypatch.setattr(cli, "identity", lambda: {**checks.identity(), "bundle_sha256": checks.digest(archive)})
    with pytest.raises(ValueError, match="non-file"):
        cli.install_artifacts(archive, tmp_path / "new-data")
    assert not (tmp_path / "new-data").exists()


def test_duplicate_archive_entries_are_rejected(tmp_path, monkeypatch):
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for _ in range(2):
            member = tarfile.TarInfo("duplicate")
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
    monkeypatch.setattr(cli, "identity", lambda: {**checks.identity(), "bundle_sha256": checks.digest(archive)})
    with pytest.raises(ValueError, match="duplicate"):
        cli.install_artifacts(archive, tmp_path / "new-data")


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


@pytest.mark.parametrize("steps", [0, 399, 401])
def test_incomplete_or_excess_training_is_rejected(tmp_path, steps):
    (tmp_path / "training_completion.json").write_text(json.dumps({"status": "complete", "global_step": steps}))
    with pytest.raises(ValueError, match="400"):
        checks.check_completion(tmp_path)


def test_dry_runs_do_not_import_models_or_call_scheduler(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "require_gpu_step", lambda: pytest.fail("GPU guard called in CPU-only dry run"))
    monkeypatch.setattr(cli, "delegate", lambda *args: pytest.fail("Low-level runtime imported during dry run"))
    cli.main(["train", "--data-root", str(tmp_path / "missing"), "--output", str(tmp_path / "unused"), "--dry-run"])
    cli.main(["framework", "--dry-run"])
    cli.main(["export", "--benchmark", "imgedit", "--config", str(checks.ROOT / "configs/eval/imgedit_hardened.yaml"), "--dry-run"])
    assert not (tmp_path / "unused").exists()
    assert "Qwen/Qwen-Image-Edit-2509" in capsys.readouterr().out


def test_results_preserve_measured_scores_and_negative_control():
    evidence = json.loads((checks.ROOT / "reproducibility/results/results.json").read_text())
    assert evidence["imgedit_737"]["extraction_v1"]["overall"] == 4.5550881953866975
    assert evidence["complex_edit_real_c4_531"]["extraction_v1"]["overall"] == 8.8064
    assert evidence["gedit_full_1212"]["round_loop_candidate"]["overall_delta"] < 0
    for item in json.loads((checks.ROOT / "reproducibility/score_sources.json").read_text()):
        assert checks.digest(checks.ROOT / item["release_path"]) == item["release_sha256"]


def test_complex_edit_config_keeps_strict_dataset_and_judge():
    from qwen_edit_project.eval.complex_edit_contract import ComplexEditContractError, validate_hardened_config
    config = yaml.safe_load((checks.ROOT / "configs/eval/complex_edit_c4_hardened.yaml").read_text())
    config["runtime"]["python_executable"] = sys.executable
    validate_hardened_config(config)
    config["dataset"]["complexities"] = [1]
    with pytest.raises(ComplexEditContractError):
        validate_hardened_config(config)


def test_cli_resolves_caller_relative_paths_before_chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "adapter.bin"
    checkpoint.write_bytes(b"test-adapter")
    monkeypatch.setattr(cli, "identity", lambda: {"checkpoint_sha256": checks.digest(checkpoint)})
    cli.main(["check", "--checkpoint", "adapter.bin"])


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
    name = "qwen_edit_project.train.diffusers_qwen_edit_lora_reproduction_v1"
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(ProcessorFolderCompatibilityLoader=Loader))
    monkeypatch.setattr(qwen_pipeline, "_from_pretrained_with_dtype", qwen_pipeline._from_pretrained_with_dtype)
    cli.patch_inference_processor()
    result = qwen_pipeline._from_pretrained_with_dtype(Pipeline, "official-model", "bf16", True, "pinned-revision")
    assert result == "pipeline"
    assert calls[0] == ("official-model", {"subfolder": "processor", "revision": "pinned-revision", "local_files_only": True})
    assert calls[1][1]["processor"] == "official-processor-components"
    assert calls[1][1]["revision"] == "pinned-revision"


def test_secret_audit_reports_location_without_disclosing_token(tmp_path):
    spec = importlib.util.spec_from_file_location("release_verifier", checks.ROOT / "tools/verify_release.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    token = "hf_" + "a" * 30
    (tmp_path / "accidental.txt").write_text(token)
    with pytest.raises(ValueError) as error:
        verifier.audit(tmp_path)
    assert "accidental.txt" in str(error.value)
    assert token not in str(error.value)
