"""Check scheduler ownership before importing any model or evaluation code."""

from __future__ import annotations

import os
import pwd
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from .checks import ROOT


def require_gpu_step() -> None:
    job = os.environ.get("SLURM_JOB_ID", "")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not job.isdigit() or not step.isdigit():
        raise RuntimeError("Use an owned RUNNING Slurm GPU allocation with a numeric srun step; login execution is forbidden")
    if not shutil.which("scontrol"):
        raise RuntimeError("Cannot verify the Slurm allocation: scontrol is unavailable")
    record = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(item.split("=", 1) for item in record.split() if "=" in item)
    owner = fields.get("UserId", "").split("(", 1)[0]
    actual_user = pwd.getpwuid(os.geteuid()).pw_name
    if owner != actual_user or fields.get("JobState") != "RUNNING":
        raise RuntimeError("The scheduler does not report an owned RUNNING job")
    if not re.search(r"(?:^|,)gres/gpu(?::[^=,]+)?=[1-9][0-9]*(?:,|$)", fields.get("AllocTRES", "")):
        raise RuntimeError("The allocation has no assigned GPU")
    step_gpus = os.environ.get("SLURM_STEP_GPUS", "")
    if not step_gpus or step_gpus.lower() in {"none", "n/a", "(null)", "nodevfiles"}:
        raise RuntimeError("The srun step has no GPU IDs")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ["CUDA_VISIBLE_DEVICES"] in {"-1", "NoDevFiles"}:
        raise RuntimeError("Slurm has not exposed a GPU to this process")
    # A real numeric step must be visible to the controller, not only inherited env vars.
    subprocess.run(["scontrol", "show", "step", f"{job}.{step}"], check=True, stdout=subprocess.DEVNULL)
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", fields.get("NodeList", "")], text=True)
    if socket.gethostname().split(".")[0] not in {node.split(".")[0] for node in nodes.split()}:
        raise RuntimeError("The current host is outside the allocated node list")
    if not os.environ.get("TMUX"):
        raise RuntimeError("Submit from a newly inspected tmux session and export TMUX into the Slurm worker")
    if str(ROOT).startswith("/share_6/users/ritesh_thawkar/"):
        expected = Path("/share_6/users/ritesh_thawkar/condaenvs/qedit/bin/python")
        if actual_user != "ritesh_thawkar" or Path(sys.executable).resolve() != expected.resolve():
            raise RuntimeError("This workspace requires ritesh_thawkar and the activated qedit Python")
        if os.environ.get("HF_HOME") != "/share_6/users/ritesh_thawkar/.cache/huggingface":
            raise RuntimeError("Set the workspace policy HF_HOME before model execution")
    if str(ROOT / "src") not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        raise RuntimeError("Set PYTHONPATH to this release's src directory")


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    return environment
