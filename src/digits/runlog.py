"""Writes metrics.jsonl, meta.json and the heartbeat into a run folder (plan 6.3).

    with RunLogger(config, name="cnn_small") as run:
        run.log(step=100, epoch=1, split="train", loss=0.3, acc=0.91, lr=1e-3)

Run folder $RUNS_DIR/mnist/<run-id>/: meta.json, config.yaml, heartbeat, metrics.jsonl,
checkpoints/, predictions/. meta.json gets "status" running -> finished, or crashed if the block
exits with an exception; a killed process just stops touching the heartbeat.

    with RunLogger(config, resume=run_dir) as run:      # continue an existing run folder

On resume, metrics.jsonl is appended to (records after `resume_step`, written after the checkpoint
the run resumes from, are dropped first), config.yaml is replaced by the new config and meta.json
gets the status running again plus the time in "resumed" (a list).
"""

import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from digits.data import REPO_DIR, dataset_version

HEARTBEAT_SECONDS = 10


def runs_dir() -> Path:
    """$RUNS_DIR/mnist from the environment or the repo's .env file."""
    load_dotenv(REPO_DIR / ".env")
    return Path(os.environ["RUNS_DIR"]) / "mnist"


def git_commit() -> str:
    """HEAD of the repo, with a "-dirty" suffix if there are uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RunLogger:
    def __init__(
        self,
        config: dict,
        name: str | None = None,
        root: Path | None = None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        resume: Path | None = None,
        resume_step: int | None = None,
    ):
        if resume is not None:
            self.dir = Path(resume)
            self.run_id = self.dir.name
            if resume_step is not None:
                self._truncate_metrics(resume_step)
            self.meta = json.loads((self.dir / "meta.json").read_text())
            self.meta.pop("end_time", None)
            self.meta.update(status="running")
            self.meta.setdefault("resumed", []).append(now_iso())
        else:
            name = name or str(config.get("model", "run"))
            self.run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
            self.dir = (root or runs_dir()) / self.run_id
            self.dir.mkdir(parents=True)
            (self.dir / "checkpoints").mkdir()
            (self.dir / "predictions").mkdir()
            self.meta = {
                "run_id": self.run_id,
                "git_commit": git_commit(),
                "dataset_version": dataset_version(),
                "start_time": now_iso(),
                "host": socket.gethostname(),
                "status": "running",
            }
        self._write_meta()
        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._stop = threading.Event()
        self._heartbeat_seconds = heartbeat_seconds
        self._touch()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _truncate_metrics(self, step: int) -> None:
        path = self.dir / "metrics.jsonl"
        lines = path.read_text().splitlines(keepends=True) if path.exists() else []
        path.write_text("".join(s for s in lines if json.loads(s).get("step", 0) <= step))

    def _write_meta(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2) + "\n")
        tmp.replace(self.dir / "meta.json")

    def _touch(self) -> None:
        (self.dir / "heartbeat").write_text(now_iso() + "\n")

    def _beat(self) -> None:
        while not self._stop.wait(self._heartbeat_seconds):
            self._touch()

    def log(self, **fields) -> None:
        """Append one JSON line, e.g. step, epoch, split, loss, acc, lr, img_s, gpu_mem_gb."""
        record = {**fields, "ts": round(time.time(), 3)}
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def close(self, status: str = "finished") -> None:
        self._stop.set()
        self._thread.join()
        self._metrics.close()
        self._touch()
        self.meta.update(status=status, end_time=now_iso())
        self._write_meta()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close("finished" if exc_type is None else "crashed")
