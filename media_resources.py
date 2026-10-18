"""Bound FFmpeg memory on small hosts; permit explicit high-resource jobs on opted-in hosts."""
from __future__ import annotations

import asyncio
import contextvars
import os
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import HTTPException

_HIGH = contextvars.ContextVar("media_high_resources", default=False)


def high_resources_enabled() -> bool:
    return _HIGH.get()


def validate_high_resources(enabled: bool) -> None:
    if enabled and os.getenv("MEDIA_ALLOW_UNLIMITED_RESOURCES", "false").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "High-resource mode is disabled on this server")


def spawn_media_task(coro, high_resources: bool):
    context = contextvars.copy_context()
    context.run(_HIGH.set, high_resources)
    return asyncio.create_task(coro, context=context)


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _cgroup_pressure() -> tuple[int, int] | None:
    """Return container usage and limit on Linux cgroup v2 or v1."""
    try:
        root = Path("/sys/fs/cgroup")
        maximum = (root / "memory.max").read_text().strip()
        if maximum == "max":
            return None
        used = int((root / "memory.current").read_text())
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        return max(0, used - int(stats.get("inactive_file", "0"))), int(maximum)
    except (OSError, ValueError):
        try:
            root = Path("/sys/fs/cgroup/memory")
            maximum = int((root / "memory.limit_in_bytes").read_text())
            if maximum >= 1 << 60:
                return None
            used = int((root / "memory.usage_in_bytes").read_text())
            stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
            return max(0, used - int(stats.get("total_inactive_file", "0"))), maximum
        except (OSError, ValueError):
            return None


def require_memory_headroom(stage: str, *, minimum_mb: int = 176) -> None:
    """Admission check before a large allocation; never treat a missing limit as unlimited."""
    if high_resources_enabled():
        return
    pressure = _cgroup_pressure()
    if pressure is None:
        if os.getenv("MEDIA_REQUIRE_CGROUP", "false").lower() in {"1", "true", "yes"}:
            raise RuntimeError(f"{stage}: cgroup memory limit unavailable; refusing resource-limited work")
        return
    used, limit = pressure
    reserve = max(32, int(os.getenv("MEDIA_CGROUP_RESERVE_MB", "128"))) * 1048576
    minimum = max(64, int(os.getenv("MEDIA_MIN_START_HEADROOM_MB", str(minimum_mb)))) * 1048576
    if limit - used - reserve < minimum:
        mb = 1048576
        raise RuntimeError(
            f"{stage}: insufficient memory before starting "
            f"(active={used // mb} MiB, limit={limit // mb} MiB, "
            f"reserved={reserve // mb} MiB, required_free={minimum // mb} MiB)"
        )


def validate_video_source(path: Path, *, kind: str = "video") -> None:
    """Reject unusually large decoded input frames before FFmpeg/Whisper is launched."""
    if high_resources_enabled():
        return
    probe = shutil.which("ffprobe")
    if not probe:
        raise RuntimeError("ffprobe is required for media admission")
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError("Could not probe input video dimensions")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("Source contains no video track")
    width, height = int(streams[0]["width"]), int(streams[0]["height"])
    setting = "MEDIA_MAX_IMAGE_PIXELS" if kind == "image" else "MEDIA_MAX_INPUT_PIXELS"
    maximum = max(1, int(os.getenv(setting, "2073600")))
    if width <= 0 or height <= 0 or width * height > maximum:
        raise RuntimeError(
            f"Input {kind} {width}x{height} exceeds small-host decoded-pixel "
            f"admission limit {maximum}; submit to the high-resource host"
        )


def _tuned_command(command: list[str], high: bool) -> list[str]:
    command = list(command)
    if "ffmpeg" not in Path(command[0]).name:
        return command
    # These flags must precede inputs to control decoder and filter graph workers.
    if high:
        for flag in ("-threads", "-filter_threads", "-filter_complex_threads"):
            for i in range(len(command) - 1):
                if command[i] == flag and command[i + 1] == "1":
                    command[i + 1] = "0"
    else:
        command[1:1] = ["-max_alloc", str(64 * 1048576),
                         "-filter_threads", "1", "-filter_complex_threads", "1", "-threads", "1"]
    return command


def run_media_command(command: list[str], timeout: int, label: str) -> subprocess.CompletedProcess:
    """Stream process logs to disk and stop a job before its host hits OOM pressure.

    The process RSS and cgroup checks are sampled, not a guaranteed hard memory
    limit; a kernel OOM can still occur between samples or inside Python/Whisper.
    """
    high = _HIGH.get()
    command = _tuned_command(command, high)
    if not high:
        require_memory_headroom(label)
    rss_budget = "MEDIA_WHISPER_MAX_RSS_MB" if "media_transcribe" in command else "MEDIA_FFMPEG_MAX_RSS_MB"
    rss_limit = max(96, int(os.getenv(rss_budget, "256"))) * 1048576
    reserve = max(32, int(os.getenv("MEDIA_CGROUP_RESERVE_MB", "128"))) * 1048576
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        started = time.monotonic()
        reason = None
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    reason = f"{label} exceeded {timeout}s timeout"
                    break
                if not high:
                    rss = _rss_bytes(process.pid)
                    if rss > rss_limit:
                        reason = f"{label} exceeded subprocess RSS budget ({rss // 1048576} MiB)"
                        break
                    pressure = _cgroup_pressure()
                    if pressure and pressure[0] > pressure[1] - reserve:
                        mb = 1048576
                        reason = (
                            f"{label} stopped before cgroup memory exhaustion "
                            f"(active={pressure[0] // mb} MiB, limit={pressure[1] // mb} MiB, "
                            f"reserved={reserve // mb} MiB, ffmpeg_rss={rss // mb} MiB, "
                            f"api_rss={_rss_bytes(os.getpid()) // mb} MiB)"
                        )
                        break
                time.sleep(0.1)
            if reason:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        out.seek(0)
        err.seek(0, 2)
        err.seek(max(0, err.tell() - 16384))
        result = subprocess.CompletedProcess(command, process.returncode, out.read(), err.read())
        if reason:
            raise RuntimeError(reason + "; use the memory figures to adjust the workload or host")
        return result
