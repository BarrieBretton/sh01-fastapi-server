"""One-shot faster-whisper worker. Its native memory returns to the OS on exit."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def transcribe(audio: Path, language: str | None) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        os.getenv("MEDIA_CLIP_WHISPER_MODEL", "tiny").strip() or "tiny",
        device=os.getenv("MEDIA_CLIP_WHISPER_DEVICE", "cpu").strip() or "cpu",
        compute_type=os.getenv("MEDIA_CLIP_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8",
        download_root=os.getenv("MEDIA_CLIP_WHISPER_CACHE_DIR", "/tmp/faster-whisper").strip() or "/tmp/faster-whisper",
        cpu_threads=max(1, int(os.getenv("MEDIA_CLIP_WHISPER_CPU_THREADS", "1"))),
        num_workers=1,
    )
    segments, _ = model.transcribe(
        str(audio), language=language or None, beam_size=1, word_timestamps=True,
        vad_filter=os.getenv("MEDIA_CLIP_WHISPER_VAD", "true").lower() in {"true", "1", "yes", "on"},
        condition_on_previous_text=False, temperature=0.0,
    )
    words = [
        {"word": str(w.word or "").strip(), "start": float(w.start), "end": float(w.end),
         "probability": float(w.probability) if w.probability is not None else None}
        for segment in segments for w in segment.words or []
        if str(w.word or "").strip() and w.start is not None and w.end is not None
    ]
    if not words:
        raise RuntimeError("faster-whisper returned no word timestamps")
    return words


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m media_transcribe AUDIO OUTPUT LANGUAGE")
    audio, output, language = sys.argv[1:]
    Path(output).write_text(json.dumps(transcribe(Path(audio), language or None)), encoding="utf-8")


if __name__ == "__main__":
    main()
