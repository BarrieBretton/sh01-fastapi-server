"""Local synthesized transition cues and optional ambient score; bounded RAM."""
from __future__ import annotations
import math
import random
import struct
import wave
from pathlib import Path


def synthesize(path: Path, kind: str, duration: float):
    """Create original PCM audio in one-second chunks. No downloaded samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rate, rng = 24000, random.Random(17)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        total = int(duration * rate)
        for offset in range(0, total, rate):
            chunk = bytearray()
            for n in range(offset, min(total, offset + rate)):
                t = n / rate
                envelope = min(1, t/.02) * min(1, (duration-t)/.08)
                if kind == "music":
                    # Quiet, slowly changing pad. This is a procedural bed, not generative music AI.
                    roots = (130.81, 110., 174.61, 146.83)
                    root = roots[int(t/4) % len(roots)]
                    envelope *= .3 * math.sin(math.pi * (t % 4)/4)**2
                    value = sum(math.sin(2*math.pi*root*r*t) for r in (1, 1.25, 1.5))/3
                elif kind in {"click", "pop", "impact"}:
                    value = math.sin(2*math.pi*(600*t-500*t*t))*math.exp(-t*20)
                else:
                    value = rng.uniform(-1, 1) * math.sin(math.pi*t/duration)**2 * .4
                chunk.extend(struct.pack("<h", int(max(-1,min(1,value*envelope)) * 22000)))
            output.writeframes(chunk)
    return path
