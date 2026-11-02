"""Insert measured narration slots without moving or overwriting source dialogue.

The director's timestamps refer to the selected body. This adapter expands them
once, immediately before rendering. Receipts retain original source coordinates.
No model-generated path or command is executed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import wave

from ai_compose_schema import BaseSegment, GraphicPlan, VoiceoverPlan


@dataclass
class PreparedProgram:
    plan: object
    layers: list
    audio: list | None
    mapping: list


def prepare_sections(request, plan, layers, root: Path, warnings):
    """Synthesize first, measure durations, then remap body cues around inserts."""
    if request.editor.narration == 'off' or not plan.narration_sections:
        return PreparedProgram(plan, layers, None, [])
    from ai_compose_engine import _synthesize_voiceovers, ComposeError

    slots = {}
    for item in plan.narration_sections:
        boundary = (0 if item.position == 'intro' else len(plan.base_segments)
                    if item.position == 'outro' else (item.after_segment + 1
                    if item.after_segment is not None else max(1, len(plan.base_segments)//2)))
        if item.position == 'bridge' and not 0 < boundary < len(plan.base_segments):
            raise ComposeError('A bridge needs a boundary between two source segments')
        if boundary in slots:
            raise ComposeError('Two narration sections cannot occupy the same boundary')
        slots[boundary] = item

    prepared = {}
    budget = 0.0
    for boundary, item in sorted(slots.items()):
        import os
        identity = item.text + '\0' + (request.voiceover.voice or os.getenv('AI_COMPOSE_TTS_MODEL', ''))
        cache = root / 'narration' / hashlib.sha256(identity.encode()).hexdigest()[:20]
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / 'voiceover_00.wav'
        if not cached.is_file():
            speech = plan.model_copy(update={'voiceovers': [VoiceoverPlan(text=item.text, volume=request.voiceover.volume)]})
            enabled = request.model_copy(deep=True)
            enabled.voiceover.mode = 'full'
            generated = _synthesize_voiceovers(enabled, speech, cache, warnings)
            if not generated:
                raise ComposeError('Narration synthesis failed; required sections cannot be skipped')
        with wave.open(str(cached), 'rb') as wav:
            seconds = wav.getnframes() / wav.getframerate()
        if seconds < .1:
            raise ComposeError('Narration synthesis returned empty audio')
        # Whole frames plus a short breathing space; no guessed speech duration.
        duration = math.ceil((seconds + .25) * request.fps) / request.fps
        budget += duration
        prepared[boundary] = (item, cached, duration)
    if budget > request.editor.narration_budget_seconds + .001:
        raise ComposeError(f'Narration takes {budget:.2f}s; shorten scripts to fit the '
                           f'{request.editor.narration_budget_seconds:.2f}s narration budget')

    expanded = plan.model_copy(deep=True)
    expanded.base_segments, expanded.voiceovers = [], []
    mapping, audio, cards = [], [], []
    cursor = body = 0.0
    for index in range(len(plan.base_segments) + 1):
        if index in prepared:
            item, path, duration = prepared[index]
            expanded.base_segments.append(BaseSegment(source_start=0, source_end=duration,
                kind='narration', audio_gain=0, reason=f'{item.position}_narration'))
            expanded.voiceovers.append(VoiceoverPlan(text=item.text, start=cursor, volume=request.voiceover.volume))
            audio.append((path, cursor, request.voiceover.volume, 'voiceover'))
            # Captions carry the complete script. The card is a short design anchor.
            title = ' '.join(item.text.split()[:8])[:72]
            cards.append(GraphicPlan(kind='hook' if item.position == 'intro' else 'outro'
                if item.position == 'outro' else 'callout', start=cursor, end=cursor+duration,
                text=title, placement='center', animation='slide_up', style='accent',
                reason=f'{item.position}_narration', evidence_refs=item.evidence_refs))
            mapping.append({'kind':'narration','position':item.position,'program_start':cursor,
                            'program_end':cursor+duration,'text':item.text,'audio_path':str(path)})
            cursor += duration
        if index < len(plan.base_segments):
            segment = plan.base_segments[index]
            duration = segment.source_end - segment.source_start
            expanded.base_segments.append(segment.model_copy(deep=True))
            mapping.append({'kind':'source','source_start':segment.source_start,'source_end':segment.source_end,
                'body_start':body,'body_end':body+duration,'program_start':cursor,'program_end':cursor+duration})
            cursor += duration
            body += duration

    def pieces(start, end):
        """Split an interval at inserted sections, so overlays never leak into VO."""
        pending = None
        for clip in mapping:
            if clip['kind'] != 'source':
                continue
            a, b = max(start, clip['body_start']), min(end, clip['body_end'])
            if b-a > .02:
                current = (clip['program_start']+a-clip['body_start'], clip['program_start']+b-clip['body_start'])
                if pending is not None and abs(current[0]-pending[1]) < .001:
                    pending = (pending[0],current[1])
                else:
                    if pending is not None:
                        yield pending
                    pending = current
        if pending is not None:
            yield pending

    def point(at):
        for clip in mapping:
            if clip['kind'] == 'source' and clip['body_start'] <= at < clip['body_end']:
                return clip['program_start'] + at - clip['body_start']
        raise ComposeError(f'Audio cue {at}s is outside the body timeline')

    expanded.graphics = []
    for cue in plan.graphics:
        spans = list(pieces(cue.start, cue.end))
        if len(spans) > 1:
            readable = [(a,b) for a,b in spans if b-a >= max(.8, len(cue.text)/24)]
            if not readable:
                raise ComposeError('Graphic crosses narration and has no readable interval; shorten or move: '+cue.text[:100])
            # A single readable appearance beats repeating the full quote as a flash.
            spans = [max(readable, key=lambda span: span[1]-span[0])]
            warnings.append('Kept one readable graphic interval across narration: '+cue.text[:100])
        expanded.graphics.extend(cue.model_copy(update={'start':a,'end':b}) for a,b in spans)
    expanded.graphics += cards
    expanded.hook_text = expanded.outro_text = None
    expanded.sfx_events = [cue.model_copy(update={'at':point(cue.at)}) for cue in plan.sfx_events if cue.at < body]
    expanded.asset_requests = []  # retrieval already happened on the body timeline
    mapped_layers = [layer.model_copy(update={'start':a,'end':b, 'source_offset':layer.source_offset + max(0., next(c['body_start']+a-c['program_start'] for c in mapping if c['kind']=='source' and c['program_start'] <= a < c['program_end'])-layer.start)}) for layer in layers for a,b in pieces(layer.start,layer.end)]
    return PreparedProgram(expanded, mapped_layers, audio, mapping)
