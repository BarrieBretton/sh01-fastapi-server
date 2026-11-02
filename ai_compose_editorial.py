"""Evidence-bounded editorial checks; no model score is a virality prediction."""
from __future__ import annotations

import math
import re


def speaker_safe(request):
    return request.editor.layout == 'speaker_safe' or (
        request.editor.layout == 'auto' and request.editor.quality_profile == 'polished')


def normalized(text):
    return ' '.join(re.findall(r"\w+", text.casefold()))


def speech_units(words):
    """ASR sentence/pause boundaries, not fixed-duration highlight samples."""
    units, current = [], []
    for word in sorted(words, key=lambda w: w['start']):
        if current and word['start']-current[-1]['end'] >= .65:
            units.append(current)
            current = []
        current.append(word)
        if re.search(r'[.!?][\"\u2019\u201d]*$', word['word'].strip()):
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


def narrative_errors(request, plan, words):
    if request.editor.quality_profile != 'polished':
        return []
    errors = []
    source = [s for s in plan.base_segments if s.kind == 'source']
    units = speech_units(words)
    # Only discontinuous joins need sentence boundaries. Continuous cuts are legal.
    for i, segment in enumerate(source):
        selected = [w for w in words if segment.source_start <= w['start'] < segment.source_end]
        if not selected:
            continue
        start_required = i == 0 or abs(source[i-1].source_end-segment.source_start) > .1
        end_required = i == len(source)-1 or abs(segment.source_end-source[i+1].source_start) > .1
        if start_required and not any(abs(selected[0]['start']-u[0]['start']) < .05 for u in units):
            errors.append(f'Narrative cut {i} starts mid-sentence at {segment.source_start:.2f}; select a complete speech unit')
        if end_required and not any(abs(selected[-1]['end']-u[-1]['end']) < .05 and segment.source_end >= u[-1]['end']-.02 for u in units):
            errors.append(f'Narrative cut {i} ends mid-sentence at {segment.source_end:.2f}; preserve the payoff')
    if sum(s.source_end-s.source_start for s in source) > request.target_seconds+.05:
        errors.append('Narrative exceeds body time budget; choose fewer COMPLETE speech units rather than truncating the last sentence')
    scripts = [normalized(s.text) for s in plan.narration_sections]
    if len(scripts) != len(set(scripts)):
        errors.append('Narration repeats the same script; intro, bridge and outro must perform distinct narrative jobs')
    for section in plan.narration_sections:
        if len(section.text.split()) < 4:
            errors.append(f'{section.position} narration is a label, not a meaningful spoken sentence')
        refs = [re.fullmatch(r'source:(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', ref) for ref in section.evidence_refs]
        if not any(m and float(m[1]) < float(m[2]) and any(
                max(s.source_start,float(m[1])) < min(s.source_end,float(m[2])) for s in source) for m in refs):
            errors.append(f'{section.position} narration needs a source:start-end reference overlapping selected speech')
    return errors


def parse_source_observations(value, frames):
    """Source descriptions need frame evidence, not an ungrounded quality score."""
    evidence = value.get('observations') if isinstance(value, dict) else None
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Source analysis needs timestamped observations, not a score or revision template')
    placeholders = {
        'timestamped visible subjects actions readable text and suggested safe overlay regions',
        'specific visible fact', 'describe what is visible', 'visible subjects and actions',
    }
    checked = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError('Each source observation must be an object')
        raw_time = item.get('time')
        if isinstance(raw_time, bool) or not isinstance(raw_time, (int, float)):
            raise ValueError('Source observation time must be a numeric supplied frame timestamp')
        t = float(raw_time)
        if not math.isfinite(t) or not any(abs(t-f['time']) <= .15 for f in frames):
            raise ValueError('Source observation timestamp was not one of the supplied frames')
        description = item.get('description')
        if not isinstance(description, str) or not 12 <= len(description.strip()) <= 600:
            raise ValueError('Each source frame needs a concrete visible description of 12-600 characters')
        description = description.strip()
        if normalized(description) in placeholders:
            raise ValueError('Source description repeats a placeholder; describe the actual frame')
        if re.search(r'\b(audio|audible|inaudible|background noise|sound quality|voice quality|music volume)\b', description, re.I):
            raise ValueError('Source stills cannot establish audio quality; describe visible evidence only')
        checked.append({'time': t, 'description': description})
    if any(not any(abs(o['time']-f['time']) <= .15 for o in checked) for f in frames):
        raise ValueError('Source analysis must describe every supplied frame')
    return {'performed': True, 'observations': checked, 'issues': [],
            'revision': '\n'.join(f"{o['time']:.2f}s: {o['description']}" for o in checked),
            'scope': 'sampled source stills; no audio, motion or quality score assessment'}


def parse_visual_review(value, frames, threshold):
    """Require timestamped visual observations; reject unsupported audio scoring."""
    score = float(value['score'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Review score must be finite and between 0 and 1')
    evidence = value.get('observations')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Review needs timestamped observations from supplied frames')
    checked = []
    allowed = {'legibility','occlusion','framing','asset_relevance','design'}
    for obs in evidence:
        if not isinstance(obs, dict) or obs.get('category') not in allowed:
            raise ValueError('Review category must describe visible pixels, not audio or temporal transitions')
        t = float(obs['time'])
        if not math.isfinite(t) or not any(abs(t-f['time']) <= .15 for f in frames):
            raise ValueError('Review timestamp was not one of the supplied frames')
        description = str(obs.get('observation','')).strip()
        action = str(obs.get('action','')).strip()
        if not description or not isinstance(obs.get('problem'), bool):
            raise ValueError('Each observation needs visible evidence and a boolean problem')
        if re.search(r'\b(audio|audible|inaudible|background noise|sound quality|voice quality|music volume)\b', description+' '+action, re.I):
            raise ValueError('Still-frame review cannot judge audio; remove unsupported assessment and rescore')
        if obs['problem'] and not action:
            raise ValueError('Each visible problem needs a concrete correction')
        checked.append(dict(obs, time=t))
    if any(not any(abs(o['time']-f['time']) <= .15 for o in checked) for f in frames):
        raise ValueError('Review must include observations for every supplied frame')
    issues = [f"{o['time']:.2f}s {o['category']}: {o['observation']}" for o in checked if o['problem']]
    return {'performed':True,'score':score,'ok':score >= threshold and not issues,
            'issues':issues,'observations':checked,
            'revision':'\n'.join(str(o['action']) for o in checked if o['problem']),
            'scope':'sampled still frames; audio and transition motion not assessed'}


def aggregate_review(review, outputs, visual, request):
    """All failures have a truthful top-level status and their own cause."""
    result = dict(review, errors=list(review['errors']))
    result['body_review'] = dict(review)
    result['render_issues'] = [o['quality_review'] for o in outputs]
    result['visual'] = visual
    technical_ok = bool(outputs) and all(o['quality_review']['technical']['ok'] for o in outputs)
    editorial_ok = review['ok'] and all(o['quality_review']['editorial']['ok'] for o in outputs)
    for o in outputs:
        result['errors'].extend(o['quality_review']['editorial']['errors'])
        if not o['quality_review']['technical']['ok']:
            result['errors'].append(f"Technical render review failed for variant {o.get('variant', '?')}")
    required_ok = not any(e.startswith('Required feature absent') for e in result['errors'])
    visual_ok = visual.get('performed',False) and visual.get('ok', visual.get('score',0) >= request.editor.visual_review_threshold)
    if request.editor.visual_review_required and not visual_ok:
        result['errors'].append(f'Required visual review failed (threshold {request.editor.visual_review_threshold:.2f}); inspect observations')
    accepted = technical_ok and required_ok and (editorial_ok or not request.editor.strict_requirements) and (
        visual_ok or not request.editor.visual_review_required)
    result['errors'] = list(dict.fromkeys(result['errors']))
    result['ok'] = accepted and not result['errors']
    result['accepted'] = accepted
    result['relaxed_editorial_gate'] = accepted and not editorial_ok
    result['score'] = min([review['score']] + [o['quality_review']['editorial']['score'] for o in outputs])
    result['duration'] = outputs[0]['duration'] if outputs else review['duration']
    return result
