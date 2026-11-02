"""Frame-timed vector graphics and kinetic captions, rendered locally by libass.

No browser, generated executable code, or frame-sized image sequences. All
coordinates are normalized to the existing 1080x1920 ASS design canvas.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path


def escape(text):
    # Untrusted transcript/model copy must never inject ASS override commands.
    return str(text).replace("\\", "／").replace("{", "(").replace("}", ")").replace("\n", " ")


def clock(value):
    n = max(0, round(value * 100))
    return f"{n // 360000}:{n // 6000 % 60:02}:{n // 100 % 60:02}.{n % 100:02}"


def map_words(words, segments):
    result, cursor = [], 0.0
    for segment in segments:
        if getattr(segment, 'kind', 'source') == 'narration':
            cursor += segment.source_end-segment.source_start
            continue
        for word in words:
            a, b = float(word["start"]), float(word["end"])
            if a >= segment.source_start and a < segment.source_end:
                end = min(b, segment.source_end)
                if end > a:
                    result.append(dict(word, start=cursor+a-segment.source_start,
                                       end=cursor+end-segment.source_start))
        cursor += segment.source_end-segment.source_start
    return sorted(result, key=lambda w: w["start"])


def write_motion_ass(path: Path, request, plan, edit_plan, words):
    """Render cards, accent rails, animated progress and per-word emphasis.

    Returns counts from events actually emitted, rather than plan lengths.
    Captions and graphics have separate tracks and explicit off switches.
    """
    theme = {"editorial": "&H00D7F066", "bold": "&H003BDFFF", "minimal": "&H00FFFFFF"}[request.editor.theme]
    lines = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2",
             "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
             "Style: Default,DejaVu Sans,62,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,3,0,5,80,80,120,1",
             "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    counts = dict(graphics_rendered=0, caption_events=0, shape_events=0, progress_bars_rendered=0)

    def event(layer, start, end, tags, text):
        start, end = max(0, start), min(edit_plan.duration, end)
        if end > start:
            lines.append(f"Dialogue: {layer},{clock(start)},{clock(end)},Default,,0,0,0,,{{{tags}}}{text}")

    def rect(start, end, x, y, w, h, color, extra="", layer=5):
        event(layer, start, end, f"\\an7\\pos({x},{y})\\bord0\\shad0\\1c{color}\\p1"+extra,
              f"m 0 0 l {w} 0 {w} {h} 0 {h}")
        counts["shape_events"] += 1

    if request.captions_enabled and plan.captions_enabled and request.editor.caption_style != "off":
        mapped = map_words(words, plan.base_segments)
        groups, group = [], []
        for word in mapped:
            if group and (len(group) >= 4 or len(" ".join(str(w['word']) for w in group)) > 24
                          or word["start"]-group[-1]["end"] > .5):
                groups.append(group); group = []
            group.append(word)
        if group:
            groups.append(group)
        for group in groups:
            # Fixed caption band, clear of the supporting visual band.
            a, b = group[0]["start"], group[-1]["end"]
            style = request.editor.caption_style
            indices = range(len(group)) if style != "phrase" else [0]
            for i in indices:
                start = a if style == "phrase" else group[i]["start"]
                end = b if style == "phrase" or i == len(group)-1 else group[i+1]["start"]
                pieces = []
                for j, word in enumerate(group):
                    active = style != "phrase" and (j == i or style == "karaoke" and j <= i)
                    pieces.append((f"{{\\1c{theme}}}" if active else "{\\1c&H00FFFFFF}") + escape(word['word']).strip())
                event(20, start, end, "\\an5\\pos(540,1510)\\fs62", " ".join(pieces))
                counts["caption_events"] += 1

    for cue in edit_plan.graphics:
        # Fitted card text: long text wraps and shrinks, never silently truncates.
        text = escape(cue.text)
        if cue.reason in {'intro_narration','bridge_narration','outro_narration'}:
            # Dedicated title composition, with large typography and restrained
            # vector movement. It occupies its own silent-base narration slot.
            ms = int((cue.end-cue.start)*1000)
            section = {'intro_narration':'CONTEXT','bridge_narration':'THE CONNECTION',
                       'outro_narration':'THE TAKEAWAY'}[cue.reason]
            rect(cue.start,cue.end,0,0,1080,1920,'&H002C1B10',layer=1)
            rect(cue.start,cue.end,760,-120,420,2160,theme,
                 f'\\1a&HED\\frz-12\\t(0,{ms},\\frz-6)',2)
            for i in range(6):
                rect(cue.start,cue.end,100,1160+i*36,880,2,theme,
                     f'\\1a&HD8\\fscx40\\t(0,{ms},\\fscx100)',2)
            rect(cue.start,cue.end,100,570,90,8,theme,'\\fad(120,120)',5)
            event(6,cue.start,cue.end,'\\an7\\pos(100,490)\\fs30\\fsp5\\bord0'+f'\\1c{theme}\\fad(160,120)',section)
            wrapped=textwrap.wrap(text,19,break_long_words=True)
            event(7,cue.start,cue.end,'\\an7\\move(100,690,100,640,0,320)\\fs88\\bord0\\fad(160,120)',r'\N'.join(wrapped))
            counts['graphics_rendered'] += 1
            continue
        labels = [part.strip() for part in text.split('|') if part.strip()]
        if cue.kind in {'checklist', 'timeline'} and len(labels) <= 5 and all(len(label)<=43 for label in labels):
            y = 220 if cue.placement in {'top','upper_left'} else 1000
            h = 70 + len(labels)*80
            rect(cue.start,cue.end,80,y,920,h,'&H00201A16','\\1a&H24\\fad(100,140)')
            if cue.kind == 'timeline':
                rect(cue.start,cue.end,115,y+50,5,max(8,h-90),theme,'\\fad(100,140)',6)
            for index,label in enumerate(labels):
                a = cue.start + min(index*.2,(cue.end-cue.start)*.4)
                rect(a,cue.end,103,y+42+index*80,26,26,theme,'\\fad(100,140)',6)
                event(7,a,cue.end,f'\\an4\\pos(155,{y+55+index*80})\\fs42\\bord0\\fad(100,120)',label)
            counts['graphics_rendered'] += 1
            continue
        if cue.kind == 'badge' and len(text) <= 40:
            rect(cue.start,cue.end,80,240,920,100,theme,'\\fad(100,140)')
            event(7,cue.start,cue.end,'\\an5\\pos(540,290)\\fs46\\bord0\\1c&H00101010\\fad(100,140)',text)
            counts['graphics_rendered'] += 1
            continue
        wrapped = textwrap.wrap(text, 27, break_long_words=True) or [text]
        font = min(76 if cue.kind == 'stat' else 62, max(30, int(220 / max(2, len(wrapped)))))
        h = min(520, 65 + len(wrapped) * (font + 10))
        y = {"top": 175, "upper_left": 280, "center": 700,
             "lower_left": 1100, "bottom": 1140}.get(cue.placement, 700)
        y = min(y, 1380-h)  # reserve caption band even for a tall card
        x, w = 80, 920
        ms = min(280, int((cue.end-cue.start)*1000*.25))
        anim = "\\fad(100,140)"
        if cue.animation in {"slide_up", "slide_down"}:
            delta = 70 if cue.animation == "slide_up" else -70
            pos = f"\\move(540,{y+h/2+delta},540,{y+h/2},0,{ms})"
        else:
            pos = f"\\pos(540,{y+h/2})"
        if cue.animation == "pop":
            anim += f"\\fscx85\\fscy85\\t(0,{ms},\\fscx100\\fscy100)"
        rect(cue.start, cue.end, x, y, w, h, "&H00201A16", "\\1a&H24\\fad(100,140)")
        rect(cue.start, cue.end, x, y, 9, h, theme, "\\fad(100,140)", 6)
        event(7, cue.start, cue.end, f"\\an5{pos}\\fs{font}\\bord0"+anim, r"\N".join(wrapped))
        counts["graphics_rendered"] += 1
    if request.editor.progress_bar:
        rect(0, edit_plan.duration, 80, 1670, 920, 7, "&H00444444")
        rect(0, edit_plan.duration, 80, 1670, 920, 7, theme,
             f"\\fscx0\\t(0,{int(edit_plan.duration*1000)},1,\\fscx100)", 6)
        counts["progress_bars_rendered"] = 1
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    counts["captions_rendered"] = counts["caption_events"] > 0
    return counts
