"""Render a repeatable compositor demo with no model, provider key or upload.

This deliberately authored plan proves renderer execution, not autonomous
editorial judgment. The notebook's prompt-based run tests that separately.
"""
from pathlib import Path
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_compose_schema import AIComposeRequest, CompositionPlan, BaseSegment, GraphicPlan, ResolvedLayer, SfxEvent
from ai_compose_engine import render_composition, probe_media


def render(root):
    root.mkdir(parents=True, exist_ok=True)
    source = root/'source.mp4'
    asset = root/'support.mp4'
    for path, color in ((source,'0x142535'),(asset,'0x116b73')):
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i',
            f'color=c={color}:s=540x960:r=30,drawgrid=w=90:h=90:t=1:c=white@0.08',
            '-t','10','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
    request = AIComposeRequest(source_path=str(source),target_seconds=10,output_width=540,output_height=960,
        editor={'enabled':True,'min_graphics':1,'music':'procedural'},fetch_assets=False,
        editorial_policy={'camera_motion':'dynamic','motion_graphics':'hero'},refinement_iterations=0)
    plan = CompositionPlan(base_segments=[BaseSegment(source_start=0,source_end=10)],graphics=[
        GraphicPlan(kind='hook',start=0,end=2.6,text='Make every edit earn its place.',placement='top',animation='slide_down'),
        GraphicPlan(kind='checklist',start=2.8,end=6.4,text='Choose the story|Show the evidence|Keep speech clear',placement='top'),
        GraphicPlan(kind='badge',start=6.8,end=9.8,text='PLAN • RENDER • REVIEW',placement='top'),
    ],sfx_events=[SfxEvent(kind='whoosh',at=2.8,volume=.14),SfxEvent(kind='click',at=6.8,volume=.1)])
    words = [dict(word=w,start=i*.45,end=i*.45+.4) for i,w in enumerate(
        'Every visual supports the story. The video keeps its voice. Clear captions guide attention. Review the finished result.'.split())]
    layer = ResolvedLayer(asset_id='demo',source='generated_demo',local_path=str(asset),query='demo',kind='video',
        start=3,end=6.5,x=.08,y=.48,width=.84,height=.23,animation='slide_up')
    result = render_composition(source,probe_media(source),request,plan,[layer],words,root,1,[])
    (root/'showcase_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    render(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/sh01-showcase'))
