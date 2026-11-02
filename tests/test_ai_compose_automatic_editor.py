"""Offline tests: actual FFmpeg composition plus mocked model/ASR boundaries."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ai_compose_schema import AIComposeRequest, CompositionPlan, BaseSegment, GraphicPlan, ResolvedLayer, SfxEvent, VoiceoverPlan
from ai_compose_engine import render_composition, probe_media, _score_candidate
from ai_compose_editor import run_editor, finish_plan, editorial_review


class AutomaticEditorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.request = AIComposeRequest(source_path=str(self.root/'source.mp4'),target_seconds=6,
            output_width=270,output_height=480,refinement_iterations=0,fetch_assets=False,
            editor={'enabled':True,'min_graphics':1,'music':'off'},sfx={'mode':'off'})
        self.words = [dict(word=w,start=i*.4,end=i*.4+.35) for i,w in enumerate(
            'A clear idea. Show the evidence. Explain the result. Keep the meaning.'.split())]
        self.plan = CompositionPlan(base_segments=[BaseSegment(source_start=0,source_end=6)],
            graphics=[GraphicPlan(kind='hook',start=0,end=2.5,text='A clear idea.',placement='top')])

    def tearDown(self):
        self.temp.cleanup()

    def source(self):
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=0x18364a:s=270x480:r=24',
            '-f','lavfi','-i','sine=frequency=220:sample_rate=48000','-t','6','-c:v','libx264',
            '-pix_fmt','yuv420p','-c:a','aac',self.request.source_path],check=True)
        return Path(self.request.source_path)

    def test_retrieval_query_cannot_prove_relevance(self):
        self.assertEqual(_score_candidate('astronomy moon',{'query':'astronomy moon','title':'tomatoes'}),0)

    def test_no_fabricated_fallback_graphic(self):
        result=finish_plan(self.request,CompositionPlan(base_segments=self.plan.base_segments),{'duration':6},self.words)
        self.assertIn('A clear idea.',result.graphics[0].text)

    def test_required_assets_block_plain_clip(self):
        self.request.editor.min_visual_layers=1
        self.assertFalse(editorial_review(self.request,self.plan,[],self.words)['ok'])

    def test_invented_number_fails_editorial_gate(self):
        self.plan.graphics[0].text = 'Results improved by 97%'
        self.assertFalse(editorial_review(self.request,self.plan,[],self.words)['ok'])

    def test_notebook_receipt_survives_restart(self):
        from ai_compose_local_store import NotebookJobStore
        store = NotebookJobStore(self.root/'jobs.sqlite3')
        store.put_job('done','ai_compose','complete',result={'status':'complete','outputs':['ok']})
        store.put_job('running','ai_compose','running',result={'status':'running'})
        reopened = NotebookJobStore(self.root/'jobs.sqlite3')
        reopened.mark_interrupted()
        self.assertEqual(reopened.get_job('done')['result']['outputs'],['ok'])
        self.assertEqual(reopened.get_job('running')['result']['stage'],'interrupted')

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_real_graphics_captions_and_audio(self):
        source=self.source()
        self.request.editor.music='procedural'
        self.request.sfx.mode='adaptive'
        self.plan.sfx_events=[SfxEvent(kind='pop',at=2,volume=.1)]
        result=render_composition(source,probe_media(source),self.request,self.plan,[],self.words,self.root,1,[])
        counts=result['render_features']
        self.assertTrue(counts['captions_rendered'])
        self.assertGreater(counts['caption_events'],3)
        self.assertGreater(counts['shape_events'],1)
        self.assertEqual(counts['sfx_rendered'],1)
        self.assertEqual(counts['music_rendered'],1)
        self.assertGreater(result['duration'],5.9)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_overlay_has_correct_delayed_window(self):
        source=self.source()
        from PIL import Image
        asset=self.root/'red.png'; Image.new('RGB',(100,100),'red').save(asset)
        self.request.captions_enabled=False
        self.request.editor.progress_bar=False
        self.request.editorial_policy.motion_graphics='off'
        self.request.editorial_policy.camera_motion='static'
        layer=ResolvedLayer(asset_id='test',local_path=str(asset),source='local',query='red',kind='image',
            start=3,end=5,x=.2,y=.2,width=.4,height=.4)
        result=render_composition(source,probe_media(source),self.request,self.plan,[layer],[],self.root,1,[])
        def pixel(t):
            raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',result['local_path'],
                '-frames:v','1','-vf','crop=2:2:100:160','-pix_fmt','rgb24','-f','rawvideo','-'])
            return raw[0]
        self.assertLess(pixel(1),100)
        self.assertGreater(pixel(4),180)
        self.assertLess(pixel(5.5),100)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_captions_off_even_when_graphics_on(self):
        source=self.source(); self.request.captions_enabled=False
        result=render_composition(source,probe_media(source),self.request,self.plan,[],self.words,self.root,1,[])
        self.assertFalse(result['render_features']['captions_rendered'])
        self.assertEqual(result['render_features']['caption_events'],0)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_narration_captions_use_generated_audio_timing(self):
        from ai_compose_sound import synthesize
        source=self.source()
        audio=synthesize(self.root/'voice.wav','music',5)
        self.request.voiceover.mode='full'
        self.request.editor.music='procedural'
        self.request.editorial_policy.source_audio='replace'
        self.plan.voiceovers=[VoiceoverPlan(text='A clear idea.')]
        with patch('ai_compose_engine._synthesize_voiceovers',return_value=[(audio,0,1,'voiceover')]), \
             patch('ai_compose_engine.transcribe_chunked',return_value=self.words[:3]):
            result=render_composition(source,probe_media(source),self.request,self.plan,[],[],self.root,1,[])
        self.assertEqual(result['render_features']['voiceovers_rendered'],1)
        self.assertEqual(result['render_features']['caption_events'],3)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_dissolve_preserves_later_source_timing(self):
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=red:s=270x480:r=30:d=2',
            '-f','lavfi','-i','color=green:s=270x480:r=30:d=2','-f','lavfi','-i','color=blue:s=270x480:r=30:d=2',
            '-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]','-map','[v]',
            '-c:v','libx264','-pix_fmt','yuv420p',self.request.source_path],check=True)
        self.request.editorial_policy.editing_intensity='cinematic'
        self.request.editorial_policy.camera_motion='static'
        self.request.editorial_policy.motion_graphics='off'
        self.request.editor.progress_bar=False
        self.request.captions_enabled=False
        self.plan.base_segments=[BaseSegment(source_start=i,source_end=i+2) for i in (0,2,4)]
        source=Path(self.request.source_path)
        result=render_composition(source,probe_media(source),self.request,self.plan,[],[],self.root,1,[])
        self.assertEqual(result['render_features']['transitions_rendered'],2)
        self.assertAlmostEqual(result['duration'],6,delta=.08)
        pixel=subprocess.check_output(['ffmpeg','-v','error','-ss','3.9','-i',result['local_path'],
            '-frames:v','1','-vf','crop=2:2:100:160','-pix_fmt','rgb24','-f','rawvideo','-'])
        self.assertGreater(pixel[1],pixel[2])  # still green, not prematurely blue

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_failed_revision_retains_validated_draft(self):
        self.source(); self.request.refinement_iterations=1
        state={'job_id':'test','request_id':'test'}
        async def update(**values): state.update(values)
        with patch.dict(os.environ,{'MEDIA_LOCAL_SOURCE_DIR':str(self.root),'AI_COMPOSE_WORK_DIR':str(self.root)}), \
             patch('ai_compose_engine.transcribe_chunked',return_value=self.words), \
             patch('ai_compose_engine.LocalDirector.plan',side_effect=[(self.plan,[]),(self.plan,[]),RuntimeError('bad revision')]):
            asyncio.run(run_editor(self.request,state,update))
        self.assertEqual(state['status'],'complete')
        self.assertTrue(Path(state['outputs'][0]['local_path']).is_file())
        self.assertEqual(state['plan']['winning_round'],0)
        self.assertTrue(any('retaining' in w for w in state['warnings']))

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_failed_revision_render_keeps_original_file(self):
        self.source(); self.request.refinement_iterations=1
        state={'job_id':'render-failure','request_id':'test'}
        async def update(**values): state.update(values)
        real_render=render_composition
        def render_once(*args, **kwargs):
            if args[6].name == 'draft_01':
                raise RuntimeError('simulated encoder failure')
            return real_render(*args, **kwargs)
        with patch.dict(os.environ,{'MEDIA_LOCAL_SOURCE_DIR':str(self.root),'AI_COMPOSE_WORK_DIR':str(self.root)}), \
             patch('ai_compose_engine.transcribe_chunked',return_value=self.words), \
             patch('ai_compose_engine.LocalDirector.plan',return_value=(self.plan,[])), \
             patch('ai_compose_engine.render_composition',side_effect=render_once):
            asyncio.run(run_editor(self.request,state,update))
        self.assertEqual(state['status'],'complete')
        self.assertEqual(state['plan']['winning_round'],0)
        self.assertTrue(any('render failed' in w for w in state['warnings']))


if __name__ == '__main__':
    unittest.main()
