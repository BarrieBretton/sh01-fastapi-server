"""Measured-slot regressions; real FFmpeg, synthetic voices, no network or GPU."""
import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from ai_compose_schema import AIComposeRequest, BaseSegment, CompositionPlan, GraphicPlan, NarrationSection, ResolvedLayer, SfxEvent
from ai_compose_sections import prepare_sections
from ai_compose_editor import editorial_review, run_editor
from ai_compose_engine import render_composition, probe_media
from ai_compose_motion import map_words
from ai_compose_commons import parse_candidates


class SectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.request = AIComposeRequest(source_path=str(self.root/'source.mp4'),target_seconds=12,
            output_width=270,output_height=480,fps=24,refinement_iterations=0,fetch_assets=False,
            editor={'enabled':True,'narration':'intro_bridge_outro','narration_budget_seconds':4,
                    'min_graphics':0,'music':'off','transition':'dissolve'},
            editorial_policy={'source_audio':'preserve','camera_motion':'static'},sfx={'mode':'off'})
        self.plan = CompositionPlan(base_segments=[BaseSegment(source_start=0,source_end=3),BaseSegment(source_start=3,source_end=6)],
            narration_sections=[NarrationSection(position='intro',text='The idea.'),
                NarrationSection(position='bridge',after_segment=0,text='The evidence.'),
                NarrationSection(position='outro',text='The result.')])
        self.words=[{'word':s,'start':t,'end':t+.3} for s,t in [('Start.',.3),('Middle.',2.),('Finish.',4.)]]

    def tearDown(self):
        self.tmp.cleanup()

    def voice(self, request, plan, root, warnings):
        import math, struct
        dest=root/'voiceover_00.wav'
        with wave.open(str(dest),'wb') as f:
            f.setparams((1,2,24000,0,'NONE','not compressed'))
            f.writeframes(b''.join(struct.pack('<h',int(9000*math.sin(2*math.pi*880*n/24000))) for n in range(18000)))
        return [(dest,0,1,'voiceover')]

    def prepare(self,layers=None):
        with patch('ai_compose_engine._synthesize_voiceovers',side_effect=self.voice):
            return prepare_sections(self.request,self.plan,layers or [],self.root,[])

    def test_source_caption_timestamps_skip_insertions(self):
        prepared=self.prepare()
        self.assertEqual([s.kind for s in prepared.plan.base_segments],['narration','source','narration','source','narration'])
        mapped=map_words(self.words,prepared.plan.base_segments)
        self.assertEqual([round(w['start'],2) for w in mapped],[1.3,3.,6.])
        self.assertEqual([a[1] for a in prepared.audio],[0,4,8])
        self.assertEqual([c['source_start'] for c in prepared.mapping if c['kind']=='source'],[0,3])

    def test_overlay_crossing_bridge_is_split(self):
        layer=ResolvedLayer(asset_id='one',local_path='image.png',source='local',query='test',kind='image',
            start=2,end=4,x=.1,y=.3,width=.5,height=.3)
        prepared=self.prepare([layer])
        self.assertEqual([(a.start,a.end) for a in prepared.layers],[(3,4),(5,6)])

    def test_contiguous_body_cuts_do_not_duplicate_layers(self):
        self.plan.narration_sections=self.plan.narration_sections[:1]
        layer=ResolvedLayer(asset_id='one',local_path='image.png',source='local',query='test',kind='image',
            start=2,end=4,x=.1,y=.3,width=.5,height=.3)
        prepared=self.prepare([layer])
        self.assertEqual([(a.start,a.end) for a in prepared.layers],[(3,5)])

    def test_oversize_narration_fails_before_render(self):
        self.request.editor.narration_budget_seconds=2
        with self.assertRaisesRegex(RuntimeError,'shorten scripts'):
            self.prepare()

    def test_bridge_requires_two_source_clips(self):
        self.plan.base_segments=self.plan.base_segments[:1]
        with self.assertRaisesRegex(RuntimeError,'bridge needs'):
            self.prepare()

    def test_missing_required_audio_never_counts_as_complete(self):
        self.request.editor.required_features=['voiceover','sfx','music','camera_motion','transitions','graphics','supplementary_media','captions']
        report=editorial_review(self.request,self.plan,[],self.words,{})
        self.assertFalse(report['ok'])
        self.assertEqual(sum('Required feature absent' in x for x in report['errors']),8)

    def test_missing_requested_section_is_an_error(self):
        self.plan.narration_sections=self.plan.narration_sections[:1]
        report=editorial_review(self.request,self.plan,[],self.words)
        self.assertTrue(any('Missing narration sections' in e for e in report['errors']))

    def test_off_conflict_fails_before_acquiring_source(self):
        self.request.editor.required_features=['music']
        with patch('ai_compose_editor.capabilities',return_value={'visual_review':False,'tts':True}), \
             patch('ai_compose_engine.acquire_source') as acquire:
            with self.assertRaisesRegex(RuntimeError,'explicit off controls'):
                asyncio.run(run_editor(self.request,{'job_id':'conflict'},None))
            acquire.assert_not_called()

    def test_commons_license_and_metadata_are_not_invented(self):
        payload={'query':{'pages':{'1':{'pageid':1,'title':'File:Astronomy moon.jpg','imageinfo':[{
            'mime':'image/jpeg','width':800,'height':600,'size':1000,'url':'https://upload.wikimedia.org/moon.jpg',
            'descriptionurl':'https://commons.wikimedia.org/wiki/File:Moon.jpg',
            'extmetadata':{'LicenseShortName':{'value':'CC BY-SA 4.0'},'ImageDescription':{'value':'<b>Moon</b> crater'},
                           'Artist':{'value':'Example photographer'}}}]}}}}
        items=parse_candidates(payload,'image')
        self.assertEqual(len(items),1)
        self.assertIn('Moon',items[0]['description'])
        payload['query']['pages']['1']['imageinfo'][0]['extmetadata']['LicenseShortName']['value']='All rights reserved'
        self.assertEqual(parse_candidates(payload,'image'),[])

    def test_assembly_does_not_apply_old_voiceover_overlay(self):
        from ai_compose_schema import VoiceoverPlan
        self.plan.voiceovers=[VoiceoverPlan(text='Old overlay',start=2)]
        result=self.prepare()
        self.assertEqual([v.text for v in result.plan.voiceovers],['The idea.','The evidence.','The result.'])
        self.assertFalse(any(v.start==2 for v in result.plan.voiceovers))

    def test_empty_asr_is_allowed_only_by_explicit_flag(self):
        import types, sys
        from media_transcribe import transcribe
        class EmptyModel:
            def __init__(self,*args,**kwargs): pass
            def transcribe(self,*args,**kwargs): return (iter([]),None)
        with patch.dict(sys.modules,{'faster_whisper':types.SimpleNamespace(WhisperModel=EmptyModel)}):
            with self.assertRaisesRegex(RuntimeError,'no word timestamps'):
                transcribe(Path('silent.wav'),None)
            self.assertEqual(transcribe(Path('silent.wav'),None,allow_empty=True),[])

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_editor_pipeline_publishes_the_expanded_mapping(self):
        source=Path(self.request.source_path)
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=navy:s=270x480:r=24',
            '-f','lavfi','-i','sine=frequency=220:sample_rate=48000','-t','6','-c:v','libx264',
            '-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True)
        self.plan.base_segments[0].audio_gain=0  # preserve policy must override model muting
        state={'job_id':'sections-pipeline','request_id':'offline-test'}
        async def update(**values): state.update(values)
        capabilities={key:False for key in ('local_assets','pexels','pixabay','unsplash','wikimedia','music_library','visual_review')}
        capabilities['tts']=True
        with patch.dict(os.environ,{'MEDIA_LOCAL_SOURCE_DIR':str(self.root),'AI_COMPOSE_WORK_DIR':str(self.root)}), \
             patch('ai_compose_editor.capabilities',return_value=capabilities), \
             patch('ai_compose_engine.transcribe_chunked',return_value=self.words), \
             patch('ai_compose_engine._synthesize_voiceovers',side_effect=self.voice), \
             patch('ai_compose_engine.LocalDirector.plan',return_value=(self.plan,[])):
            asyncio.run(run_editor(AIComposeRequest.model_validate(self.request.model_dump()),state,update))
        self.assertEqual(state['status'],'complete')
        self.assertEqual(state['plan']['workflow'],'sh01/automatic-editor@2')
        self.assertEqual(len(state['plan']['section_mapping']),5)
        self.assertTrue(all(c['audio_gain']==1 for c in state['plan']['edit_plan']['source_clips'] if c['kind']=='source'))
        self.assertEqual(state['plan']['render_features'][0]['narration_sections_rendered'],3)
        self.assertAlmostEqual(state['outputs'][0]['duration'],9,delta=.06)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_real_render_all_features_preserves_dialogue_and_caption_windows(self):
        import numpy as np
        from PIL import Image
        source=Path(self.request.source_path)
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=0x18364a:s=270x480:r=24',
            '-f','lavfi','-i','sine=frequency=220:sample_rate=48000','-t','6','-c:v','libx264',
            '-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True)
        image=self.root/'support.png'; Image.new('RGB',(120,120),'red').save(image)
        layer=ResolvedLayer(asset_id='red',local_path=str(image),source='local',query='red',kind='image',
            start=2,end=4,x=.1,y=.3,width=.5,height=.3)
        self.request.editor.music='procedural'
        self.request.editorial_policy.camera_motion='dynamic'
        self.request.sfx.mode='adaptive'
        self.request.editor.progress_bar=False
        self.plan.sfx_events=[SfxEvent(kind='click',at=2,volume=.1)]
        self.request.editor.required_features=['voiceover','sfx','music','camera_motion','transitions','graphics','supplementary_media','captions']
        prepared=self.prepare([layer])
        voice_words=[{'word':'Narration.','start':.1,'end':.6}]
        with patch('ai_compose_engine.transcribe_chunked',return_value=voice_words):
            rendered=render_composition(source,probe_media(source),self.request,prepared.plan,prepared.layers,
                self.words,self.root,1,[],prepared_voiceovers=prepared.audio)
        self.assertAlmostEqual(rendered['duration'],9,delta=.06)
        self.assertTrue(editorial_review(self.request,prepared.plan,prepared.layers,self.words,rendered['render_features'])['ok'])
        self.assertEqual(rendered['render_features']['narration_sections_rendered'],3)
        text=(self.root/'variant_01.ass').read_text()
        self.assertIn('0:00:01.30',text)
        self.assertIn('0:00:06.00',text)
        # Frequencies independently prove source audio remains and cannot bleed into narration.
        def magnitude(at,freq):
            pcm=subprocess.check_output(['ffmpeg','-v','error','-ss',str(at),'-i',rendered['local_path'],
                '-t','0.3','-ac','1','-ar','16000','-f','f32le','-'])
            signal=np.frombuffer(pcm,dtype='<f4')
            spectrum=abs(np.fft.rfft(signal*np.hanning(len(signal))))
            bins=np.fft.rfftfreq(len(signal),1/16000)
            return max(spectrum[abs(bins-freq)<8])
        self.assertGreater(magnitude(1.5,220),magnitude(1.5,880)*8)
        self.assertGreater(magnitude(.2,880),magnitude(.2,220)*8)
        self.assertGreater(magnitude(5.5,220),magnitude(5.5,880)*8)
        (self.root/'validated.json').write_text(json.dumps(rendered))
        # Save the reproducible test fixture only when explicitly requested.
        destination=os.getenv('SH01_TEST_DEMO_DIR')
        if destination:
            target=Path(destination);target.mkdir(parents=True,exist_ok=True)
            shutil.copy2(rendered['local_path'],target/'section-render-demo.mp4')
            shutil.copy2(self.root/'variant_01.ass',target/'section-render-demo.ass')
            (target/'section-render-features.json').write_text(json.dumps(rendered['render_features'],indent=2))


if __name__=='__main__':
    unittest.main()
