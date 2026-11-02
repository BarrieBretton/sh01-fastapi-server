"""Regressions from the operator's rejected 19-second Kaggle draft."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ai_compose_schema import AIComposeRequest, BaseSegment, CompositionPlan, GraphicPlan, NarrationSection, AssetRequest, ResolvedLayer
from ai_compose_editorial import aggregate_review, narrative_errors, parse_visual_review, speaker_safe
from ai_compose_motion import map_words, write_motion_ass
from ai_compose_sections import prepare_sections
from ai_compose_ir import compile_edit_plan
from ai_compose_policy import resolve_editorial_policy
from ai_compose_engine import LocalDirector, render_composition, probe_media, resolve_assets
from ai_compose_asset_review import select_asset_window


class EditorialPolishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.request = AIComposeRequest(source_path='source.mp4',fetch_assets=False,target_seconds=10,
            output_width=270,output_height=480,fps=24,
            editor={'enabled':True,'quality_profile':'polished','min_graphics':0,'music':'off'},
            editorial_policy={'camera_motion':'static'},sfx={'mode':'off'})
        self.words = [{'word':word,'start':i*.5,'end':i*.5+.4} for i,word in enumerate(
            ['Here','is','the','context.','Here','is','the','payoff.'])]
        self.plan = CompositionPlan(base_segments=[BaseSegment(source_start=0,source_end=4)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_mid_sentence_cuts_but_keeps_contiguous_cuts(self):
        bad = self.plan.model_copy(deep=True)
        bad.base_segments = [BaseSegment(source_start=0,source_end=1),BaseSegment(source_start=2.5,source_end=4)]
        self.assertEqual(len(narrative_errors(self.request,bad,self.words)),2)
        bad.base_segments = [BaseSegment(source_start=0,source_end=1),BaseSegment(source_start=1,source_end=4)]
        self.assertEqual(narrative_errors(self.request,bad,self.words),[])

    def test_repeated_label_scripts_fail_and_distinct_grounded_sentences_pass(self):
        self.plan.narration_sections = [NarrationSection(position=p,text='Decision-Making Process') for p in ['intro','outro']]
        self.assertTrue(any('repeats' in e for e in narrative_errors(self.request,self.plan,self.words)))
        self.plan.narration_sections = [NarrationSection(position='intro',text='Here is the context.',evidence_refs=['source:0-1.9']),
            NarrationSection(position='outro',text='Here is the payoff.',evidence_refs=['source:2-4'])]
        self.assertEqual(narrative_errors(self.request,self.plan,self.words),[])

    def test_director_retries_narrative_defects(self):
        director = LocalDirector(self.request)
        director._planning_words = self.words
        bad = self.plan.model_copy(deep=True)
        bad.base_segments[0].source_end = 1
        answers = iter([bad.model_dump_json(),self.plan.model_dump_json()])
        result = director._generate_valid_plan(lambda _:next(answers),'plan','plan')
        self.assertEqual(result.base_segments[0].source_end,4)

    def test_caption_groups_never_cross_section_identity(self):
        words = [{'word':'Secretary','start':0,'end':.4,'caption_section':'source:0'},
                 {'word':'Decision','start':.45,'end':.8,'caption_section':'narration:0'}]
        policy = resolve_editorial_policy(self.request,{'duration':4,'has_audio':True},self.words)
        edit = compile_edit_plan(self.request,self.plan,[],self.words,policy,{'duration':4,'has_audio':True})
        dest=self.root/'captions.ass'
        write_motion_ass(dest,self.request,self.plan,edit,words)
        events=[line for line in dest.read_text().splitlines() if line.startswith('Dialogue: 20,')]
        self.assertEqual(len(events),2)
        self.assertFalse(any('Secretary' in line and 'Decision' in line for line in events))
        mapped = map_words(words,self.plan.base_segments)
        self.assertEqual([w['caption_section'] for w in mapped],['source:0','narration:0'])

    def test_tiny_quote_fragment_after_bridge_is_not_emitted(self):
        import wave
        def voice(request,plan,root,warnings):
            path=root/'voiceover_00.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'))
                wav.writeframes(b'\0\0'*8000)
            return [(path,0,1,'voiceover')]
        self.request.editor.narration='intro_bridge_outro'
        self.plan.base_segments=[BaseSegment(source_start=0,source_end=3.12),BaseSegment(source_start=10,source_end=13)]
        self.plan.narration_sections=[NarrationSection(position='bridge',after_segment=0,text='The story now continues.')]
        text="We made the decision we're going to war with Iraq."
        self.plan.graphics=[GraphicPlan(start=0,end=3.341,text=text)]
        with patch('ai_compose_engine._synthesize_voiceovers',side_effect=voice):
            prepared=prepare_sections(self.request,self.plan,[],self.root,[])
        quotes=[g for g in prepared.plan.graphics if g.text==text]
        self.assertEqual(len(quotes),1)
        self.assertAlmostEqual(quotes[0].end-quotes[0].start,3.12)

    def test_visual_evidence_rejects_audio_invented_timestamps_and_nan(self):
        value={'score':.8,'observations':[{'time':1.,'category':'framing','observation':'Speaker remains visible.',
                                       'problem':False,'action':''}]}
        self.assertTrue(parse_visual_review(value,[{'time':1}],.7)['ok'])
        for update in [{'score':float('nan')},{'observations':[dict(value['observations'][0],time=5)]},
                       {'observations':[dict(value['observations'][0],observation='Speaker is inaudible.')]}]:
            with self.assertRaises(ValueError):
                parse_visual_review(dict(value,**update),[{'time':1}],.7)

    def test_editorial_failure_not_misattributed_to_visual_review(self):
        self.request.editor.visual_review_required=True
        body={'ok':True,'errors':[],'warnings':[],'score':.94,'duration':12.884}
        outputs=[{'duration':19.1,'quality_review':{'technical':{'ok':True},
                  'editorial':{'ok':False,'errors':['Graphic disappears too quickly'],'score':.69}}}]
        result=aggregate_review(body,outputs,{'performed':True,'ok':True,'score':.8},self.request)
        self.assertFalse(result['ok'])
        self.assertEqual(result['duration'],19.1)
        self.assertEqual(result['score'],.69)
        self.assertNotIn('visual review',' '.join(result['errors']))
        self.assertEqual(body['errors'],[])

    def test_explicit_layout_control_is_preserved(self):
        self.assertTrue(speaker_safe(self.request))
        self.request.editor.layout='director'
        self.assertFalse(speaker_safe(self.request))

    def test_logo_window_rejected_then_later_content_chosen_and_cached(self):
        asset=AssetRequest(query='military briefing',kind='video',start=0,end=3)
        path=self.root/'stock.mp4';path.write_bytes(b'mocked media')
        calls=[]
        def frames(path,duration,root,maximum,offset):
            return [{'time':0,'path':'image.jpg'},{'time':2,'path':'image.jpg'}]
        def reviewer(request,frames,prompt):
            calls.append(prompt)
            logo=frames[0]['time']==0
            return {'usable':not logo,'score':.9,'observations':[
                {'time':f['time'],'visible':'Logo' if logo else 'Soldiers in a briefing room',
                 'supports_speech':not logo,'title_or_logo_only':logo} for f in frames]}
        with patch('ai_compose_engine.probe_media',return_value={'duration':20}), \
             patch('ai_compose_engine.extract_frames',side_effect=frames), \
             patch('ai_compose_editor.visual_json',side_effect=reviewer):
            folder=self.root/'reviews';folder.mkdir()
            offset,reports=select_asset_window(self.request,asset,path,self.plan,self.words,folder)
            self.assertEqual(offset,5.)
            self.assertEqual(len(reports),2)
            self.assertEqual(select_asset_window(self.request,asset,path,self.plan,self.words,folder)[0],5.)
            self.assertEqual(len(calls),2)

    def test_invalid_asset_review_is_retried_and_never_cached_as_rejection(self):
        from PIL import Image
        path=self.root/'asset.jpg';Image.new('RGB',(20,20),'blue').save(path)
        folder=self.root/'review';folder.mkdir()
        asset=AssetRequest(query='test scene',kind='image',start=0,end=2)
        with patch('ai_compose_editor.visual_json',return_value={'usable':False}) as review:
            with self.assertRaisesRegex(ValueError,'contract failed'):
                select_asset_window(self.request,asset,path,self.plan,self.words,folder)
            self.assertEqual(review.call_count,2)
        self.assertFalse(list((self.root/'asset_review_cache').glob('*.json')))

    def test_split_layer_keeps_selected_source_offset(self):
        import wave
        def voice(request,plan,root,warnings):
            path=root/'voiceover_00.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'))
                wav.writeframes(b'\0\0'*8000)
            return [(path,0,1,'voiceover')]
        self.request.editor.narration='intro_bridge_outro'
        self.plan.base_segments=[BaseSegment(source_start=0,source_end=3),BaseSegment(source_start=10,source_end=13)]
        self.plan.narration_sections=[NarrationSection(position='bridge',after_segment=0,text='Now continue to the evidence.')]
        layer=ResolvedLayer(asset_id='stock',local_path='stock.mp4',source='local',query='test',kind='video',
            start=2,end=4,x=.04,y=.66,width=.92,height=.14,source_offset=5)
        with patch('ai_compose_engine._synthesize_voiceovers',side_effect=voice):
            prepared=prepare_sections(self.request,self.plan,[layer],self.root,[])
        self.assertEqual([l.source_offset for l in prepared.layers],[5,6])

    def test_failed_job_archive_contains_labelled_preview_not_secrets(self):
        import zipfile
        from ai_compose_kaggle import archive_run
        job=self.root/'job';draft=job/'draft_02';draft.mkdir(parents=True)
        (draft/'variant_01.mp4').write_bytes(b'preview')
        (draft/'private.env').write_text('PRIVATE=hidden')
        archive=archive_run(job,[])
        with zipfile.ZipFile(archive) as z:
            self.assertIn('rejected_previews/variant_01.mp4',z.namelist())
            self.assertIn('REJECTED_PREVIEW.txt',z.namelist())
            self.assertFalse(any('env' in name for name in z.namelist()))
            self.assertIn('No accepted final',z.read('ASSET_CREDITS.txt').decode())

    def test_render_protected_regions_and_selected_asset_offset(self):
        # Red source in speaker pane; stock is red for 1s, blue for 2s.
        from PIL import Image
        src=self.root/'source.mp4';stock=self.root/'stock.mp4'
        def ff(*args):
            subprocess.run(['ffmpeg','-y','-v','error',*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        ff('-f','lavfi','-i','color=red:s=270x480:d=4:r=24','-c:v','libx264','-threads','1',str(src))
        ff('-f','lavfi','-i','color=red:s=270x160:d=1:r=24','-f','lavfi','-i','color=blue:s=270x160:d=3:r=24',
           '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[v]','-map','[v]','-c:v','libx264','-threads','1',str(stock))
        layer=ResolvedLayer(asset_id='stock',local_path=str(stock),source='local',query='test',kind='video',
            source_offset=1,start=0,end=4,x=.04,y=.66,width=.92,height=.14)
        self.plan.graphics=[GraphicPlan(start=0,end=4,text='A readable header',placement='center')]
        with patch.dict(os.environ,{'MEDIA_FFMPEG_MAX_RSS_MB':'1024','AI_COMPOSE_VIDEO_CODEC':'libx264'}):
            result=render_composition(src,probe_media(src),self.request,self.plan,[layer],self.words,self.root,1,[])
        out=self.root/'frame.png';ff('-ss','0.5','-i',result['local_path'],'-frames:v','1',str(out))
        with Image.open(out) as im:
            speaker=im.getpixel((135,200));asset=im.getpixel((135,350))
        self.assertGreater(speaker[0],150)
        self.assertGreater(asset[2],150)
        self.assertLess(asset[0],80)
        self.assertAlmostEqual(result['duration'],4,delta=.1)


if __name__=='__main__':
    unittest.main()
