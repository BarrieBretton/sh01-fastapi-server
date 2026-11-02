"""Offline regression tests for failed searches, correction, and diagnostic export."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ai_compose_engine import LocalDirector, ComposeError, resolve_assets, asset_search_key
from ai_compose_schema import AIComposeRequest, CompositionPlan
from ai_compose_kaggle import archive_run

class RetrievalFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.request = AIComposeRequest(source_path='/tmp/input.mp4', fetch_assets=True)
        self.plan = CompositionPlan(base_segments=[{'source_start':0,'source_end':10}],
            asset_requests=[{'query':'Decision-Making Process','kind':'image','start':0,'end':3}])

    def test_empty_results_are_skipped_across_rounds(self):
        failures = set()
        with tempfile.TemporaryDirectory() as d, patch('ai_compose_engine._search_remote', return_value=[]) as search:
            root = Path(d)
            resolve_assets(self.request,self.plan,root/'round0',failed_searches=failures)
            resolve_assets(self.request,self.plan,root/'round1',failed_searches=failures)
            self.assertEqual(search.call_count,1)
            report=json.loads((root/'round1/retrieval_report.json').read_text())
            self.assertEqual(report[0]['outcome'],'previously_failed_search_skipped')

    def test_provider_errors_do_not_poison_cache(self):
        def search(*args, diagnostics):
            diagnostics.append({'provider':'wikimedia','error_type':'TimeoutError'})
            return []
        failures=set()
        with tempfile.TemporaryDirectory() as d, patch('ai_compose_engine._search_remote',side_effect=search):
            resolve_assets(self.request,self.plan,Path(d),failed_searches=failures)
        self.assertFalse(failures)

    def test_score_rejection_recorded(self):
        candidate={'asset_id':'a','title':'unrelated','score':0.01}
        with tempfile.TemporaryDirectory() as d, patch('ai_compose_engine._search_remote',return_value=[candidate]):
            root=Path(d)
            resolve_assets(self.request,self.plan,root)
            report=json.loads((root/'retrieval_report.json').read_text())[0]
            self.assertEqual(report['candidates'][0]['reason'],'below_score_floor')

    def test_download_failure_stays_retryable(self):
        candidate={'asset_id':'a','title':'relevant','score':1,'download_url':'https://example.org/a.jpg'}
        failures=set()
        with tempfile.TemporaryDirectory() as d, patch('ai_compose_engine._search_remote',return_value=[candidate]), patch('ai_compose_engine._download_sync',side_effect=TimeoutError('temporary')):
            root=Path(d)
            resolve_assets(self.request,self.plan,root,failed_searches=failures)
            report=json.loads((root/'retrieval_report.json').read_text())[0]
            self.assertEqual(report['download_decode_errors'][0]['error_type'],'TimeoutError')
        self.assertFalse(failures)

    def test_repeated_query_requires_model_correction(self):
        director=LocalDirector(self.request,failed_searches={asset_search_key('decision making process','image')})
        prompts=[]
        def generate(prompt):
            prompts.append(prompt)
            plan=self.plan.model_dump()
            if len(prompts)>1:
                plan['asset_requests'][0].update(query='Pentagon exterior',benefit='Locate the account',evidence_refs=['transcript'])
            return json.dumps(plan)
        result=director._generate_valid_plan(generate,'Transcript mentions Pentagon')
        self.assertEqual(len(prompts),2)
        self.assertIn('Previously failed asset searches',prompts[1])
        self.assertEqual(result.asset_requests[0].query,'Pentagon exterior')

    def test_archive_includes_failed_draft_diagnostics_not_env(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'job'; draft=root/'draft_00'; draft.mkdir(parents=True)
            (draft/'storyboard.json').write_text('{}')
            (draft/'retrieval_report.json').write_text('[]')
            (root/'.env').write_text('PRIVATE=value')
            with zipfile.ZipFile(archive_run(root,[])) as z:
                self.assertIn('draft_00/storyboard.json',z.namelist())
                self.assertIn('draft_00/retrieval_report.json',z.namelist())
                self.assertNotIn('.env',z.namelist())
