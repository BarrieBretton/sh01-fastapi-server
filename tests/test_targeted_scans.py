import unittest
from unittest.mock import Mock, patch
import social_inventory as core
import dashboard_scanner as scanner


class TargetedScannerTests(unittest.TestCase):
    def test_exact_posts_reuse_existing_fingerprints_without_rewalking_catalogue(self):
        fp={'algorithm':core.ALGORITHM,'version':core.VERSION,'media_type':'IMAGE','hashes':['123456789abcdef0'],'gray_stddev':20}
        rows=[{'platform':'instagram','account_id':'1','post_id':str(i),'media_type':'IMAGE','scan_kind':'IMAGE','permalink':f'https://www.instagram.com/p/fixture{i}/'} for i in (3,7)]
        def bridge(client,path,params):
            self.assertEqual(path,'posts');self.assertEqual(params['post_ids'],'3,7');self.assertNotIn('after',params)
            return {'source':'dashboard_catalogue','rows':rows,'next_after':None}
        with patch.object(scanner,'bridge_client',return_value=Mock()),patch.object(scanner,'read_bridge',side_effect=bridge),patch.object(scanner.FingerprintSink,'__init__',return_value=None),patch.object(scanner.FingerprintSink,'delivery_http',return_value=Mock()),patch.object(scanner.FingerprintSink,'cache',return_value=({'3':fp,'7':fp,'9':fp},False)),patch.object(scanner.FingerprintSink,'ingest',side_effect=lambda kind,values:len(values)):
            result=scanner.scan_catalogue(core.ScanRequest(platform='instagram',account_id='1',post_ids=['3','7'],max_posts=2))
        self.assertTrue(result['ok']);self.assertEqual(result['counts']['reused'],2);self.assertEqual(result['counts']['fingerprints_saved'],2)
        self.assertTrue(result['coverage']['targeted_posts']);self.assertFalse(result['coverage']['catalogue_exhausted']);self.assertFalse(result['coverage']['full_history_verified']);self.assertIsNone(result['resume_cursor'])

    def test_malformed_or_ambiguous_target_requests_fail_before_network(self):
        for extra in ({'post_ids':[],'max_posts':2},{'post_ids':['1','1'],'max_posts':2},{'post_ids':['1'],'max_posts':2},{'post_ids':['1'],'max_posts':1,'cursor':'old-cursor'},{'post_ids':['bad,id'],'max_posts':1}):
            with self.subTest(extra=extra),patch.object(scanner,'bridge_client') as bridge:
                with self.assertRaisesRegex(core.SafeError,'invalid_targeted_posts'):
                    scanner.scan_catalogue(core.ScanRequest(platform='instagram',account_id='1',**extra))
                bridge.assert_not_called()

    def test_missing_or_unexpected_target_does_not_claim_completion(self):
        provider=scanner.CatalogueProvider({'platform':'instagram','account_id':'1'},Mock());provider.post_ids=['3','7']
        for ids in (['3'],['3','9'],['3','7','9'],['3','3']):
            with self.subTest(ids=ids),patch.object(scanner,'read_bridge',return_value={'source':'dashboard_catalogue','rows':[{'post_id':i} for i in ids],'next_after':None}):
                with self.assertRaisesRegex(core.SafeError,'targeted_posts_missing_or_mismatched'):provider.page(None,20)


if __name__=='__main__':unittest.main()
