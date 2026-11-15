import unittest
from unittest.mock import Mock, patch
import social_inventory as core
import dashboard_scanner as scanner


def row(i, kind='IMAGE'):
    return {'platform':'instagram','account_id':'1','post_id':str(i),'media_type':kind,'scan_kind':kind,
            'permalink':f'https://www.instagram.com/p/fixture{i}/',
            'stored_media':{'available':True,'url':f'https://a.cdninstagram.com/{i}.media',
                            'media_type':kind,'kind':'video' if kind=='VIDEO' else 'image'}}


class CatchupTests(unittest.TestCase):
    def test_eight_stored_originals_need_one_catalogue_read_and_no_media_resolution(self):
        rows=[row(i) for i in range(8)]
        provider=scanner.CatalogueProvider({'platform':'instagram','account_id':'1'},Mock())
        provider.post_ids=[str(i) for i in range(8)]
        with patch.object(scanner,'read_bridge',return_value={'source':'dashboard_catalogue','rows':rows,'next_after':None}) as bridge:
            posts,_=provider.page(None,20)
            for raw in posts:
                post=provider.normalize(raw,provider.cfg)
                self.assertEqual(provider.media_url(post),raw['stored_media']['url'])
            self.assertEqual(bridge.call_count,1)

    def test_unsafe_or_wrong_kind_stored_media_cannot_become_a_video_source(self):
        cases=[{'url':'http://a.fbcdn.net/x'}, {'url':'https://evil.test/x'},
               {'url':'https://secret@a.fbcdn.net/x'}, {'url':'https://a.fbcdn.net:80/x'},
               {'url':'https://a.fbcdn.net/x#secret'}, {'kind':'image'}, {'media_type':'IMAGE'}]
        for changes in cases:
            raw=row(1,'VIDEO');raw['stored_media'].update(changes)
            provider=scanner.CatalogueProvider({'platform':'instagram','account_id':'1'},Mock())
            with self.subTest(changes=changes),patch.object(scanner,'read_bridge',return_value={'source':'dashboard_catalogue','rows':[raw],'next_after':None}):
                provider.page(None,20)
                self.assertEqual(provider.media,{})

    def test_expired_stored_original_refreshes_once_and_still_validates_download(self):
        raw=row(1,'VIDEO');provider=scanner.CatalogueProvider({'platform':'instagram','account_id':'1'},Mock())
        refreshed={**raw['stored_media'],'url':'https://a.fbcdn.net/refreshed.mp4'}
        with patch.object(scanner,'read_bridge',side_effect=[{'source':'dashboard_catalogue','rows':[raw],'next_after':None},refreshed]) as bridge:
            provider.page(None,20);post=provider.normalize(raw,provider.cfg)
            http=Mock();http.request.side_effect=[core.GraphHTTPError('upstream_http_error',403),b'video bytes']
            self.assertEqual(provider.download(post,provider.media_url(post),http),b'video bytes')
            self.assertEqual(bridge.call_count,2);self.assertEqual(bridge.call_args.args[1],'media');self.assertEqual(bridge.call_args.args[2]['refresh'],'1')
            self.assertEqual(http.request.call_args.kwargs['suffix'],True)
            self.assertEqual(http.request.call_args.args[1],core.CDNS['instagram'])

    def test_nineteen_identical_videos_grow_one_group_across_bounded_batches(self):
        frame={'algorithm':core.ALGORITHM,'version':1,'media_type':'IMAGE','hashes':['123456789abcdef0'],'gray_stddev':20}
        fp={'algorithm':core.ALGORITHM,'version':1,'media_type':'VIDEO','duration':12,'sample_positions':[.1,.3,.5,.7,.9],'frames':[frame]*5,'sha256':'a'*64}
        known={};largest=[]
        for start in range(0,19,8):
            batch=[str(i) for i in range(start,min(start+8,19))]
            for pid in batch:known[pid]=fp
            groups,comparisons,truncated=core.build_groups(known,set(batch),{'platform':'instagram','account_id':'1'},core.Budget())
            self.assertFalse(truncated)
            largest=max(groups,key=lambda g:len(g['members']))
        self.assertEqual(len(largest['members']),19)
        self.assertLessEqual(comparisons,171)


if __name__=='__main__':unittest.main()
