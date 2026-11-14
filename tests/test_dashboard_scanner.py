import io
import os
import socket
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
import social_inventory as inv
import dashboard_scanner as ds

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ, {'MEDIA_CLIP_API_KEY':'scanner-key'},clear=True);self.env.start()
    def tearDown(self):self.env.stop()
    def test_accounts_and_diagnostics_do_not_require_meta_token(self):
        app=FastAPI();app.include_router(inv.router);client=TestClient(app)
        with patch.object(ds,'read_bridge',side_effect=[{'ok':True,'source':'dashboard_catalogue'},
                                                       {'source':'dashboard_catalogue','rows':[{'platform':'facebook','account_id':'1'}]}]):
            self.assertEqual(client.get('/social/inventory/diagnostics').status_code,401)
            r=client.get('/social/inventory/diagnostics',headers={'X-API-KEY':'scanner-key'})
            self.assertTrue(r.json()['dashboard']['auth_verified'])
            r=client.get('/social/inventory/accounts',headers={'X-API-KEY':'scanner-key'})
            self.assertEqual(r.status_code,200);self.assertEqual(len(r.json()['rows']),1)
    def test_diagnostic_redacts_exception_text(self):
        with patch.object(ds,'read_bridge',side_effect=inv.TransportError(OSError(101,'SECRET URL token'),'tcp_connect')):
            result=ds.connection_diagnostics()
        self.assertEqual(result['dashboard']['errno'],101);self.assertNotIn('SECRET',str(result))
        self.assertFalse(result['dashboard']['auth_verified'])
    def test_sink_only_sends_fingerprints_never_old_metrics(self):
        http=Mock();http.json.return_value={'accepted':1};sink=ds.FingerprintSink(http)
        row={'platform':'instagram','account_id':'1','post_id':'2','fingerprint':{'media_type':'IMAGE'},'metrics':{'views':100},'caption':'old'}
        self.assertEqual(sink.ingest('posts',[row,{'metrics':{}}]),1)
        sent=http.json.call_args.kwargs['payload']['rows'][0]
        self.assertNotIn('metrics',sent);self.assertNotIn('caption',sent)
    def test_video_thumbnail_is_not_accepted_as_video(self):
        cfg={'platform':'facebook','account_id':'1'};p=ds.CatalogueProvider(cfg,Mock())
        p.media['2']={'available':True,'url':'https://a.fbcdn.net/x.jpg','kind':'image','media_type':'IMAGE'}
        with self.assertRaisesRegex(inv.SafeError,'original_media_type_mismatch'):
            p.media_url(inv.Post({'post_id':'2','media_type':'VIDEO'}))
    def test_expired_url_refreshes_once(self):
        p=ds.CatalogueProvider({'platform':'instagram','account_id':'1'},Mock())
        p.media['2']={'available':True,'url':'https://a.cdninstagram.com/new','kind':'image','media_type':'IMAGE'}
        h=Mock();h.request.side_effect=[inv.GraphHTTPError('upstream_permission_denied',403),b'image']
        with patch.object(p,'resolve',return_value=p.media['2']) as resolve:
            data=p.download(inv.Post({'post_id':'2','media_type':'IMAGE'}),'https://a.cdninstagram.com/old',h)
        self.assertEqual(data,b'image');self.assertEqual(h.request.call_count,2)
        self.assertTrue(resolve.call_args_list[0].kwargs['refresh'])
    def test_catalogue_reuses_fingerprint_and_uses_separate_cursor_scope(self):
        buf=io.BytesIO();im=Image.new('RGB',(32,32));im.putdata([(x*7%255,x*11%255,x*13%255) for x in range(1024)]);im.save(buf,format='PNG')
        fp=inv.fingerprint_bytes(buf.getvalue(),'IMAGE',inv.Budget())
        rows=[{'platform':'instagram','account_id':'1','post_id':str(i),'media_type':'IMAGE','scan_kind':'IMAGE','permalink':'https://www.instagram.com/p/a'+str(i)+'/'} for i in (2,3)]
        def bridge(client,path,params=None):
            if path=='posts':return {'source':'dashboard_catalogue','rows':[r for r in rows if not params.get('after') or r['post_id']>params['after']], 'next_after':None}
            raise AssertionError('Cached fingerprints must not fetch media')
        with patch.object(ds,'read_bridge',side_effect=bridge), patch.object(ds.FingerprintSink,'cache',return_value=({'2':fp,'3':fp},False)), patch.object(ds.FingerprintSink,'ingest',side_effect=lambda kind,rows:len(rows)):
            result=ds.scan_catalogue(inv.ScanRequest(platform='instagram',account_id='1',max_posts=1))
            self.assertEqual(result['counts']['reused'],1)
            self.assertFalse(result['coverage']['catalogue_exhausted'])
            self.assertFalse(result['coverage']['provider_exhausted'])
            cursor=result['resume_cursor']
            with self.assertRaises(inv.SafeError):inv.decode_cursor(cursor,'1','instagram')
            state=inv.decode_cursor(cursor,'catalogue-v1:1','instagram');self.assertEqual(state['after'],'2');self.assertEqual(state['skip'],0)
            result=ds.scan_catalogue(inv.ScanRequest(platform='instagram',account_id='1',max_posts=1,cursor=cursor))
            self.assertTrue(result['coverage']['catalogue_exhausted'])
            self.assertEqual(result['counts']['reused'],1)
    def test_connect_falls_back_only_to_pinned_addresses_and_preserves_sni(self):
        raw=Mock();wrapped=Mock()
        connection=inv.PinnedHTTPS('graph.facebook.com',['1.1.1.1','2606:4700::1111'],10)
        context=Mock();context.wrap_socket.return_value=wrapped;connection._context=context
        with patch.object(inv.socket,'create_connection',side_effect=[OSError(101,'network unreachable'),raw]) as connect:
            connection.connect()
        self.assertEqual(connect.call_args_list[0].args[0],('1.1.1.1',443))
        self.assertEqual(connect.call_args_list[1].args[0],('2606:4700::1111',443))
        context.wrap_socket.assert_called_once_with(raw,server_hostname='graph.facebook.com')
    def test_dns_public_validation_and_ipv4_preference(self):
        def resolve(*a,**k):return [(socket.AF_INET6,socket.SOCK_STREAM,6,'',('2606:4700::1111',443,0,0)),(socket.AF_INET,socket.SOCK_STREAM,6,'',('1.1.1.1',443))]
        _,addresses,_=inv.validate_url('https://graph.facebook.com/',('graph.facebook.com',),resolver=resolve,all_addresses=True)
        self.assertEqual(addresses,['1.1.1.1','2606:4700::1111'])
        with self.assertRaises(inv.SafeError):
            inv.validate_url('https://graph.facebook.com/',('graph.facebook.com',),resolver=lambda *a,**k:[(2,1,6,'',('127.0.0.1',443))],all_addresses=True)

if __name__=='__main__':unittest.main()
