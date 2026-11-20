"""Offline Snapchat contracts. No live Snapchat, workbook or B2 calls."""
import asyncio
import base64
import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, parse_qs
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from snapchat_feature.core import Config, SnapError, HEADERS, queue_rows, schedules, source_locator, public_job, job_id
from snapchat_feature.client import SnapClient, MultipartChunk, encrypt_file, request, download, probe
from snapchat_feature.service import Service
from snapchat_feature.router import router

PROFILE = '11111111-2222-3333-4444-555555555555'
def config(enabled=True):
    return Config('client', 'private-client-secret', 'https://example.com/snapchat/oauth/callback', PROFILE,
                  Fernet.generate_key().decode(), ('f005.backblazeb2.com',), enabled)
def values(item='one', destination='spotlight'):
    return [HEADERS, [item, 'https://f005.backblazeb2.com/file/bucket/videos/movie.mp4?Authorization=PRIVATE', destination, 'ready', 'yes', 'yes', 'Original episode', 'en_IN']]


class State:
    def __init__(self):
        self.values, self.writes, self.owner = {}, [], None
        self.control = {'lease': None, 'verified': True}; self.fail = None
    def get(self, key): return copy.deepcopy(self.values.get(key))
    def put(self, key, value): self.guard(); self.commit({key: value})
    def commit(self, values, **control):
        if self.fail and self.fail(values): raise SnapError('state_write_failed')
        self.writes.append(copy.deepcopy(values)); self.values.update(copy.deepcopy(values)); self.control.update(control)
    def load(self): return True
    def acquire(self):
        if self.owner: return False
        self.owner='test-owner'; self.control['lease']={'owner': self.owner, 'until': 10**12}; return True
    def release(self): self.owner=None; self.control['lease']=None
    def guard(self):
        if not self.owner: raise SnapError('lease_lost')


class Book:
    def __init__(self, cfg):
        self.cfg=cfg; self.state=State(); self.rows=values(); self.reads=0; self.on_read=None
        self.google=type('Google', (), {'schedule_rows': lambda _: [['Workflow','Timings'],['Server::Snapchat-Spotlight','12:00am']]})()
    def queue(self):
        self.reads+=1
        if self.on_read: self.on_read(self)
        return queue_rows(self.rows, 'workbook', 42, 'Snapchat-Queue', self.cfg)


class Provider:
    def __init__(self, cfg, state): self.cfg,self.state=cfg,state; self.posts=0; self.uploads=0; self.publish_error=None; self.verify_error=False; self.status='LIVE'; self.on_publish=None
    def profile(self): return {'id':PROFILE,'display_name':'Creator'}
    def upload(self,*args): self.uploads+=1; return 'media-id'
    def publish(self, media, row):
        self.posts+=1
        if self.on_publish:self.on_publish()
        if self.publish_error:raise self.publish_error
        return {'request_status':'SUCCESS','spotlight_id':'spotlight_one','request_id':'request-id'}
    def verify(self, ident):
        if self.verify_error:raise SnapError('snapchat_http_503')
        return self.status
    def metrics(self):return {'values':{},'checked_at':10000}


class CoreTests(unittest.TestCase):
    def test_config_is_disabled_by_default(self):
        with patch.dict('os.environ', {}, clear=True): self.assertFalse(Config.env().enabled)
    def test_encrypted_state_and_key_mismatch(self):
        c=config(); text=c.seal({'refresh':'private'})
        self.assertNotIn('private',text); self.assertEqual(c.unseal(text),{'refresh':'private'})
        with self.assertRaises(SnapError):config().unseal(text)
    def test_config_rejects_bad_callback(self):
        for url in ['http://example.com/snapchat/oauth/callback','https://user:pass@example.com/snapchat/oauth/callback','https://example.com/other','https://example.com/snapchat/oauth/callback?q=x']:
            with self.assertRaises(SnapError): replace(config(),redirect_uri=url).validate()
    def test_source_strict_host_and_query_redaction(self):
        c=config(); url=values()[1][1]
        self.assertNotIn('PRIVATE',source_locator(url,c.source_hosts))
        for url in ['https://evil.f005.backblazeb2.com/a','https://f005.backblazeb2.com.evil.org/a','http://f005.backblazeb2.com/a','https://x@y/a','https://f005.backblazeb2.com:444/a']:
            with self.assertRaises(SnapError):source_locator(url,c.source_hosts)
    def test_queue_references_and_no_unsigned_public_url(self):
        row=queue_rows(values(),'book',42,'Snapchat-Queue',config())[0]
        self.assertEqual(row['reference']['cell'],'B2');self.assertIn('#gid=42&range=B2',row['reference']['link']);self.assertFalse(row['problems'])
        self.assertNotIn('Authorization',row['source_locator'])
    def test_reordered_headers_and_row_numbers(self):
        v=values();v=[list(reversed(v[0])),[],list(reversed(v[1]))]
        row=queue_rows(v,'book',42,'Snapchat-Queue',config())[0]
        self.assertEqual(row['reference']['cell'],'G3')
    def test_duplicates_all_blocked(self):
        v=values();v.append(copy.deepcopy(v[1]));v[2][2]='story'
        self.assertTrue(all('duplicate_item_id' in r['problems'] for r in queue_rows(v,'b',1,'s',config())))
    def test_missing_approval_rights_status(self):
        for name in ['status','approved','rights_confirmed']:
            v=values();v[1][HEADERS.index(name)]=''
            self.assertTrue(queue_rows(v,'b',1,'s',config())[0]['problems'])
    def test_schedule_ist_midnight(self):
        now=datetime(2026,11,18,18,0,tzinfo=timezone.utc)
        result=schedules([['Workflow','Timings'],['Server::Snapchat-Spotlight','11:15pm, 12:00am']],now)[0]
        self.assertEqual(result['due'],'2026-11-18T23:15+05:30'); self.assertEqual(result['next_at'],'2026-11-19T00:00:00+05:30')
    def test_schedule_blank_pauses_and_duplicate_blocks(self):
        self.assertIsNone(schedules([['Workflow','Timings'],['Server::Snapchat-Spotlight','']])[0]['due'])
        result=schedules([['Workflow','Timings'],['Server::Snapchat-Spotlight','11:15pm'],['Server::Snapchat-Spotlight','11:30pm']])
        self.assertEqual(result[0]['error'],'duplicate_schedule_rows'); self.assertIsNone(result[0]['due'])
    def test_schedule_invalid_and_native_time(self):
        self.assertEqual(schedules([['Workflow','Timings'],['Server::Snapchat-Spotlight',.5]])[0]['times'],[720])
        self.assertEqual(schedules([['Workflow','Timings'],['Server::Snapchat-Spotlight','24:01']])[0]['error'],'invalid_schedule_time')


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config();self.book=Book(self.cfg);self.state=self.book.state
        self.state.values['oauth_tokens']={'encrypted':self.cfg.seal({'client_id':'client','profile_id':PROFILE})}
        self.provider=Provider(self.cfg,self.state)
        self.service=Service(self.cfg,self.book,lambda c,s:self.provider)
    def enqueue(self):
        row=self.book.queue()[0];return self.service.run(row['item_id'],row['fingerprint'])
    def process(self):
        def dl(url,path,c,guard):guard();Path(path).write_bytes(b'fake video');return {'bytes':10,'sha256':'a'*64}
        def enc(src,dst,guard):Path(dst).write_bytes(b'cipher');return 'key','iv'
        return self.service.process_one(dl,lambda p:{'duration_seconds':40},enc)
    def test_preview_is_read_only_and_secret_free(self):
        self.service.preview(); self.assertFalse(self.state.writes); self.assertEqual(self.provider.posts,0)
        self.assertNotIn('PRIVATE',json.dumps(self.service.preview()))

    def test_health_before_setup_reads_the_state_once(self):
        with patch.object(self.state, 'load', return_value=False) as load, patch.object(self.state, 'get', side_effect=AssertionError):
            result = self.service.health()
        load.assert_called_once_with()
        self.assertEqual(result['workbook_state'], 'setup_required')
        self.assertEqual(result['next_step'], 'initialize_workbook')
        self.assertEqual(len(result['schedules']), 2)
        self.assertFalse(self.state.writes)

    def test_partial_setup_requires_explicit_initialization(self):
        self.state.control['verified'] = False
        self.assertEqual(self.service.health()['next_step'], 'initialize_workbook')
        self.assertFalse(self.state.writes)

    def test_health_distinguishes_unavailable_state_and_schedule(self):
        with patch.object(self.state, 'load', side_effect=SnapError('sheets_unavailable')):
            result = self.service.health()
        self.assertEqual(result['state_error'], 'sheets_unavailable')
        self.assertEqual(result['workbook_state'], 'unknown')
        self.assertEqual(result['authorization'], 'unknown')
        self.assertEqual(result['next_step'], 'check_workbook_access')
        self.assertEqual(len(result['schedules']), 2)

    def test_health_survives_missing_google_configuration_on_normal_route(self):
        app = FastAPI()
        app.include_router(router)
        with patch.dict('os.environ', {'MEDIA_CLIP_API_KEY': 'test-key'}, clear=True):
            response = TestClient(app).get('/snapchat/health', headers={'X-API-KEY': 'test-key'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        result = response.json()
        self.assertFalse(result['enabled'])
        self.assertIn('SNAPCHAT_CLIENT_ID', result['missing_configuration'])
        self.assertEqual(result['state_error'], 'spreadsheet_not_configured')
        self.assertEqual(result['schedule_error'], result['state_error'])
        self.assertEqual(result['next_step'], 'check_workbook_access')

    def test_health_explains_missing_and_malformed_snap_settings(self):
        self.service.config = replace(self.cfg, profile_id='')
        self.assertEqual(self.service.health()['missing_configuration'], ['SNAPCHAT_PROFILE_ID'])
        self.assertEqual(self.service.health()['next_step'], 'configure_snapchat')
        self.service.config = replace(self.cfg, profile_id='incorrect')
        result = self.service.health()
        self.assertEqual(result['configuration_error'], 'snapchat_profile_id_invalid')
        self.assertEqual(result['next_step'], 'configure_snapchat')
        self.assertFalse(self.state.writes)

    def test_health_displays_historical_read_for_matching_app_and_profile_only(self):
        self.state.values['connection'] = {'id': PROFILE, 'display_name': 'Creator'}
        self.assertEqual(self.service.health()['next_step'], 'check_profile_access')  # legacy evidence
        self.state.values['connection']['client_id'] = 'other-app'
        self.assertEqual(self.service.health()['api_access'], 'not_verified')
        self.service.check_connection()
        self.state.writes.clear()
        self.service.client_factory = lambda *args: self.fail('Health contacted Snapchat')
        result = self.service.health()
        self.assertEqual(result['api_access'], 'profile_read_verified_previously')
        self.assertEqual(result['next_step'], 'review_queue')
        self.assertNotIn('client_id', result['connection'])
        self.assertFalse(self.state.writes)

    def test_failed_or_absent_tokens_never_reuse_a_previous_profile_read(self):
        self.service.check_connection()
        self.state.values['oauth_tokens']['refresh_pending'] = True
        result = self.service.health()
        self.assertEqual(result['next_step'], 'reconnect_snapchat')
        self.assertNotIn('connection', result)
        self.state.values['oauth_tokens'] = {'encrypted': 'PRIVATE-CIPHERTEXT'}
        result = self.service.health()
        self.assertEqual(result['next_step'], 'check_encryption_key')
        self.assertEqual(result['authorization_error'], 'snapchat_encrypted_state_unreadable')
        self.assertNotIn('PRIVATE-CIPHERTEXT', json.dumps(result))
        del self.state.values['oauth_tokens']
        self.assertEqual(self.service.health()['next_step'], 'connect_snapchat')

    def test_new_grant_atomically_clears_earlier_profile_read_until_verified(self):
        self.service.check_connection()
        self.state.acquire()
        client = SnapClient(self.cfg, self.state)
        client.save_tokens({'access_token': 'sample-access', 'refresh_token': 'sample-refresh',
                            'expires_in': 3600, 'scope': 'snapchat-profile-api'})
        self.assertEqual(set(self.state.writes[-1]), {'oauth_tokens', 'connection'})
        self.state.release()
        result = self.service.health()
        self.assertEqual(result['next_step'], 'check_profile_access')
        self.assertNotIn('sample-refresh', json.dumps(result))
    def test_disabled_manual_and_schedule(self):
        self.service.config=replace(self.cfg,enabled=False)
        with self.assertRaisesRegex(SnapError,'disabled'):self.enqueue()
        self.assertEqual(self.service.tick()['status'],'paused');self.assertEqual(self.provider.posts,0)
    def test_durable_manual_deduplication(self):
        a=self.enqueue();b=self.enqueue();self.assertEqual(a['job_id'],b['job_id']);self.assertEqual(len([k for k in self.state.values if k.startswith('job:')]),1)
    def test_stale_preview_blocks(self):
        row=self.book.queue()[0]; self.book.rows[1][6]='changed'
        with self.assertRaisesRegex(SnapError,'queue_changed'):self.service.run('one',row['fingerprint'])
    def test_due_occurrence_atomically_claims_job(self):
        self.service.tick();self.state.values['next_poll_at']=0;self.service.tick()
        writes=[w for w in self.state.writes if any(k.startswith('occurrence:') for k in w)]
        self.assertEqual(len(writes),1);self.assertTrue(any(k.startswith('job:') for k in writes[0]))
    def test_no_ready_content_records_occurrence(self):
        self.book.rows[1][4]='no';self.service.tick()
        self.assertFalse(any(k.startswith('job:') for k in self.state.values))
        self.assertTrue(any(k.startswith('occurrence:') for k in self.state.values))
    def test_publish_intent_precedes_dispatch_and_live_verified(self):
        self.enqueue()
        def check():
            self.assertTrue(any(k.startswith('content:') for k in self.state.values))
            self.assertEqual([v['status'] for k,v in self.state.values.items() if k.startswith('job:')],['publishing'])
        self.provider.on_publish=check
        result=self.process();self.assertEqual(result['status'],'published');self.assertEqual(self.provider.posts,1);self.assertFalse(result['link_verified'])
        self.process();self.assertEqual(self.provider.posts,1)
    def test_ambiguous_publish_is_never_replayed(self):
        self.enqueue();self.provider.publish_error=SnapError('snapchat_transport_outcome_unknown')
        self.assertEqual(self.process()['status'],'outcome_unknown');self.process();self.enqueue();self.process();self.assertEqual(self.provider.posts,1)
    def test_accepted_pending_not_claimed_published(self):
        self.enqueue();self.provider.status='PENDING';r=self.process();self.assertEqual(r['status'],'accepted');self.assertEqual(self.provider.posts,1)
    def test_verification_failure_does_not_repost(self):
        self.enqueue();self.provider.verify_error=True;self.assertEqual(self.process()['status'],'accepted');self.process();self.assertEqual(self.provider.posts,1)
    def test_changed_row_before_execution_stops(self):
        self.enqueue();self.book.rows[1][4]='no';r=self.process();self.assertEqual(r['status'],'failed_before_publication');self.assertEqual(self.provider.posts,0)
    def test_changed_row_during_upload_stops_before_publish(self):
        self.enqueue(); original=self.provider.upload
        def upload(*args):self.book.rows[1][4]='no';return original(*args)
        self.provider.upload=upload;r=self.process();self.assertEqual(r['status'],'failed_before_publication');self.assertEqual(self.provider.posts,0)
    def test_crash_after_publish_intent_is_quarantined(self):
        job=self.enqueue();self.state.values['job:'+job['job_id']].update(status='publishing',publish_started=True)
        self.process();self.assertEqual(self.service.job(job['job_id'])['status'],'outcome_unknown');self.assertEqual(self.provider.posts,0)
    def test_crash_before_publish_can_resume(self):
        job=self.enqueue();self.state.values['job:'+job['job_id']]['status']='running'
        self.assertEqual(self.process()['status'],'published');self.assertEqual(self.provider.posts,1)
    def test_duplicate_bytes_under_new_item_id_blocked(self):
        self.enqueue();self.process();self.book.rows[1][0]='two';self.enqueue()
        r=self.process();self.assertEqual(r['error'],'content_already_submitted_do_not_repost');self.assertEqual(self.provider.posts,1)
    def test_rate_limit_before_dispatch_backoff(self):
        self.enqueue();self.provider.upload=lambda *a:(_ for _ in ()).throw(SnapError('snapchat_http_429',retry_after=120))
        r=self.process();self.assertEqual(r['status'],'waiting_retry');self.assertGreater(r['retry_at'],self.service.clock());self.assertEqual(self.provider.posts,0)
    def test_intent_write_failure_never_dispatches(self):
        self.enqueue();self.state.fail=lambda changes:any(k.startswith('content:') for k in changes)
        r=self.process();self.assertEqual(r['status'],'outcome_unknown');self.assertEqual(self.provider.posts,0)
    def test_story_acceptance_is_not_live_verification(self):
        self.book.rows[1][2]='story';self.enqueue();r=self.process();self.assertEqual(r['status'],'accepted');self.assertNotIn('published_url',r)
    def test_public_record_excludes_secrets(self):
        self.enqueue();self.process();result=json.dumps(self.service.health())
        for secret in ['PRIVATE','candidate_encrypted','refresh_token','access_token','private-client-secret']:self.assertNotIn(secret,result)
    def test_connection_state_is_one_time(self):
        # Replace OAuth transport with a contract fake. No publication occurs.
        client=SnapClient(self.cfg,self.state,transport=lambda *a,**k:{'access_token':'access','refresh_token':'refresh','expires_in':3600,'scope':'snapchat-profile-api'})
        client.profile=lambda:{'id':PROFILE};self.service.client_factory=lambda c,s:client
        url=self.service.connect()['authorize_url'];nonce=parse_qs(urlsplit(url).query)['state'][0]
        with self.assertRaisesRegex(SnapError,'state_invalid'):self.service.callback('code','wrong')
        self.service.callback('code',nonce)
        with self.assertRaisesRegex(SnapError,'state_invalid'):self.service.callback('code',nonce)
        self.assertEqual(self.provider.posts,0)


class TransferTests(unittest.TestCase):
    def test_streaming_encryption_is_whole_file_pkcs7_cbc(self):
        raw=b'0123456789'*10003
        with tempfile.TemporaryDirectory() as d:
            src,dst=Path(d)/'in',Path(d)/'out';src.write_bytes(raw);key,iv=encrypt_file(src,dst)
            dec=Cipher(algorithms.AES(base64.b64decode(key)),modes.CBC(base64.b64decode(iv))).decryptor()
            unpad=padding.PKCS7(128).unpadder();padded=dec.update(dst.read_bytes())+dec.finalize()
            self.assertEqual(unpad.update(padded)+unpad.finalize(),raw)
    def test_multipart_reads_only_one_chunk_bounded(self):
        file=io.BytesIO(b'a'*100+b'b'*100);part=MultipartChunk(file,100,1);pieces=[]
        while True:
            p=part.read(13)
            if not p:break
            self.assertLessEqual(len(p),13);pieces.append(p)
        result=b''.join(pieces);self.assertEqual(len(result),part.length);self.assertEqual(file.tell(),100);self.assertIn(b'a'*100,result);self.assertNotIn(b'b'*100,result)
    def test_upload_paths_are_bound_to_profile_and_media(self):
        c=config();s=State();s.acquire();client=SnapClient(c,s)
        client.api=lambda *a,**kw:{'media_id':'media','add_path':'https://evil.example/upload','finalize_path':'/x'}
        with self.assertRaisesRegex(SnapError,'path_rejected'):client.upload('/not/read','key','iv')
    def test_upload_sends_ordered_complete_parts_then_multipart_finalize(self):
        c=config();s=State();s.acquire();client=SnapClient(c,s);calls=[];parts=[]
        upload_path='/us/v1/public_profiles/'+PROFILE+'/media/media-id/multipart-upload'
        def api(path, **kwargs):
            calls.append((path, kwargs))
            if path.endswith('/media'):
                self.assertEqual(kwargs['data']['key'],'key');self.assertEqual(kwargs['data']['iv'],'iv')
                return {'media_id':'media-id','add_path':upload_path,'finalize_path':upload_path}
            body=kwargs['multipart'];chunks=[]
            while True:
                data=body.read(7)
                if not data:break
                self.assertLessEqual(len(data),7);chunks.append(data)
            raw=b''.join(chunks);self.assertEqual(len(raw),body.length)
            self.assertTrue(body.content_type.startswith('multipart/form-data; boundary='))
            if b'FINALIZE' in raw:
                self.assertEqual(parts,[b'0123456789',b'ABCDEFGHIJ',b'xyz'])
            else:
                self.assertIn(('\r\n\r\n'+str(len(parts)+1)+'\r\n').encode(),raw)
                parts.append(raw.split(b'Content-Type: application/octet-stream\r\n\r\n')[1].split(b'\r\n--')[0])
            return {'request_status':'SUCCESS'}
        client.api=api
        with tempfile.TemporaryDirectory() as directory,patch('snapchat_feature.client.CHUNK',10):
            file=Path(directory)/'encrypted';file.write_bytes(b'0123456789ABCDEFGHIJxyz')
            self.assertEqual(client.upload(file,'key','iv'),'media-id')
        self.assertEqual(len(calls),5)
    def test_download_rejects_partial_oversize_and_redirect_responses(self):
        def attempt(status,declared,data,max_bytes=100):
            response=io.BytesIO(data);response.status=status
            response.getheader=lambda name,default=None: {'Content-Length':declared}.get(name,default)
            connection=type('Connection',(),{'request':lambda *a,**k:None,'getresponse':lambda self:response,'close':lambda self:None})()
            with tempfile.TemporaryDirectory() as directory,patch('snapchat_feature.client.validate_url',return_value=('f005.backblazeb2.com','1.1.1.1','/x')),patch('snapchat_feature.client.PinnedHTTPS',return_value=connection):
                return download(values()[1][1],Path(directory)/'video',replace(config(),max_bytes=max_bytes))
        with self.assertRaisesRegex(SnapError,'incomplete'):attempt(200,'30',b'only 7')
        with self.assertRaisesRegex(SnapError,'size_limit'):attempt(200,None,b'123456',5)
        with self.assertRaisesRegex(SnapError,'http_302'):attempt(302,'0',b'')
        self.assertEqual(attempt(200,'3',b'abc')['bytes'],3)
    def test_metrics_preserve_zero_and_reject_missing_fields(self):
        c=config();s=State();s.acquire();client=SnapClient(c,s)
        expected=['STORY_VIEW_TIME_MILLIS','SPOTLIGHT_VIEW_TIME_MILLIS','STORY_VIEWS','SPOTLIGHT_VIEWS','SUBSCRIBERS']
        def api(path):
            fields=parse_qs(urlsplit(path).query)['fields'][0].split(',')
            return {'assets':[{'asset_id':PROFILE,'sub_request_status':'SUCCESS','timeseries':[{'fields':[{'field':{'field_name':f},'stats':[{'value':'0'}]} for f in fields]}]}]}
        client.api=api
        result=client.metrics();self.assertEqual(set(result['values']),set(expected));self.assertEqual(result['values']['STORY_VIEWS'],0)
        client.api=lambda path:{'assets':[]}
        with self.assertRaisesRegex(SnapError,'incomplete'):client.metrics()
    def test_rate_limit_retry_after_is_bounded(self):
        response=type('Response',(),{'status':429,'getheader':lambda self,k:'999999'})()
        connection=type('Connection',(),{'request':lambda *a,**k:None,'getresponse':lambda self:response,'close':lambda self:None})()
        with patch('snapchat_feature.client.validate_url',return_value=('businessapi.snapchat.com','1.1.1.1','/x')),patch('snapchat_feature.client.PinnedHTTPS',return_value=connection):
            with self.assertRaises(SnapError) as raised:request('https://businessapi.snapchat.com/x')
            self.assertEqual(raised.exception.retry_after,3600)
    def test_token_refresh_rotation_stored_encrypted(self):
        c=config();s=State();s.acquire();client=SnapClient(c,s,clock=lambda:10000,transport=lambda *a,**k:{'access_token':'new','refresh_token':'rotated','expires_in':3600,'scope':'snapchat-profile-api'})
        s.put('oauth_tokens',{'encrypted':c.seal({'access_token':'old','refresh_token':'initial','expires_at':0,'client_id':c.client_id,'profile_id':PROFILE,'scope':'snapchat-profile-api'})})
        self.assertEqual(client.token(),'new');self.assertFalse(s.get('oauth_tokens')['refresh_pending']);self.assertEqual(c.unseal(s.get('oauth_tokens')['encrypted'])['refresh_token'],'rotated')
        self.assertNotIn('rotated',json.dumps(s.values))
    def test_lost_refresh_response_requires_reconnection(self):
        c=config();s=State();s.acquire();s.put('oauth_tokens',{'encrypted':c.seal({'access_token':'a','refresh_token':'r','expires_at':0,'client_id':'client','profile_id':PROFILE})})
        client=SnapClient(c,s,transport=lambda *a,**k:(_ for _ in ()).throw(SnapError('transport')))
        with self.assertRaises(SnapError):client.token()
        with self.assertRaisesRegex(SnapError,'refresh_unconfirmed'):client.token()
    def test_response_html_static_error(self):
        response=type('Response',(),{'status':200,'read':lambda self,n:b'<!doctype html>private'})()
        connection=type('Connection',(),{'request':lambda *a,**k:None,'getresponse':lambda self:response,'close':lambda self:None})()
        with patch('snapchat_feature.client.validate_url',return_value=('businessapi.snapchat.com','1.1.1.1','/x')),patch('snapchat_feature.client.PinnedHTTPS',return_value=connection):
            with self.assertRaisesRegex(SnapError,'response_not_json'):request('https://businessapi.snapchat.com/x')
    def test_probe_rejects_non_mp4_before_subprocess(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'not video');f.flush()
            with patch('snapchat_feature.client.subprocess.run') as run:
                with self.assertRaises(SnapError):probe(f.name)
                run.assert_not_called()


class RouteTests(unittest.TestCase):
    def setUp(self):
        app=FastAPI();app.include_router(router);self.client=TestClient(app)
    def test_auth_required_for_all_operator_routes(self):
        with patch.dict('os.environ',{'MEDIA_CLIP_API_KEY':'private'}):
            for method,path in [('GET','/health'),('POST','/setup'),('POST','/connect'),('POST','/preview'),('POST','/run'),('POST','/tick'),('GET','/metrics'),('POST','/metrics/refresh'),('GET','/jobs/'+'a'*40)]:
                r=self.client.request(method,'/snapchat'+path,json={});self.assertEqual(r.status_code,401,path)
    def test_provider_errors_do_not_leak(self):
        with patch.dict('os.environ',{'MEDIA_CLIP_API_KEY':'private'}),patch('snapchat_feature.router.Service',side_effect=RuntimeError('Authorization=SECRET')):
            r=self.client.get('/snapchat/health',headers={'X-API-KEY':'private'});self.assertNotIn('SECRET',r.text);self.assertEqual(r.status_code,503)
    def test_bad_job_identifier_rejected(self):
        with patch.dict('os.environ',{'MEDIA_CLIP_API_KEY':'private'}):
            r=self.client.get('/snapchat/jobs/not_valid',headers={'X-API-KEY':'private'});self.assertEqual(r.status_code,404)
    def test_callback_never_reflects_code(self):
        with patch('snapchat_feature.router.Service',side_effect=RuntimeError('private-code')):
            r=self.client.get('/snapchat/oauth/callback?code=private-code&state=bad');self.assertEqual(r.status_code,400);self.assertNotIn('private-code',r.text);self.assertEqual(r.headers['referrer-policy'],'no-referrer')


if __name__=='__main__':unittest.main()
