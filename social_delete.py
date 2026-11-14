"""Owned-post deletion with a mandatory dry-run, explicit confirmation and runtime opt-in.

This module is installed beside social_inventory.py. Never enabled by installation.
"""
from __future__ import annotations
import asyncio, hmac, json, os, re
from typing import Literal
from urllib.parse import urlencode, urlsplit, quote
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
try:
    from .social_inventory import SafeError, Budget, SafeHTTP, Provider, Dashboard, config_for, env_secret, validate_url, PinnedHTTPS
except ImportError:
    from social_inventory import SafeError, Budget, SafeHTTP, Provider, Dashboard, config_for, env_secret, validate_url, PinnedHTTPS

router = APIRouter(prefix='/social/posts', tags=['social post cleanup'])
class DeleteRequest(BaseModel):
    platform: Literal['instagram','facebook','tumblr']
    permalink: str = Field(max_length=2000)
    account_id: str | None = Field(default=None,max_length=160)
    dry_run: bool = True
    confirmation: str | None = Field(default=None,max_length=500)

def normalized_link(value):
    u=urlsplit(value)
    if u.scheme!='https' or u.username or u.password:
        raise SafeError('invalid_permalink')
    host=(u.hostname or '').lower().replace('www.','',1)
    # Facebook query IDs can be the post identity; never drop them when comparing.
    query=u.query if host.endswith('facebook.com') else ''
    return (host,u.path.rstrip('/'),query)

def owned_post(provider, post):
    cfg=provider.cfg;pid=post['post_id'];platform=cfg['platform']
    if not re.fullmatch(r'[0-9_]+',pid):raise SafeError('unsupported_provider_post_id')
    if platform=='instagram':
        data=provider.get(f'{provider.version}/{pid}',{'fields':'id,owner,permalink'})
        if str((data.get('owner') or {}).get('id'))!=cfg['provider_id']:raise SafeError('owner_not_verified')
        if normalized_link(data.get('permalink',''))!=normalized_link(post['permalink']):raise SafeError('permalink_mismatch')
    elif platform=='facebook':
        data=provider.get(f'{provider.version}/{pid}',{'fields':'id,from,permalink_url'})
        if str((data.get('from') or {}).get('id'))!=cfg['provider_id']:raise SafeError('owner_not_verified')
        if normalized_link(data.get('permalink_url',''))!=normalized_link(post['permalink']):raise SafeError('permalink_mismatch')
    else:
        data=provider.get(f"v2/blog/{quote(cfg['provider_id'],safe='')}/posts",{'id':pid})
        posts=data.get('response',{}).get('posts',[])
        if len(posts)!=1 or str(posts[0].get('id_string',posts[0].get('id')))!=pid:raise SafeError('owner_or_post_not_verified')
        expected=cfg.get('handle','').lower().removesuffix('.tumblr.com')
        if str(posts[0].get('blog_name','')).lower()!=expected:raise SafeError('owner_not_verified')
    return True

def perform_delete(provider, post):
    cfg=provider.cfg;pid=post['post_id']
    if cfg['platform']!='tumblr':
        result=provider.http.json(f'https://graph.facebook.com/{provider.version}/{pid}',('graph.facebook.com',),method='DELETE',headers={'Authorization':'Bearer '+env_secret(cfg.get('token_env',''))})
        if result.get('success') is not True:raise SafeError('provider_did_not_confirm_deletion')
        return
    url=f"https://api.tumblr.com/v2/blog/{quote(cfg['provider_id'],safe='')}/post/delete"
    form=urlencode({'id':pid});headers={'Content-Type':'application/x-www-form-urlencoded','User-Agent':'SH01-Post-Cleanup/1'}
    if cfg.get('token_env'):headers['Authorization']='Bearer '+env_secret(cfg['token_env'])
    else:
        from oauthlib.oauth1 import Client,SIGNATURE_TYPE_AUTH_HEADER
        m=cfg.get('oauth_env',{});client=Client(env_secret(m.get('consumer_key','')),client_secret=env_secret(m.get('consumer_secret','')),resource_owner_key=env_secret(m.get('token','')),resource_owner_secret=env_secret(m.get('token_secret','')),signature_type=SIGNATURE_TYPE_AUTH_HEADER)
        _,headers,form=client.sign(url,http_method='POST',body=form,headers=headers)
    host,address,path=validate_url(url,('api.tumblr.com',));conn=PinnedHTTPS(host,address,12)
    try:
        conn.request('POST',path,body=form,headers=headers);response=conn.getresponse();raw=response.read(100000)
        if response.status!=200:raise SafeError('tumblr_delete_rejected')
        data=json.loads(raw)
        if data.get('meta',{}).get('status')!=200:raise SafeError('tumblr_delete_not_confirmed')
    except SafeError:raise
    except Exception:raise SafeError('delete_outcome_unknown_verify_provider_before_retry') from None
    finally:conn.close()

def delete_post(request:DeleteRequest,*,dashboard=None,provider_factory=Provider):
    http=SafeHTTP(Budget(seconds=45,max_bytes=2*1024*1024));dashboard=dashboard or Dashboard(http)
    result=dashboard.http.json(dashboard.base+'/api/ingest/resolve-post',dashboard.hosts,method='POST',headers=dashboard.headers,payload={'platform':request.platform,'permalink':request.permalink})
    post=result['post'];account=post['account_id']
    if request.account_id and request.account_id!=account:raise SafeError('account_selection_mismatch')
    cfg=config_for(account,request.platform);provider=provider_factory(cfg,http);owned_post(provider,post)
    confirmation=f"DELETE {request.platform}:{account}:{post['post_id']}"
    result={'platform':request.platform,'account_id':account,'post_id':post['post_id'],'permalink':post['permalink'],'confirmation':confirmation,'status':'preview','provider_permissions':'read_verified_delete_permission_unverified'}
    if request.dry_run:return result
    if os.environ.get('SOCIAL_DELETE_ENABLED')!='1' or cfg.get('allow_delete') is not True:raise SafeError('deletion_not_enabled_for_account')
    if not request.confirmation or not hmac.compare_digest(request.confirmation,confirmation):raise SafeError('deletion_confirmation_required')
    if post.get('delete_status')=='deleted':return {**result,'status':'already_recorded_deleted'}
    # A transport failure after DELETE is ambiguous. No automatic retry is performed.
    try:perform_delete(provider,post)
    except Exception:
        # A provider rejection/transport loss needs review before another destructive attempt.
        if post.get('request_id'):
            try:dashboard.http.json(dashboard.base+'/api/ingest/deletion-result',dashboard.hosts,method='POST',headers=dashboard.headers,payload={'request_id':post['request_id'],'status':'manual_required','reason':'delete_outcome_unconfirmed_verify_provider_before_retry'})
            except Exception:pass
        raise SafeError('delete_outcome_unknown_verify_provider_before_retry') from None
    result.update(status='deleted',provider_permissions='delete_succeeded')
    if post.get('request_id'):
        try:dashboard.http.json(dashboard.base+'/api/ingest/deletion-result',dashboard.hosts,method='POST',headers=dashboard.headers,payload={'request_id':post['request_id'],'status':'deleted','reason':'provider_confirmed'})
        except Exception:result['dashboard_ack']='failed_provider_deletion_succeeded'
    return result

@router.post('/delete')
async def delete_route(request:DeleteRequest,x_api_key:str|None=Header(default=None,alias='X-API-KEY')):
    expected=os.environ.get('SOCIAL_INVENTORY_API_KEY','')
    if not expected:raise HTTPException(status_code=503,detail='social_api_not_configured')
    if not x_api_key or not hmac.compare_digest(expected,x_api_key):raise HTTPException(status_code=401,detail='invalid_api_key')
    try:return await asyncio.to_thread(delete_post,request)
    except SafeError as exc:raise HTTPException(status_code=400,detail=str(exc)) from None
    except Exception:raise HTTPException(status_code=503,detail='post_cleanup_failed') from None
