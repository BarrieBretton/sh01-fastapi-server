"""Key-free Wikimedia Commons retrieval with provenance and explicit licenses.

Search metadata is candidate evidence, not proof of factual relevance. The local
visual reviewer still reviews the composed frames. Never attach the search query
to candidate descriptions to inflate relevance scores.
"""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse


def plain(value):
    return html.unescape(re.sub(r'<[^>]+>', ' ', str(value))).strip()


def media_download_url(info, kind):
    """Prefer a usable thumbnail, otherwise try the independently checked original.

    A populated thumburl is not necessarily an acceptable download endpoint.
    Keep the media-host restriction; never trust another host merely because
    the URL appeared in API metadata. Protocol-relative media URLs use HTTPS.
    """
    choices = [info.get('thumburl'), info.get('url')] if kind == 'image' else [info.get('url')]
    for value in choices:
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value.startswith('//'):
            value = 'https:' + value
        try:
            parsed = urlparse(value)
            allowed = (parsed.scheme == 'https' and parsed.hostname == 'upload.wikimedia.org'
                       and parsed.username is None and parsed.password is None
                       and parsed.port in (None, 443))
        except ValueError:
            continue
        if allowed:
            return value
    return None


def parse_candidates(payload, kind):
    results = []
    for page in payload.get('query', {}).get('pages', {}).values():
        info = (page.get('imageinfo') or [{}])[0]
        meta = info.get('extmetadata', {})
        get = lambda name: plain(meta.get(name, {}).get('value', ''))
        license_name = get('LicenseShortName')
        # Only familiar reuse licenses, with explicit provenance retained.
        if not re.match(r'^(CC0|CC BY(?:-SA)?(?: |$)|Public domain)', license_name, re.I):
            continue
        mime = info.get('mime', '')
        if kind == 'video' and mime not in {'video/webm', 'video/ogg', 'video/mp4'}:
            continue
        if kind == 'image' and mime not in {'image/jpeg', 'image/png', 'image/webp'}:
            continue
        if info.get('size', 0) > 80*1024*1024 or min(info.get('width', 0), info.get('height', 0)) < 240:
            continue
        url = media_download_url(info, kind)
        if not url:
            continue
        results.append({'asset_id':'commons_'+str(page['pageid']), 'source':'wikimedia',
            'title':plain(page.get('title', '')), 'description':get('ImageDescription'),
            'download_url':url, 'source_url':info.get('descriptionurl'), 'license':license_name,
            'license_url':get('LicenseUrl'), 'attribution':get('Artist')+'; '+get('Credit'),
            'kind':kind, 'tags':get('Categories')})
    return results


def search(query, kind, limit=8):
    import requests
    response = requests.get('https://commons.wikimedia.org/w/api.php',
        headers={'User-Agent':'SH01LocalEditor/2.0 (https://github.com/BarrieBretton/sh01-fastapi-server)'},
        params={'action':'query','format':'json','generator':'search',
                'gsrsearch':query+' filetype:' + ('video' if kind == 'video' else 'bitmap'),
                'gsrnamespace':6,'gsrlimit':min(20,limit),'prop':'imageinfo',
                'iiprop':'url|mime|size|extmetadata','iiurlwidth':1280},timeout=(15,40))
    response.raise_for_status()
    payload = response.json()
    if 'error' in payload:
        raise RuntimeError('Commons search: '+str(payload['error'].get('code','unknown error')))
    return parse_candidates(payload,kind)
