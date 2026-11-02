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
        url = info.get('thumburl') if kind == 'image' else info.get('url')
        url = url or info.get('url', '')
        if urlparse(url).scheme != 'https' or urlparse(url).hostname != 'upload.wikimedia.org':
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
