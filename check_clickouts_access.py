"""Read-only Clickouts access check using existing keys. Python standard library only."""
import argparse
import getpass
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check(base, request_id, api_key, reference_token):
    body = {'check_provider': True, 'request_id': request_id,
            'expected_token_sha256': hashlib.sha256(reference_token.encode()).hexdigest()}
    request = urllib.request.Request(base + '/social/clickouts/preview',
        data=json.dumps(body).encode(), method='POST',
        headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(NoRedirects())
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read(262145)
        if len(raw) > 262144:
            payload = {'code': 'response_too_large'}
        else:
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError):
                payload = {'code': 'response_was_not_json'}
        result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                  'http_status': response.code,
                  'headers': {name: response.headers.get(name) for name in (
                      'x-backend-used', 'x-backend-status', 'x-failover-attempts', 'cf-ray')
                      if response.headers.get(name)}, 'body': payload}
    if result['http_status'] == 200 and (not isinstance(payload, dict) or payload.get('check') != 'provider_access'):
        result['client_note'] = 'This backend did not run the access check. Confirm the updated patch is deployed.'
    # Do not expose either key even if a malformed upstream response echoes one.
    def redact(value):
        if isinstance(value, str):
            for secret in (api_key, reference_token):
                if secret:
                    value = value.replace(secret, '[REDACTED]')
            return value
        if isinstance(value, dict):
            return {redact(k): redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value
    return redact(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--base-url', default='https://sh01.vivojaymail.workers.dev',
                        choices=['https://sh01.vivojaymail.workers.dev', 'https://sh01.onrender.com'])
    args = parser.parse_args()
    api_key = getpass.getpass('Existing MEDIA_CLIP_API_KEY: ').strip()
    reference_token = getpass.getpass('Working Meta System User token (local comparison only): ').strip()
    if not api_key or not reference_token:
        raise SystemExit('Both prompts require a value; no request sent.')
    try:
        result = check(args.base_url, args.request_id, api_key, reference_token)
    except Exception:
        raise SystemExit('Read-only check could not reach the server. No deletion request was sent.') from None
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
