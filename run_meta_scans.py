"""Read dashboard accounts; optionally scan one bounded batch per account sequentially."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, build_opener
from urllib.parse import urlsplit
from read_only_client import NoRedirect

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', action='store_true', help='Without this flag, only discover/list accounts.')
    p.add_argument('--max-posts', type=int, default=3)
    p.add_argument('--checkpoint-dir', type=Path, default=Path.home() / '.sh01-scanner' / 'catalogue-checkpoints-v1')
    a = p.parse_args()
    if not 1 <= a.max_posts <= 20: raise SystemExit('Use 1–20 posts per account.')
    key = os.getenv('SOCIAL_INVENTORY_API_KEY', '').strip() or os.getenv('MEDIA_CLIP_API_KEY', '').strip() or getpass.getpass('Existing media API key (hidden): ')
    if not key: raise SystemExit('API key required.')
    host = os.getenv('SOCIAL_API_URL', 'https://sh01.vivojaymail.workers.dev').rstrip('/')
    u = urlsplit(host)
    if u.scheme != 'https' or not u.hostname or u.username or u.password or u.path or u.query or u.fragment:
        raise SystemExit('HTTPS API origin required.')
    req = Request(host + '/social/inventory/accounts', headers={'X-API-KEY': key})
    try:
        with build_opener(NoRedirect()).open(req, timeout=60) as response: data = json.load(response)
    except Exception:
        raise SystemExit('Dashboard catalogue unavailable. Check authenticated /social/inventory/diagnostics; no scans started.') from None
    rows = data.get('rows')
    if not isinstance(rows, list): raise SystemExit('Invalid discovery response.')
    print(json.dumps(data, indent=2))
    if not a.scan: return
    # Validate all identities before using any as a checkpoint filename.
    for row in rows:
        if row.get('platform') not in ('facebook', 'instagram') or not str(row.get('account_id', '')).isascii() or not str(row.get('account_id', '')).isdigit():
            raise SystemExit('Invalid account identity; no scans started.')
    env = dict(os.environ, SOCIAL_INVENTORY_API_KEY=key)
    for row in rows:
        if row.get('stored_posts') == 0:
            print('No collected posts:', row['platform'], row['account_id']); continue
        checkpoint = a.checkpoint_dir / (row['platform'] + '-' + row['account_id'] + '.json')
        if checkpoint.exists() and json.loads(checkpoint.read_text()).get('complete'):
            print('Catalogue checkpoint complete:', row['platform'], row['account_id']); continue
        print('Scanning:', row['platform'], row['account_id'], flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_social_scan.py')),
                                 row['platform'], row['account_id'], '--batches', '1', '--max-posts', str(a.max_posts),
                                 '--checkpoint', str(checkpoint)], env=env)
        if result.returncode: raise SystemExit('Batch failed; stopped. Existing checkpoints retained.')
    print('One bounded pass finished. Re-run --scan to resume unfinished accounts; this only covers collected dashboard posts. Use a new checkpoint directory for a fresh catalogue pass.')

if __name__ == '__main__': main()
