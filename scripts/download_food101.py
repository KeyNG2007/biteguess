"""Download the official archive, resume only when the server accepts ranges."""
from pathlib import Path
import urllib.request
import time

URL = 'https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
DEST = Path(__file__).resolve().parents[1] / 'data/raw/food-101.tar.gz'

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    size = DEST.stat().st_size if DEST.exists() else 0
    headers = {'Range': f'bytes={size}-'} if size else {}
    with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=60) as response:
        resumed = size > 0 and response.status == 206
        if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
            raise RuntimeError('Unexpected range response')
        mode = 'ab' if resumed else 'wb'
        done = size if resumed else 0
        length = response.headers.get('Content-Length')
        total = done + int(length) if length else None
        print(f'Downloading {DEST}; total={total}', flush=True)
        last = time.monotonic()
        with DEST.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                done += len(chunk)
                if time.monotonic() - last >= 15:
                    print(f'{done/1024**2:.1f} MiB received', flush=True)
                    last = time.monotonic()
        if total is not None and done != total:
            raise RuntimeError(f'Incomplete download: {done}/{total}')
        print(f'Download finished: {done} bytes. Archive validation still required.', flush=True)

if __name__ == '__main__':
    main()
