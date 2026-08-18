from pathlib import Path
from urllib.parse import quote
import requests
import time

DRUID = "qh001qt3946"

outdir = Path("raw")
outdir.mkdir(exist_ok=True)

names = [
    x.strip()
    for x in Path("stanford_file_list.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

print("Files to download:", len(names))
print()

ok = 0
fail = 0

for i, name in enumerate(names, 1):

    encoded_name = quote(name, safe="/")

    url = (
        f"https://stacks.stanford.edu/file/"
        f"{DRUID}/{encoded_name}"
    )

    dest = outdir / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{i}/{len(names)}]")
    print("FILE:", name)

    try:
        with requests.get(
            url,
            stream=True,
            timeout=120,
            allow_redirects=True
        ) as r:

            print("HTTP:", r.status_code)
            print("TYPE:", r.headers.get("content-type"))
            print("SIZE:", r.headers.get("content-length"))

            r.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        size = dest.stat().st_size

        if size == 0:
            print("FAIL: zero-byte file")
            dest.unlink(missing_ok=True)
            fail += 1
        else:
            print("PASS:", size, "bytes")
            ok += 1

    except Exception as e:
        print("FAIL:", repr(e))
        fail += 1

    print()
    time.sleep(0.3)

print("=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print("PASS:", ok)
print("FAIL:", fail)
print("TOTAL:", len(names))
