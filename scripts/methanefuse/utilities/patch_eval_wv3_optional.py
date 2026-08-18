from pathlib import Path
import re

path = Path("scripts/eval/evaluate_classification.py")
text = path.read_text(encoding="utf-8")

if "skipping WV3 SRF loading" in text:
    print("Evaluation script is already patched.")
    raise SystemExit(0)

# 加入 Python 內建 csv。
if not re.search(r"^import csv\s*$", text, flags=re.MULTILINE):
    future = "from __future__ import annotations\n"

    if future in text:
        text = text.replace(
            future,
            future + "\nimport csv\n",
            1,
        )
    else:
        match = re.search(
            r"^(?:import|from)\s+",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise SystemExit("找不到 import 區域。")

        text = (
            text[:match.start()]
            + "import csv\n"
            + text[match.start():]
        )

lines = text.splitlines(keepends=True)

assign_index = next(
    (
        index
        for index, line in enumerate(lines)
        if (
            "wv3_chn_ids"
            in line
            and "load_wv3_channel_ids_from_srf"
            in line
        )
    ),
    None,
)

if assign_index is None:
    raise SystemExit(
        "找不到 evaluation 的 WV3 channel-ID 載入位置。"
    )

start_index = None

for index in range(
    assign_index,
    max(-1, assign_index - 30),
    -1,
):
    if re.match(
        r"^\s*wv3_band_names\s*=",
        lines[index],
    ):
        start_index = index
        break

if start_index is None:
    raise SystemExit(
        "找不到 wv3_band_names 區塊起點。"
    )

fragment = ""
end_index = None

for index in range(
    assign_index,
    min(len(lines), assign_index + 15),
):
    fragment += lines[index]

    if ".unsqueeze(-1)" in fragment:
        end_index = index + 1
        break

if end_index is None:
    raise SystemExit(
        "找不到 WV3 載入區塊終點。"
    )

indent = re.match(
    r"^(\s*)",
    lines[start_index],
).group(1)

new_block = f'''{indent}def manifest_contains_wv3(csv_path):
{indent}    with open(
{indent}        csv_path,
{indent}        "r",
{indent}        encoding="utf-8-sig",
{indent}        newline="",
{indent}    ) as handle:
{indent}        columns = next(csv.reader(handle))

{indent}    return any(
{indent}        "wv3" in str(column).lower()
{indent}        or "worldview" in str(column).lower()
{indent}        for column in columns
{indent}    )

{indent}has_wv3 = manifest_contains_wv3(args.eval_csv)
{indent}wv3_chn_ids = None

{indent}if has_wv3:
{indent}    wv3_band_names = [
{indent}        band.strip()
{indent}        for band in args.wv3_bands.split(",")
{indent}        if band.strip()
{indent}    ]

{indent}    if not wv3_band_names:
{indent}        raise ValueError(
{indent}            "--wv3_bands must provide at least one "
{indent}            "WV3 band column name"
{indent}        )

{indent}    wv3_chn_ids = load_wv3_channel_ids_from_srf(
{indent}        args.wv3_srf_csv,
{indent}        wv3_band_names,
{indent}    ).unsqueeze(-1)
{indent}else:
{indent}    print(
{indent}        "[WV3] Sentinel-2-only evaluation manifest; "
{indent}        "skipping WV3 SRF loading.",
{indent}        flush=True,
{indent}    )

'''

text = (
    "".join(lines[:start_index])
    + new_block
    + "".join(lines[end_index:])
)

path.write_text(text, encoding="utf-8")
print("Patched:", path)
