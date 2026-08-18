from __future__ import annotations

import argparse
import csv
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_DRUID = "qh001qt3946"

DEFAULT_PATTERN = (
    r"(?i)("
    r"summary\.csv$"
    r"|readme"
    r"|metadata"
    r"|data[_\s-]*dictionary"
    r")"
)


def create_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry
        ),
    )

    session.headers.update({
        "User-Agent": (
            "methane-release-research-downloader/1.0"
        )
    })

    return session


def local_tag(element: ET.Element) -> str:
    return element.tag.rsplit(
        "}",
        1,
    )[-1]


def fetch_public_xml(
    session: requests.Session,
    druid: str,
) -> bytes:
    url = (
        f"https://purl.stanford.edu/"
        f"{druid}.xml"
    )

    print(f"Reading manifest:\n{url}")

    response = session.get(
        url,
        timeout=(30, 300),
    )

    response.raise_for_status()

    return response.content


def parse_files(xml_content: bytes) -> list[dict]:
    root = ET.fromstring(
        xml_content
    )

    records = []

    for element in root.iter():
        if local_tag(element) != "file":
            continue

        file_id = (
            element.attrib.get("id")
            or element.attrib.get("filename")
        )

        if not file_id:
            continue

        publish = (
            element.attrib.get(
                "publish",
                "",
            )
            .strip()
            .lower()
        )

        shelve = (
            element.attrib.get(
                "shelve",
                "",
            )
            .strip()
            .lower()
        )

        # 明確標示不公開或不上架的檔案不下載。
        accessible = (
            publish != "no"
            and shelve != "no"
        )

        try:
            size_bytes = int(
                element.attrib.get(
                    "size",
                    "0",
                )
            )
        except ValueError:
            size_bytes = 0

        records.append({
            "file_id": file_id,
            "size_bytes": size_bytes,
            "size_mb": (
                size_bytes
                / 1024
                / 1024
            ),
            "mimetype":
                element.attrib.get(
                    "mimetype",
                    "",
                ),
            "publish": publish,
            "shelve": shelve,
            "accessible": accessible,
        })

    # 移除重複項目。
    unique = {}

    for record in records:
        unique[
            record["file_id"]
        ] = record

    return sorted(
        unique.values(),
        key=lambda item:
            item["file_id"].lower(),
    )


def safe_local_path(
    output_root: Path,
    remote_file_id: str,
) -> Path:
    pure_path = PurePosixPath(
        remote_file_id
    )

    safe_parts = [
        part
        for part in pure_path.parts
        if part not in {
            "",
            ".",
            "..",
            "/",
        }
    ]

    if not safe_parts:
        raise ValueError(
            f"Unsafe file path: "
            f"{remote_file_id}"
        )

    target = output_root.joinpath(
        *safe_parts
    )

    resolved_root = (
        output_root.resolve()
    )

    resolved_target_parent = (
        target.parent.resolve()
    )

    if (
        resolved_root
        not in (
            resolved_target_parent,
            *resolved_target_parent.parents,
        )
    ):
        raise ValueError(
            f"Path escaped output folder: "
            f"{remote_file_id}"
        )

    return target


def candidate_download_urls(
    druid: str,
    remote_file_id: str,
) -> list[str]:
    # Stanford 官方形式通常不包含 druid: 前綴。
    encoded_with_slashes = quote(
        remote_file_id,
        safe="/",
    )

    encoded_all = quote(
        remote_file_id,
        safe="",
    )

    candidates = [
        (
            f"https://stacks.stanford.edu/"
            f"file/{druid}/"
            f"{encoded_with_slashes}"
        ),
        (
            f"https://stacks.stanford.edu/"
            f"file/{druid}/"
            f"{encoded_all}"
        ),
        (
            f"https://stacks.stanford.edu/"
            f"file/druid:{druid}/"
            f"{encoded_with_slashes}"
        ),
        (
            f"https://stacks.stanford.edu/"
            f"file/druid:{druid}/"
            f"{encoded_all}"
        ),
    ]

    # 保持順序並移除重複。
    return list(
        dict.fromkeys(candidates)
    )


def download_one(
    session: requests.Session,
    druid: str,
    record: dict,
    output_root: Path,
    overwrite: bool,
) -> tuple[str, str]:
    remote_file_id = record[
        "file_id"
    ]

    expected_size = record[
        "size_bytes"
    ]

    target = safe_local_path(
        output_root,
        remote_file_id,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if target.exists() and not overwrite:
        actual_size = target.stat().st_size

        if (
            expected_size <= 0
            or actual_size
            == expected_size
        ):
            return (
                "skipped_existing",
                str(target),
            )

        print(
            f"Existing size mismatch, "
            f"downloading again: "
            f"{target}"
        )

    temporary = target.with_suffix(
        target.suffix + ".part"
    )

    last_error = "No URL attempted"

    for url in candidate_download_urls(
        druid,
        remote_file_id,
    ):
        try:
            with session.get(
                url,
                stream=True,
                timeout=(30, 300),
                allow_redirects=True,
            ) as response:
                if response.status_code in {
                    403,
                    404,
                    410,
                }:
                    last_error = (
                        f"HTTP "
                        f"{response.status_code} "
                        f"for {url}"
                    )
                    continue

                response.raise_for_status()

                content_type = (
                    response.headers
                    .get(
                        "Content-Type",
                        "",
                    )
                    .lower()
                )

                # 防止將錯誤 HTML 儲存成 CSV。
                if (
                    "text/html"
                    in content_type
                    and not remote_file_id
                    .lower()
                    .endswith((
                        ".html",
                        ".htm",
                    ))
                ):
                    last_error = (
                        f"Unexpected HTML "
                        f"response from {url}"
                    )
                    continue

                with temporary.open(
                    "wb"
                ) as output_file:
                    for chunk in (
                        response.iter_content(
                            chunk_size=
                                1024 * 1024
                        )
                    ):
                        if chunk:
                            output_file.write(
                                chunk
                            )

            downloaded_size = (
                temporary.stat().st_size
            )

            if (
                expected_size > 0
                and downloaded_size
                != expected_size
            ):
                temporary.unlink(
                    missing_ok=True
                )

                last_error = (
                    "Downloaded size mismatch: "
                    f"expected={expected_size}, "
                    f"actual={downloaded_size}, "
                    f"url={url}"
                )
                continue

            temporary.replace(target)

            return (
                "downloaded",
                str(target),
            )

        except requests.RequestException as error:
            last_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            temporary.unlink(
                missing_ok=True
            )

    return (
        "failed",
        last_error,
    )


def write_manifest(
    records: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "file_id",
        "size_bytes",
        "size_mb",
        "mimetype",
        "publish",
        "shelve",
        "accessible",
        "selected",
        "download_status",
        "local_path_or_error",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record.get(
                    key,
                    "",
                )
                for key in fieldnames
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List and selectively download "
            "public Stanford SDR files."
        )
    )

    parser.add_argument(
        "--druid",
        default=DEFAULT_DRUID,
    )

    parser.add_argument(
        "--match",
        default=DEFAULT_PATTERN,
        help=(
            "Regular expression applied to "
            "the complete remote file path."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "raw_data/"
            "stanford_large_scale_release/"
            "metadata"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/"
            "124_stanford_sdr_"
            "filtered_manifest.csv"
        ),
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Actually download files. "
            "Without this option, only "
            "show the matching file list."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help=(
            "Seconds to wait between "
            "downloads."
        ),
    )

    args = parser.parse_args()

    try:
        pattern = re.compile(
            args.match
        )
    except re.error as error:
        raise SystemExit(
            f"Invalid regex: {error}"
        )

    session = create_session()

    xml_content = fetch_public_xml(
        session,
        args.druid,
    )

    all_records = parse_files(
        xml_content
    )

    selected_records = []

    for record in all_records:
        selected = bool(
            record["accessible"]
            and pattern.search(
                record["file_id"]
            )
        )

        record["selected"] = selected
        record["download_status"] = ""
        record[
            "local_path_or_error"
        ] = ""

        if selected:
            selected_records.append(
                record
            )

    total_size_mb = sum(
        record["size_mb"]
        for record in selected_records
    )

    print("\n" + "=" * 90)
    print("STANFORD SDR FILE SELECTION")
    print("=" * 90)

    print(
        f"Files in XML manifest: "
        f"{len(all_records)}"
    )

    print(
        f"Selected files: "
        f"{len(selected_records)}"
    )

    print(
        f"Selected size: "
        f"{total_size_mb:.3f} MB"
    )

    for number, record in enumerate(
        selected_records,
        start=1,
    ):
        print(
            f"{number:4d}. "
            f"{record['file_id']} "
            f"({record['size_mb']:.4f} MB)"
        )

    if not args.download:
        print(
            "\nList-only mode. "
            "Nothing was downloaded."
        )

        write_manifest(
            all_records,
            args.manifest,
        )

        print(
            f"Manifest saved to: "
            f"{args.manifest}"
        )

        return

    print("\n" + "=" * 90)
    print("DOWNLOADING")
    print("=" * 90)

    downloaded = 0
    skipped = 0
    failed = 0

    for number, record in enumerate(
        selected_records,
        start=1,
    ):
        print(
            f"\n[{number}/"
            f"{len(selected_records)}] "
            f"{record['file_id']}"
        )

        status, detail = download_one(
            session=session,
            druid=args.druid,
            record=record,
            output_root=args.output_dir,
            overwrite=args.overwrite,
        )

        record[
            "download_status"
        ] = status

        record[
            "local_path_or_error"
        ] = detail

        print(
            f"{status}: {detail}"
        )

        if status == "downloaded":
            downloaded += 1
        elif status == "skipped_existing":
            skipped += 1
        else:
            failed += 1

        time.sleep(
            max(
                args.sleep,
                0,
            )
        )

    write_manifest(
        all_records,
        args.manifest,
    )

    print("\n" + "=" * 90)
    print("DOWNLOAD SUMMARY")
    print("=" * 90)

    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")
    print(f"Output:     {args.output_dir}")
    print(f"Manifest:   {args.manifest}")


if __name__ == "__main__":
    main()
