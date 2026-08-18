from pathlib import Path
from urllib.parse import urlparse
import time
import os

import pandas as pd
import earthaccess
import h5py


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/Volumes/engg-leung/dora lin/MARS_MultiSensor_Data/EMIT"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "mars_emit_nasa_match_inventory.csv"
)

OUT = (
    ROOT
    / "L2A_RFL"
)

CLOUD_LOG = (
    ROOT
    / "logs"
    / "emit_l2a_download_report_v2.csv"
)

LOCAL_LOG = (
    Path.home()
    / "methane_release_project"
    / "emit_l2a_download_report_local_v2.csv"
)

SHORT_NAME = "EMITL2ARFL"
VERSION = "001"

MAX_RETRIES = 5


# ============================================================
# HELPERS
# ============================================================

def smb_alive():

    return (
        Path("/Volumes/engg-leung").exists()
        and ROOT.exists()
    )


def stop_if_smb_dead():

    if not smb_alive():

        print()
        print("=" * 80)
        print("SMB SHARE DISCONNECTED")
        print("=" * 80)
        print()
        print("Reconnect:")
        print(
            "smb://smb.research-filer.ualberta.ca/engg-leung"
        )
        print()
        print(
            "Then run this SAME script again."
        )

        raise SystemExit(2)


def netcdf_readable(path):

    path = Path(path)

    if not path.exists():
        return False

    try:

        if path.stat().st_size < 1024:
            return False

        if not h5py.is_hdf5(path):
            return False

        with h5py.File(
            path,
            "r"
        ) as f:

            # Simply opening the HDF5/NetCDF4 structure
            # is enough for our disk-integrity check.
            _ = list(f.keys())

        return True

    except Exception:

        return False


def granule_ur(g):

    try:

        return str(
            g.get(
                "umm",
                {}
            ).get(
                "GranuleUR"
            )
        )

    except Exception:

        return ""


def expected_nc_files(g):

    names = []

    try:

        links = g.data_links()

    except Exception:

        links = []

    for url in links:

        try:

            name = Path(
                urlparse(
                    str(url)
                ).path
            ).name

        except Exception:

            continue

        if name.lower().endswith(
            ".nc"
        ):

            names.append(
                name
            )

    # preserve order but remove duplicates
    return list(
        dict.fromkeys(
            names
        )
    )


# ============================================================
# PREFLIGHT
# ============================================================

print("=" * 80)
print("MARS -> NASA EMIT L2A RFL V001")
print("SMB-SAFE DOWNLOAD V2")
print("=" * 80)

stop_if_smb_dead()

if not MANIFEST.exists():

    raise FileNotFoundError(
        f"Missing manifest:\n{MANIFEST}"
    )

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CLOUD_LOG.parent.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("Destination:")
print(OUT)


# ============================================================
# NASA AUTH
# ============================================================

print()
print("NASA Earthdata authentication...")

earthaccess.login()

print("Authentication ready.")


# ============================================================
# LOAD MATCH INVENTORY
# ============================================================

df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

selected = df[
    (df["product_type"] == "L2A_RFL")
    &
    (df["status"] == "MATCHED")
].copy()

unique = (
    selected
    .dropna(
        subset=[
            "granule_name"
        ]
    )
    .drop_duplicates(
        "granule_name"
    )
    .reset_index(
        drop=True
    )
)

print()
print(
    "MARS targets:",
    len(selected)
)

print(
    "Unique NASA L2A granules:",
    len(unique)
)

estimated_gb = (
    unique[
        "granule_size_mb"
    ]
    .fillna(0)
    .sum()
    / 1024
)

print(
    "Estimated archive size:",
    round(
        estimated_gb,
        2
    ),
    "GB"
)


# ============================================================
# LOG
# ============================================================

records = []

if LOCAL_LOG.exists():

    try:

        old = pd.read_csv(
            LOCAL_LOG,
            low_memory=False
        )

        records = (
            old.to_dict(
                "records"
            )
        )

        print(
            "Existing local log rows:",
            len(records)
        )

    except Exception as e:

        print(
            "WARNING: old local log "
            "could not be loaded:",
            repr(e)
        )


def save_log():

    if not records:
        return

    table = pd.DataFrame(
        records
    )

    # Always save locally first.
    table.to_csv(
        LOCAL_LOG,
        index=False
    )

    if not smb_alive():
        return

    try:

        tmp = Path(
            str(CLOUD_LOG)
            + ".tmp"
        )

        table.to_csv(
            tmp,
            index=False
        )

        os.replace(
            tmp,
            CLOUD_LOG
        )

    except Exception as e:

        print(
            "WARNING: cloud log write failed:",
            repr(e)
        )


# ============================================================
# DOWNLOAD GRANULES
# ============================================================

for idx, row in unique.iterrows():

    stop_if_smb_dead()

    name = str(
        row[
            "granule_name"
        ]
    )

    expected_size_mb = (
        row.get(
            "granule_size_mb"
        )
    )

    print()
    print("=" * 80)
    print(
        f"[{idx+1}/{len(unique)}]"
    )
    print(
        "Granule:",
        name
    )
    print(
        "Inventory size:",
        expected_size_mb,
        "MB"
    )
    print("=" * 80)


    # ========================================================
    # EXACT NASA CMR SEARCH
    # ========================================================

    try:

        matches = earthaccess.search_data(
            short_name=
                SHORT_NAME,

            version=
                VERSION,

            granule_name=
                name,

            downloadable=
                True,

            count=
                10,
        )

    except Exception as e:

        print(
            "CMR search error:",
            repr(e)
        )

        records.append({
            "granule_name":
                name,

            "status":
                "SEARCH_ERROR",

            "expected_size_mb":
                expected_size_mb,

            "error":
                repr(e),
        })

        save_log()

        continue


    if not matches:

        print(
            "NO NASA MATCH"
        )

        records.append({
            "granule_name":
                name,

            "status":
                "NO_MATCH",

            "expected_size_mb":
                expected_size_mb,

            "error":
                None,
        })

        save_log()

        continue


    exact = None

    for g in matches:

        if granule_ur(g) == name:

            exact = g
            break


    if exact is None:

        exact = matches[0]

        print(
            "WARNING: exact GranuleUR "
            "string not found; using first "
            "CMR result."
        )


    # ========================================================
    # EXPECTED THREE NETCDF FILES
    # ========================================================

    expected_names = (
        expected_nc_files(
            exact
        )
    )

    print(
        "Expected NetCDF files:",
        len(expected_names)
    )

    for n in expected_names:
        print(
            "  ",
            n
        )


    if len(expected_names) != 3:

        print(
            "WARNING: expected 3 NetCDF4 "
            "files for EMIT L2A, but NASA "
            "metadata returned",
            len(expected_names)
        )


    # ========================================================
    # AUDIT WHAT IS ALREADY ON DISK
    # ========================================================

    valid_before = []
    missing_before = []
    invalid_before = []

    for filename in expected_names:

        path = (
            OUT
            / filename
        )

        if not path.exists():

            missing_before.append(
                filename
            )

        elif netcdf_readable(
            path
        ):

            valid_before.append(
                filename
            )

        else:

            invalid_before.append(
                filename
            )


    print()
    print(
        "Already valid:",
        len(valid_before)
    )

    print(
        "Missing:",
        len(missing_before)
    )

    print(
        "Invalid/partial:",
        len(invalid_before)
    )


    # ========================================================
    # GRANULE ALREADY COMPLETE
    # ========================================================

    if (
        len(expected_names) == 3
        and
        len(valid_before) == 3
    ):

        print(
            "✅ SKIP — complete 3/3 "
            "NetCDF granule already present."
        )

        total_bytes = sum(
            (
                OUT
                / filename
            ).stat().st_size
            for filename
            in expected_names
        )

        records.append({
            "granule_name":
                name,

            "status":
                "EXISTING_VALID",

            "expected_file_count":
                3,

            "valid_file_count":
                3,

            "local_bytes":
                total_bytes,

            "attempts":
                0,

            "error":
                None,
        })

        save_log()

        continue


    # ========================================================
    # REMOVE ONLY CORRUPT/PARTIAL EXISTING FILES
    # ========================================================

    for filename in invalid_before:

        path = (
            OUT
            / filename
        )

        print(
            "Removing invalid partial:",
            filename
        )

        try:

            path.unlink()

        except Exception as e:

            print(
                "Could not remove:",
                repr(e)
            )


    # ========================================================
    # DOWNLOAD WITH RETRIES
    # ========================================================

    success = False
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        stop_if_smb_dead()

        try:

            print()
            print(
                f"Download attempt "
                f"{attempt}/{MAX_RETRIES}"
            )


            downloaded = earthaccess.download(
                exact,

                local_path=
                    str(OUT),

                threads=
                    1,

                show_progress=
                    True,

                force=
                    False,
            )


            stop_if_smb_dead()


            # =================================================
            # POST-DOWNLOAD DISK AUDIT
            # =================================================

            valid_after = []
            invalid_after = []
            missing_after = []

            for filename in expected_names:

                path = (
                    OUT
                    / filename
                )

                if not path.exists():

                    missing_after.append(
                        filename
                    )

                elif netcdf_readable(
                    path
                ):

                    valid_after.append(
                        filename
                    )

                else:

                    invalid_after.append(
                        filename
                    )


            print()
            print(
                "POST-DOWNLOAD AUDIT"
            )

            print(
                "Valid:",
                len(valid_after),
                "/",
                len(expected_names)
            )

            print(
                "Missing:",
                len(missing_after)
            )

            print(
                "Invalid:",
                len(invalid_after)
            )


            if (
                len(expected_names) == 3
                and
                len(valid_after) == 3
            ):

                total_bytes = sum(
                    (
                        OUT
                        / filename
                    ).stat().st_size
                    for filename
                    in expected_names
                )

                print()
                print(
                    "✅ GRANULE COMPLETE 3/3"
                )

                print(
                    "Local size:",
                    round(
                        total_bytes
                        / 1024
                        / 1024
                        / 1024,
                        3
                    ),
                    "GB"
                )

                records.append({
                    "granule_name":
                        name,

                    "status":
                        "DOWNLOADED_VALID",

                    "expected_file_count":
                        3,

                    "valid_file_count":
                        3,

                    "local_bytes":
                        total_bytes,

                    "attempts":
                        attempt,

                    "error":
                        None,
                })

                save_log()

                success = True
                break


            # -----------------------------------------------
            # Remove only newly identified invalid files
            # before another retry.
            # -----------------------------------------------

            for filename in invalid_after:

                path = (
                    OUT
                    / filename
                )

                try:

                    print(
                        "Removing invalid retry file:",
                        filename
                    )

                    path.unlink()

                except Exception:
                    pass


            raise RuntimeError(
                "Granule incomplete after download: "
                f"valid={len(valid_after)}, "
                f"missing={len(missing_after)}, "
                f"invalid={len(invalid_after)}"
            )


        except Exception as e:

            last_error = repr(e)

            print()
            print(
                "Attempt failed:",
                last_error
            )


            if not smb_alive():

                save_log()

                print()
                print("=" * 80)
                print("SMB SHARE DISCONNECTED")
                print("=" * 80)
                print()
                print(
                    "Reconnect engg-leung."
                )

                print(
                    "Then run the SAME command again."
                )

                raise SystemExit(2)


            if attempt < MAX_RETRIES:

                wait = min(
                    60,
                    5
                    * 2
                    ** (
                        attempt - 1
                    )
                )

                print(
                    f"Retrying in {wait}s..."
                )

                time.sleep(
                    wait
                )


    # ========================================================
    # FINAL FAILURE FOR THIS GRANULE
    # ========================================================

    if not success:

        records.append({
            "granule_name":
                name,

            "status":
                "FAILED",

            "expected_file_count":
                3,

            "valid_file_count":
                None,

            "local_bytes":
                None,

            "attempts":
                MAX_RETRIES,

            "error":
                last_error,
        })

        save_log()


# ============================================================
# FINAL DISK AUDIT — ALL 152 GRANULES
# ============================================================

print()
print("=" * 80)
print("FINAL EMIT L2A DISK AUDIT")
print("=" * 80)

complete = 0
incomplete = 0

audit_records = []


for _, row in unique.iterrows():

    name = str(
        row[
            "granule_name"
        ]
    )

    try:

        matches = earthaccess.search_data(
            short_name=
                SHORT_NAME,

            version=
                VERSION,

            granule_name=
                name,

            downloadable=
                True,

            count=
                10,
        )

        exact = None

        for g in matches:

            if granule_ur(g) == name:
                exact = g
                break

        if (
            exact is None
            and matches
        ):
            exact = matches[0]


        if exact is None:

            audit_records.append({
                "granule_name":
                    name,

                "complete":
                    False,

                "valid_files":
                    0,
            })

            incomplete += 1
            continue


        names = expected_nc_files(
            exact
        )

        valid = sum(
            netcdf_readable(
                OUT / filename
            )
            for filename
            in names
        )

        is_complete = (
            len(names) == 3
            and valid == 3
        )

        if is_complete:
            complete += 1
        else:
            incomplete += 1


        audit_records.append({
            "granule_name":
                name,

            "expected_files":
                len(names),

            "valid_files":
                valid,

            "complete":
                is_complete,
        })


    except Exception as e:

        incomplete += 1

        audit_records.append({
            "granule_name":
                name,

            "expected_files":
                None,

            "valid_files":
                None,

            "complete":
                False,

            "error":
                repr(e),
        })


audit = pd.DataFrame(
    audit_records
)

AUDIT_CSV = (
    ROOT
    / "logs"
    / "emit_l2a_final_disk_audit.csv"
)

audit.to_csv(
    AUDIT_CSV,
    index=False
)


print()
print(
    "Expected unique granules:",
    len(unique)
)

print(
    "Complete 3-file granules:",
    complete
)

print(
    "Incomplete granules:",
    incomplete
)

print(
    "Expected NetCDF files:",
    len(unique) * 3
)

print(
    "Complete granule NetCDF files:",
    complete * 3
)

print()
print(
    "Destination:"
)

print(
    OUT
)

print()
print(
    "Audit:"
)

print(
    AUDIT_CSV
)


if (
    complete == len(unique)
    and
    incomplete == 0
):

    print()
    print(
        "✅ ALL 152 EMIT L2A GRANULES COMPLETE"
    )

    print(
        "✅ 456/456 NetCDF4 files represented "
        "by complete granules"
    )

else:

    print()
    print(
        "⚠ EMIT L2A DOWNLOAD NOT YET COMPLETE"
    )

    print(
        "Reconnect storage/network if needed "
        "and run this SAME script again."
    )

