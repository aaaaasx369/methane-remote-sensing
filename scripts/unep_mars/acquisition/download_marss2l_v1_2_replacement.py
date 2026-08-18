from pathlib import Path

import download_marss2l_frozen_external_patches as downloader


downloader.INPUT = Path(
    "outputs/"
    "231_marss2l_remaining_replacement_manifest.csv"
)

downloader.INDEX_OUTPUT = Path(
    "outputs/"
    "226_marss2l_frozen_external_patch_index.csv"
)


if __name__ == "__main__":
    downloader.main()
