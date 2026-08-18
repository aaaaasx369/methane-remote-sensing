from pathlib import Path

import download_marss2l_frozen_external_patches as downloader


downloader.INPUT = Path(
    "outputs/227_marss2l_replacement_negative_manifest.csv"
)

# 繼續寫進原本 index，保留完整下載紀錄。
downloader.INDEX_OUTPUT = Path(
    "outputs/226_marss2l_frozen_external_patch_index.csv"
)


if __name__ == "__main__":
    downloader.main()
