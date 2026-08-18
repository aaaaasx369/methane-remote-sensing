from pathlib import Path

import download_marss2l_frozen_external_patches as downloader


downloader.INPUT = Path(
    "outputs/"
    "259_marss2l_development_replacements_round1.csv"
)

downloader.OUTPUT_DIR = Path(
    "sample_patches/"
    "marss2l_development_landsat"
)

downloader.QA_OUTPUT_DIR = Path(
    "sample_patches/"
    "marss2l_development_landsat_qa"
)

# 繼續寫入原本的下載 index。
downloader.INDEX_OUTPUT = Path(
    "outputs/"
    "257_marss2l_development_patch_index.csv"
)


if __name__ == "__main__":
    downloader.main()
