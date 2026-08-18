from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "UNEP-IMEO/MARS-S2L"
OUTPUT_DIR = Path(
    "raw_data/MARS-S2L"
)

FILES = [
    "validated_images_all.csv",
    "validated_images_plumes.csv",
    "train.csv",
    "val.csv",
    "test.csv",
]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in FILES:
        print(f"Downloading {filename}...")

        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(OUTPUT_DIR),
        )

        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
