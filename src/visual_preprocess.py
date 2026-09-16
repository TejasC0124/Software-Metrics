from pathlib import Path
import pandas as pd
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HANDWRITING_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "handwriting"
    / "extracted"
    / "Gambo"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SPLITS = {
    "Train": [
        "Corrected",
        "Normal",
        "Reversal"
    ],
    "Test": [
        "Corrected",
        "Normal",
        "Reversal"
    ]
}

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# CHECK ONE IMAGE
# ============================================================

def validate_image(image_path):
    """
    Check whether an image can be opened successfully.

    Returns:
        valid, width, height, mode
    """

    try:

        with Image.open(image_path) as image:

            # Force image loading so corrupted files
            # are detected.
            image.load()

            width, height = image.size
            mode = image.mode

        return True, width, height, mode

    except Exception:

        return False, None, None, None


# ============================================================
# SCAN DATASET
# ============================================================

def scan_dataset():

    records = []

    corrupted_count = 0
    total_count = 0

    print("=" * 60)
    print("SPBE HANDWRITING DATASET SCANNER")
    print("=" * 60)

    print(f"\nDataset location:")
    print(HANDWRITING_DIR)

    if not HANDWRITING_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n{HANDWRITING_DIR}"
        )

    # --------------------------------------------------------
    # Process Train and Test
    # --------------------------------------------------------

    for split, classes in SPLITS.items():

        print(f"\nProcessing {split} set...")

        for label in classes:

            class_dir = HANDWRITING_DIR / split / label

            if not class_dir.exists():

                print(
                    f"WARNING: Folder not found: {class_dir}"
                )

                continue

            image_files = [
                path
                for path in class_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ]

            print(
                f"{split:5s} | "
                f"{label:10s} | "
                f"{len(image_files):6d} images"
            )

            for image_path in image_files:

                total_count += 1

                valid, width, height, mode = (
                    validate_image(image_path)
                )

                if not valid:

                    corrupted_count += 1

                    print(
                        f"Corrupted image: {image_path}"
                    )

                    continue

                records.append({

                    "image_path": str(
                        image_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                    "filename": image_path.name,

                    "split": split,

                    "label": label,

                    "width": width,

                    "height": height,

                    "mode": mode
                })

    print("\n" + "=" * 60)

    print(
        f"Total image files found: {total_count}"
    )

    print(
        f"Valid images: {len(records)}"
    )

    print(
        f"Corrupted images: {corrupted_count}"
    )

    print("=" * 60)

    return pd.DataFrame(records)


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(data):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DIR
        / "SPBE_handwriting_metadata.csv"
    )

    data.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nMetadata saved to:\n{output_file}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(data):

    print("\n" + "=" * 60)
    print("HANDWRITING DATASET SUMMARY")
    print("=" * 60)

    print("\nImages by split and label:")

    summary = (
        data
        .groupby(["split", "label"])
        .size()
        .reset_index(name="count")
    )

    print(summary.to_string(index=False))

    print("\nImage dimensions:")

    dimensions = (
        data
        .groupby(["width", "height"])
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        dimensions.head(20).to_string(
            index=False
        )
    )

    print("\nImage modes:")

    print(
        data["mode"]
        .value_counts()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    data = scan_dataset()

    if data.empty:

        raise RuntimeError(
            "No valid handwriting images were found."
        )

    save_metadata(data)

    print_summary(data)

    print("\n" + "=" * 60)
    print("HANDWRITING PREPROCESSING METADATA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()