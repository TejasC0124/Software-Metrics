from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DESKTOP_FILE = RAW_DIR / "Dyt-desktop.csv"
TABLET_FILE = RAW_DIR / "Dyt-tablet.csv"


def load_dyt_data():

    desktop = pd.read_csv(
        DESKTOP_FILE,
        sep=";"
    )

    tablet = pd.read_csv(
        TABLET_FILE,
        sep=";"
    )

    desktop["Device"] = "Desktop"
    tablet["Device"] = "Tablet"

    data = pd.concat(
        [desktop, tablet],
        ignore_index=True
    )

    return data


def clean_missing_values(data):

    missing_values = [
        "",
        " ",
        "NULL",
        "null",
        "None",
        "none",
        "NA",
        "N/A",
        "na",
        "n/a"
    ]

    data = data.replace(
        missing_values,
        np.nan
    )

    return data



def remove_duplicates(data):

    before = len(data)

    data = data.drop_duplicates()

    after = len(data)

    print(
        f"Removed duplicates: {before - after}"
    )

    return data


def convert_numeric_columns(data):

    numeric_prefixes = (
        "Clicks",
        "Hits",
        "Misses",
        "Score",
        "Accuracy",
        "Missrate"
    )

    for column in data.columns:

        if column.startswith(numeric_prefixes):

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


def clean_dyslexia_label(data):

    if "Dyslexia" not in data.columns:
        raise ValueError(
            "Dyslexia column was not found."
        )

    data["Dyslexia"] = (
        data["Dyslexia"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    label_mapping = {
        "yes": 1,
        "y": 1,
        "1": 1,
        "true": 1,

        "no": 0,
        "n": 0,
        "0": 0,
        "false": 0
    }

    data["Dyslexia"] = data["Dyslexia"].map(
        label_mapping
    )

    return data


# ============================================================
# BEHAVIORAL FEATURE ENGINEERING
# ============================================================

def create_behavioral_features(data):

    accuracy_columns = [
        f"Accuracy{i}"
        for i in range(1, 33)
        if f"Accuracy{i}" in data.columns
    ]

    missrate_columns = [
        f"Missrate{i}"
        for i in range(1, 33)
        if f"Missrate{i}" in data.columns
    ]

    score_columns = [
        f"Score{i}"
        for i in range(1, 33)
        if f"Score{i}" in data.columns
    ]

    clicks_columns = [
        f"Clicks{i}"
        for i in range(1, 33)
        if f"Clicks{i}" in data.columns
    ]

    hits_columns = [
        f"Hits{i}"
        for i in range(1, 33)
        if f"Hits{i}" in data.columns
    ]

    misses_columns = [
        f"Misses{i}"
        for i in range(1, 33)
        if f"Misses{i}" in data.columns
    ]

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    data["Mean_Accuracy"] = data[
        accuracy_columns
    ].mean(axis=1)

    data["Mean_Missrate"] = data[
        missrate_columns
    ].mean(axis=1)

    data["Mean_Score"] = data[
        score_columns
    ].mean(axis=1)

    # --------------------------------------------------------
    # Interaction behavior
    # --------------------------------------------------------

    data["Total_Clicks"] = data[
        clicks_columns
    ].sum(axis=1)

    data["Total_Hits"] = data[
        hits_columns
    ].sum(axis=1)

    data["Total_Misses"] = data[
        misses_columns
    ].sum(axis=1)

    # --------------------------------------------------------
    # Performance variability
    # --------------------------------------------------------

    data["Accuracy_Std"] = data[
        accuracy_columns
    ].std(axis=1)

    data["Score_Std"] = data[
        score_columns
    ].std(axis=1)

    data["Missrate_Std"] = data[
        missrate_columns
    ].std(axis=1)

    # --------------------------------------------------------
    # Early vs late performance
    # --------------------------------------------------------

    early_accuracy = [
        f"Accuracy{i}"
        for i in range(1, 9)
        if f"Accuracy{i}" in data.columns
    ]

    late_accuracy = [
        f"Accuracy{i}"
        for i in range(25, 33)
        if f"Accuracy{i}" in data.columns
    ]

    data["Early_Accuracy"] = data[
        early_accuracy
    ].mean(axis=1)

    data["Late_Accuracy"] = data[
        late_accuracy
    ].mean(axis=1)

    data["Accuracy_Improvement"] = (
        data["Late_Accuracy"]
        - data["Early_Accuracy"]
    )

    # --------------------------------------------------------
    # Overall error rate
    # --------------------------------------------------------

    if (
        "Total_Hits" in data.columns
        and "Total_Misses" in data.columns
    ):

        total_attempts = (
            data["Total_Hits"]
            + data["Total_Misses"]
        )

        data["Error_Rate"] = np.where(
            total_attempts > 0,
            data["Total_Misses"] / total_attempts,
            np.nan
        )

    return data


# ============================================================
# REMOVE INVALID LABEL ROWS
# ============================================================

def remove_invalid_labels(data):

    before = len(data)

    data = data.dropna(
        subset=["Dyslexia"]
    )

    after = len(data)

    print(
        f"Removed rows with invalid labels: "
        f"{before - after}"
    )

    data["Dyslexia"] = data[
        "Dyslexia"
    ].astype(int)

    return data


# ============================================================
# SAVE CLEAN DATA
# ============================================================

def save_clean_data(data):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DIR
        / "SPBE_cleaned_behavioral_dataset.csv"
    )

    data.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved cleaned dataset:\n{output_file}"
    )


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SETS
# ============================================================

def create_splits(data):

    target = data["Dyslexia"]

    features = data.drop(
        columns=["Dyslexia"]
    )

    # Keep only numerical features for ML
    numerical_features = features.select_dtypes(
        include=np.number
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        numerical_features,
        target,
        test_size=0.30,
        random_state=42,
        stratify=target
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # --------------------------------------------------------
    # Scale using training data only
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_validation_scaled = pd.DataFrame(
        X_validation_scaled,
        columns=X_validation.columns,
        index=X_validation.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    # Add target
    X_train_scaled["Dyslexia"] = y_train
    X_validation_scaled["Dyslexia"] = y_validation
    X_test_scaled["Dyslexia"] = y_test

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    X_train_scaled.to_csv(
        PROCESSED_DIR / "SPBE_train.csv",
        index=False
    )

    X_validation_scaled.to_csv(
        PROCESSED_DIR / "SPBE_validation.csv",
        index=False
    )

    X_test_scaled.to_csv(
        PROCESSED_DIR / "SPBE_test.csv",
        index=False
    )

    print("\nDataset splits:")
    print("Training:", len(X_train_scaled))
    print("Validation:", len(X_validation_scaled))
    print("Testing:", len(X_test_scaled))


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("SPBE DATA PREPROCESSING")
    print("=" * 60)

    print("\nLoading Dyt datasets...")

    data = load_dyt_data()

    print(
        f"Desktop + Tablet records: {len(data)}"
    )

    print("\nCleaning missing values...")

    data = clean_missing_values(data)

    print("\nRemoving duplicates...")

    data = remove_duplicates(data)

    print("\nConverting numeric columns...")

    data = convert_numeric_columns(data)

    print("\nCleaning Dyslexia labels...")

    data = clean_dyslexia_label(data)

    print("\nCreating behavioral features...")

    data = create_behavioral_features(data)

    print("\nRemoving invalid labels...")

    data = remove_invalid_labels(data)

    print("\nFinal dataset shape:")

    print(data.shape)

    print("\nDyslexia distribution:")

    print(
        data["Dyslexia"].value_counts()
    )

    print("\nSaving cleaned dataset...")

    save_clean_data(data)

    print("\nCreating ML splits...")

    create_splits(data)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()