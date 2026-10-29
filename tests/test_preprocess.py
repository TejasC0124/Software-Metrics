import unittest

import pandas as pd

from src.preprocess import (
    clean_dyslexia_label,
    clean_missing_values,
    convert_numeric_columns,
    create_behavioral_features,
    remove_invalid_labels,
)


def make_raw_frame(rows=2):
    data = {"Dyslexia": ["Yes", "no"][:rows]}
    for i in range(1, 33):
        data[f"Accuracy{i}"] = [0.5] * rows
        data[f"Missrate{i}"] = [0.25] * rows
        data[f"Score{i}"] = [10.0] * rows
        data[f"Clicks{i}"] = [1] * rows
        data[f"Hits{i}"] = [2] * rows
        data[f"Misses{i}"] = [1] * rows
    return pd.DataFrame(data)


class PreprocessTestCase(unittest.TestCase):
    def test_placeholder_strings_become_missing(self):
        frame = pd.DataFrame({"a": ["NULL", "n/a", "ok", " "]})
        cleaned = clean_missing_values(frame)
        self.assertTrue(pd.isna(cleaned["a"].iloc[0]))
        self.assertTrue(pd.isna(cleaned["a"].iloc[1]))
        self.assertTrue(pd.isna(cleaned["a"].iloc[3]))
        self.assertEqual(cleaned["a"].iloc[2], "ok")

    def test_numeric_columns_coerce_bad_values_to_nan(self):
        frame = pd.DataFrame({"Score1": ["3.5", "abc"], "Other": ["x", "y"]})
        converted = convert_numeric_columns(frame)
        self.assertEqual(converted["Score1"].iloc[0], 3.5)
        self.assertTrue(pd.isna(converted["Score1"].iloc[1]))
        self.assertEqual(converted["Other"].iloc[0], "x")

    def test_dyslexia_labels_are_mapped_to_binary(self):
        frame = pd.DataFrame({"Dyslexia": [" Yes ", "N", "maybe"]})
        labelled = clean_dyslexia_label(frame)
        self.assertEqual(labelled["Dyslexia"].iloc[0], 1)
        self.assertEqual(labelled["Dyslexia"].iloc[1], 0)
        self.assertTrue(pd.isna(labelled["Dyslexia"].iloc[2]))

    def test_missing_label_column_raises(self):
        with self.assertRaises(ValueError):
            clean_dyslexia_label(pd.DataFrame({"Other": [1]}))

    def test_behavioral_features_are_computed(self):
        frame = create_behavioral_features(make_raw_frame())
        self.assertAlmostEqual(frame["Mean_Accuracy"].iloc[0], 0.5)
        self.assertAlmostEqual(frame["Total_Clicks"].iloc[0], 32)
        self.assertAlmostEqual(frame["Total_Hits"].iloc[0], 64)
        self.assertAlmostEqual(frame["Total_Misses"].iloc[0], 32)
        self.assertAlmostEqual(frame["Error_Rate"].iloc[0], 1 / 3)
        self.assertAlmostEqual(frame["Accuracy_Improvement"].iloc[0], 0.0)

    def test_rows_without_valid_label_are_removed(self):
        frame = clean_dyslexia_label(pd.DataFrame({"Dyslexia": ["yes", "unknown"]}))
        kept = remove_invalid_labels(frame)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept["Dyslexia"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
