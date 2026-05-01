"""
Utility functions for reading data, saving JSON, and validating submissions.

The validation step protects the final JSON from simple mistakes like missing
review IDs, invalid aspect names, invalid sentiment labels, or mismatched keys.
"""

import json
import pandas as pd

from config import absa_config
from absa_config import ALLOWED_ASPECTS, ALLOWED_SENTIMENTS


def load_input_table(path):
    """
    Load an Excel or CSV file into a pandas DataFrame.
    """
    path = str(path)

    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)

    if path.endswith(".csv"):
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path}")


def save_json_file(records, output_path):
    """
    Save prediction records as UTF-8 JSON.

    ensure_ascii=False keeps Arabic text readable if any Arabic values appear.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def validate_submission_records(records, expected_review_ids):
    """
    Validate the final prediction list before saving.

    The challenge requires:
    - one object per review
    - no missing review_id
    - review_id must be int
    - aspects must be a list
    - aspect_sentiments must be a dict
    - aspect_sentiments keys must exactly match aspects
    """
    errors = []

    if not isinstance(records, list):
        return ["Submission must be a list."]

    seen_ids = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"Record {index} is not a dictionary.")
            continue

        required_keys = {"review_id", "aspects", "aspect_sentiments"}
        missing_keys = required_keys - set(record.keys())

        if missing_keys:
            errors.append(f"Record {index} missing keys: {missing_keys}")
            continue

        review_id = record["review_id"]
        aspects = record["aspects"]
        aspect_sentiments = record["aspect_sentiments"]

        if not isinstance(review_id, int):
            errors.append(f"Record {index} review_id is not int.")

        seen_ids.append(review_id)

        if not isinstance(aspects, list):
            errors.append(f"Record {index} aspects is not a list.")
            continue

        if len(aspects) == 0:
            errors.append(f"Record {index} has an empty aspects list.")

        for aspect in aspects:
            if aspect not in ALLOWED_ASPECTS:
                errors.append(f"Record {index} has invalid aspect: {aspect}")

        if not isinstance(aspect_sentiments, dict):
            errors.append(f"Record {index} aspect_sentiments is not a dictionary.")
            continue

        if set(aspects) != set(aspect_sentiments.keys()):
            errors.append(
                f"Record {index} has mismatch between aspects and aspect_sentiments keys."
            )

        for aspect, sentiment in aspect_sentiments.items():
            if aspect not in ALLOWED_ASPECTS:
                errors.append(f"Record {index} has invalid aspect key: {aspect}")

            if sentiment not in ALLOWED_SENTIMENTS:
                errors.append(f"Record {index} has invalid sentiment: {sentiment}")

    expected_ids = set(map(int, expected_review_ids))
    seen_ids_set = set(seen_ids)

    missing_ids = expected_ids - seen_ids_set
    extra_ids = seen_ids_set - expected_ids

    if missing_ids:
        errors.append(f"Missing review_ids count: {len(missing_ids)}")

    if extra_ids:
        errors.append(f"Extra review_ids count: {len(extra_ids)}")

    if len(seen_ids) != len(seen_ids_set):
        errors.append("Duplicate review_ids found.")

    return errors