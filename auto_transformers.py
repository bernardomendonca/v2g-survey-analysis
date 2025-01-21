# auto_transformers.py

import csv
import os
from get_data import init_column_map, column_map, pull_data_rowwise

def can_parse_as_float(s):
    try:
        float(s)
        return True
    except:
        return False

def auto_build_transformers(csvfile, exclude_cols=None, max_categories=50):
    """
    Goes through each column in the CSV (except excludes),
    inspects the unique answers, and builds a dictionary-based
    or numeric-based transformer.

    - If all unique answers can parse as float, we mark it as numeric => fallback.
    - If unique answers > max_categories, we skip (or treat as invalid).
    - Else we map each unique answer to an integer, e.g. { 'METROPOLITAN':1, 'REGIONAL':2, ... }.

    Returns: a dictionary like:
      { column_name: { 'someAnswer': code, ... } }
    or if numeric, we store { '__NUMERIC__': True } to signal "just parse as float".
    """
    if exclude_cols is None:
        exclude_cols = []

    # 1) Make sure column_map is initialized
    init_column_map(csvfile)

    # 2) We'll gather unique answers for each column
    #    but skip excludes
    transformers = {}

    for col_name, col_idx in column_map.items():
        if col_name in exclude_cols:
            continue

        # Gather unique answers
        unique_answers = set()
        # We can do a quick pass to see them
        with open(csvfile, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # skip
            for row in reader:
                val = row[col_idx]
                unique_answers.add(val.strip())

        # Remove blank or empty strings if you want
        if '' in unique_answers:
            unique_answers.remove('')

        # If no unique answers remain, skip
        if len(unique_answers) == 0:
            # e.g. entire column is blank
            continue

        # Check if all parse as float
        all_numeric = True
        for ans in unique_answers:
            if not can_parse_as_float(ans):
                all_numeric = False
                break

        if all_numeric:
            # We'll store a special marker to parse as numeric
            transformers[col_name] = {"__NUMERIC__": True}
            print(f"[INFO] Column {col_name} -> numeric fallback")
        else:
            # This is a categorical text col
            # If it has too many unique categories, we might skip it
            if len(unique_answers) > max_categories:
                print(f"[WARN] Column {col_name} has {len(unique_answers)} categories; skipping.")
                continue
            # Build a map (answer -> integer code)
            sorted_vals = sorted(list(unique_answers))
            code_map = {}
            code = 1
            for val in sorted_vals:
                code_map[val] = code
                code += 1
            transformers[col_name] = code_map
            print(f"[INFO] Column {col_name} -> categorical map: {code_map}")

    return transformers
