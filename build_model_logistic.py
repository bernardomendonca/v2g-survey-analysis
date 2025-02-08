
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_score

from get_data import init_column_map, pull_data_rowwise

import itertools

from auto_transformers import auto_build_transformers

from build_model_multinomial import fallback_text_to_float

def build_v2g_model_binary_from_df(df, input_variables, target_variable, transformers, do_normalize=True, test_split_ratio=0.2):
    """
    Builds a binary logistic regression model to predict V2G adoption.
    Converts categorical inputs to numerical using transformers.
    """

    X_list = []
    y_list = []

    # Convert the input variables
    for _, row in df.iterrows():
        row_x = []
        good_row = True  # Track if we should keep this row

        for var_name in input_variables:
            if var_name in df.columns:
                raw_val = row[var_name]
                val = transformers.get(var_name, fallback_text_to_float)(raw_val)
                if val < 0:
                    good_row = False  # Mark as invalid
                row_x.append(val)
            else:
                good_row = False

        # Convert the target variable (V2G Adoption: 0 or 1)
        if target_variable in df.columns:
            raw_target = row[target_variable]
            t_val = transformers.get(target_variable, fallback_text_to_float)(raw_target)
            if t_val < 0:
                good_row = False
        else:
            good_row = False

        if good_row:
            X_list.append(row_x)
            y_list.append(t_val)

    # Convert lists to numpy arrays
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    if X.shape[0] == 0:
        raise ValueError("No valid rows found after transformation.")

    # Normalize input data
    if do_normalize:
        X = normalize(X, axis=0, norm='max')

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=42)

    # Fit a binary logistic regression model
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    # Evaluate performance
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")

    return model, X_train, y_train, X_test, y_test