
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_score

from get_data import init_column_map, pull_data_rowwise
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
            
            # Check if the target is already binary (contains only 0 and 1)
            if raw_target in {0, 1}:  
                t_val = raw_target  # Use directly
            else:
                t_val = transformers.get(target_variable, fallback_text_to_float)(raw_target)  # Apply transformation
            
            # Ensure t_val is valid
            if t_val not in {0, 1}:  
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

    print("Unique values in y_train:", np.unique(y_train, return_counts=True))
    print("Unique values in y_test:", np.unique(y_test, return_counts=True))


    # Fit a binary logistic regression model
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    # Evaluate performance
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")

    return model, X_train, y_train, X_test, y_test


def run_single_feature_regressions(
    df, 
    features_to_select,
    target_variable,
    q6a_petrol,
    q6a_ev,
    q6a_plughyb,
    q6a_hybrid,
    q6a_filters,
    TRANSFORMERS_q6a,
    feature_label_map=None
):
    """
    For each feature in features_to_select, run a single-feature logistic regression.
    
    Steps:
      1) Filter the df if the feature is in q6a_petrol / q6a_ev / q6a_plughyb / q6a_hybrid.
      2) Skip if fewer than 10 rows remain in the filter.
      3) Apply a transformer if available, else convert to numeric.
      4) Drop invalid rows (where value == -1 or NaN).
      5) Fit logistic regression (L1 penalty) and evaluate.
      6) Use statsmodels to get a p-value for the feature.
      7) Return a sorted DataFrame of results (coefficient, accuracy, p-value, etc.).

    :param df: The pandas DataFrame containing all columns.
    :param features_to_select: A list of feature (column) names.
    :param target_variable: The string name of your target (e.g. "Q10_2").
    :param q6a_petrol: List of Q6a columns for petrol/diesel (e.g. ["Q6ax1_1", ...]).
    :param q6a_ev:     List of Q6a columns for EV.
    :param q6a_plughyb: List of Q6a columns for plug-in hybrid.
    :param q6a_hybrid: List of Q6a columns for hybrid.
    :param q6a_filters: Dictionary mapping "petrol"/"ev"/"plug_hybrid"/"hybrid" -> boolean mask.
    :param TRANSFORMERS_q6a: A dict of {col_name: transformer_func} for columns.
    :param feature_label_map: Optional dict to map feature names to nicer labels in results.
    :return: A pandas DataFrame with the results, sorted by absolute coefficient.
    """
    if feature_label_map is None:
        feature_label_map = {}  # fallback

    results = []  # list of dicts for each feature

    for feature in features_to_select:
        print(f"Running Logistic Regression for: {feature}")

        # 1) Check if feature is in df
        if feature not in df.columns:
            print(f"Skipping {feature} (not in dataframe)")
            continue

        # 2) Filter by vehicle type if feature is in a specific Q6a category
        if feature in q6a_petrol:
            filtered_df = df[q6a_filters["petrol"]]
        elif feature in q6a_ev:
            filtered_df = df[q6a_filters["ev"]]
        elif feature in q6a_plughyb:
            filtered_df = df[q6a_filters["plug_hybrid"]]
        elif feature in q6a_hybrid:
            filtered_df = df[q6a_filters["hybrid"]]
        else:
            filtered_df = df  # default: no special filter

        # 3) Skip if too few total rows
        if filtered_df.shape[0] < 10:
            print(f"Skipping {feature} (too few valid rows: {filtered_df.shape[0]})")
            continue

        # 4) Transform or convert numeric
        if feature in TRANSFORMERS_q6a:
            X_single = filtered_df[feature].apply(TRANSFORMERS_q6a[feature])
            # drop rows where X_single == -1 or is NaN
            valid_rows = (X_single != -1) & ~X_single.isna()
        else:
            X_single = pd.to_numeric(filtered_df[feature], errors="coerce")
            valid_rows = ~X_single.isna()

        print(f"Total rows before filter: {filtered_df.shape[0]}")
        print(f"Valid rows after filtering: {valid_rows.sum()}")

        X_single = X_single[valid_rows].values.reshape(-1, 1)
        y = filtered_df.loc[valid_rows, target_variable].values

        # 5) If only one unique value, skip
        if len(np.unique(X_single)) == 1:
            val = np.unique(X_single)[0]
            print(f"Skipping {feature} (constant value: {val})")
            continue

        # 6) Fit logistic regression
        model = LogisticRegression(penalty='l1', solver='liblinear')
        model.fit(X_single, y)

        # Evaluate
        y_pred = model.predict(X_single)
        accuracy = accuracy_score(y, y_pred)

        coef = model.coef_[0][0]
        odds_ratio = np.exp(coef)

        # 7) Statsmodels for p-value
        X_with_intercept = sm.add_constant(X_single)
        sm_model = sm.Logit(y, X_with_intercept).fit(disp=0)
        p_value = sm_model.pvalues[1]

        # store result
        results.append({
            "Feature": feature_label_map.get(feature, feature),
            "Accuracy": accuracy,
            "Coefficient": coef,
            "Odds Ratio": odds_ratio,
            "P-value": p_value
        })

    # 8) Build results DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df["abs_coef"] = results_df["Coefficient"].abs()
        results_df = results_df.sort_values(by="abs_coef", ascending=False)
    else:
        print("No valid results. Possibly all features were skipped.")

    return results_df


def filter_significant_features(
    results_df,
    feature_label_map,
    columns_of_interest,
    target_variable,
    p_threshold=0.05
):
    """
    Filters out columns that are not significant (p >= p_threshold),
    returning a new list of columns_of_interest that only includes
    columns with p < p_threshold, plus the target_variable.

    :param results_df: DataFrame with columns: "Feature", "P-value"
    :param feature_label_map: dict mapping { original_col_name: "Pretty Label" } 
                              or { "Pretty Label": original_col_name }, 
                              depending on your setup
    :param columns_of_interest: the original list of all columns you might consider
    :param target_variable: the name of your target column (always retained)
    :param p_threshold: significance threshold (default 0.05)
    :return: a filtered list of columns (including target_variable).
    """

    # 1) Pull a list of feature names that are significant
    significant_features = results_df[results_df["P-value"] < p_threshold]["Feature"].tolist()

    # 2) Convert those "pretty" feature names back to original columns
    #    or the reverse, depending on how feature_label_map is structured.
    significant_columns = []
    # Example assumption: feature_label_map = { original_col: "Pretty Name" }
    # so we invert that map if needed
    inv_map = {v: k for k,v in feature_label_map.items()}
    for feat in significant_features:
        if feat in inv_map:
            significant_columns.append(inv_map[feat])
        else:
            # if already the original col name, or no map entry
            significant_columns.append(feat)

    # 3) Always keep the target variable
    if target_variable not in significant_columns:
        significant_columns.append(target_variable)

    # 4) Filter the original columns_of_interest list to only these columns
    columns_of_interest_filtered = [
        col for col in columns_of_interest if col in significant_columns
    ]

    # Debug / display info
    print(f"Original columns count: {len(columns_of_interest)}")
    print(f"Filtered columns count: {len(columns_of_interest_filtered)}")

    return columns_of_interest_filtered


