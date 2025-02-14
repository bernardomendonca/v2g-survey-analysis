import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_score

from get_data import init_column_map, pull_data_rowwise

import itertools

from auto_transformers import auto_build_transformers

### More info on sklearn logistic regression here:
# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression


##################
## TEXT TO CODE ##
##################

# For now, whilst we explore the idiosyncracies of each question, I'm setting functions to convert the strings to code
# Later on, we can generalise these based on the types of questions.
# TBD - Move this to other file

def text_to_code_q1_2_multi(raw_ans):
    """
    Q8_1, Q8_2, Q8_99 are individual columns with '1' or '0' in the CSV.
    We just convert '1' -> 1 (True) and '0' -> 0 (False).
    """
    if raw_ans == '1':
        return 1
    else:
        return 0

def text_to_code_q2(raw_ans):
    """
    Example mapping for Q2, which has 5 options:
      - 'Less than 10,000'
      - '10,000-20,000'
      - '20,001-50,000'
      - 'More than 50,000'
      - 'Unsure'
    We can turn these into an ordinal scale, or treat 'Unsure' specially.
    """
    mapping = {
        'Less than 10,000': 1,
        '10,000-20,000': 2,
        '20,001-50,000': 3,
        'More than 50,000': 4,
        'Unsure': 0  # or maybe -1 if we want to exclude, so it becomes invalide/unrecognised
    }
    return mapping.get(raw_ans, -1)

def text_to_code_q3_parking(raw_ans):
    """
    Converts Q3 parking questions (Q3_1, Q3_2, ..., Q3_5) to numeric values.
    - "1" → 1 (Yes, has this parking type)
    - "0" → 0 (No, does not have this parking type)
    """
    return 1 if raw_ans == '1' else 0  # Ensure binary conversion

def text_to_code_q7_likert(raw_ans):
    """
    Converts Q7 responses (Likert scale) to numerical values:
    - 'Strongly disagree' → 1
    - 'Somewhat disagree' → 2
    - 'Neither agree nor disagree' → 3
    - 'Somewhat agree' → 4
    - 'Strongly agree' → 5
    """
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5
    }
    return mapping.get(raw_ans, -1)  # Default to -1 for invalid values

def text_to_code_q8_multi(raw_ans):
    """
    Q8_1, Q8_2, Q8_99 are individual columns with '1' or '0' in the CSV.
    We just convert '1' -> 1 (True) and '0' -> 0 (False).
    """
    if raw_ans == '1':
        return 1
    else:
        return 0

def text_to_code_q9(raw_ans):
    """
    Example mapping for Q9, which has 3 options:
      - 'Very familiar'
      - 'Somewhat familiar'
      - 'Not at all familiar'
    We convert them to 3,2,1 respectively. 
    """
    mapping = {
        'Very familiar': 3,
        'Somewhat familiar': 2,
        'Not at all familiar': 1
    }
    return mapping.get(raw_ans, -1)  # -1 means "invalid/unrecognised"

def text_to_code_q10_2(raw_ans):
    mapping = {
       'Strongly disagree':1,
       'Somewhat disagree':2,
       'Neither agree nor disagree':3,
       'Somewhat agree':4,
       'Strongly agree':5
    }
    return mapping.get(raw_ans, -1)

def text_to_code_q1_2_multi(raw_ans):
    """
    Q8_1, Q8_2, Q8_99 are individual columns with '1' or '0' in the CSV.
    We just convert '1' -> 1 (True) and '0' -> 0 (False).
    """
    if raw_ans == '1':
        return 1
    else:
        return 0
    
def text_to_code_q6(raw_ans):
    """
    Converts Q6 responses (charging behavior) to numeric values.
    - "0 - Weekdays" or "" → 0
    - Valid responses (1-5) → Keep as is
    """
    if raw_ans in ["0 - Weekdays", ""]:
        return 0
    try:
        return int(raw_ans)
    except:
        return -1  # Invalid entry (unlikely but safe)


def text_to_code_q10_2_binary(raw_ans):
    """
    Converts Q10_2 from 5-category responses to binary:
    - "Strongly disagree", "Somewhat disagree", "Neither agree nor disagree" → 0 (Not adopting)
    - "Somewhat agree", "Strongly agree" → 1 (Adopting)
    """
    adopting = {"Somewhat agree", "Strongly agree"}
    return 1 if raw_ans in adopting else 0


def text_to_code_q14_benefits_v2g(raw_ans):
    """
    - "1" → 1 (Yes)
    - "0" → 0 (No)
    """
    return 1 if raw_ans == '1' else 0  # Ensure binary conversion

def text_to_code_q15_concerns_v2g(raw_ans):
    """
    - "1" → 1 (Yes)
    - "0" → 0 (No)
    """
    return 1 if raw_ans == '1' else 0  # Ensure binary conversion

#####################################
## TEXT TO CODE - GENERIC FUNCTION ##
#####################################

# Here I'm slowly setting up the generic functions to handle text-to-code

def text_to_code_binary(raw_ans):
    
    """
    Converts a binary response to 0 or 1.
    - If the value is 1 (integer), "1" (string), or True (boolean), return 1.
    - Otherwise, return 0.
    """
    if str(raw_ans).strip() in {"1", "True"}:  # Handles strings & booleans
        return 1
    elif str(raw_ans).strip() in {"0", "False"}:  # Handles strings & booleans
        return 0
    else:
        return -1  # Invalid/unrecognized value


#################################
## UNIVERSAL FALLBACK FUNCTION ##
#################################

def fallback_text_to_float(raw_ans):
    """
    Generic fallback. If raw_ans can be parsed as float, return it.
    Otherwise, return -1 (invalid).
    """
    try:
        return float(raw_ans)
    except:
        # Could also check if it's '' or 'NA' or something
        return -1

########################
# Master Transformers Dict
########################
TRANSFORMERS = {
    # Q1 can use text_to_code_binary:
    'Q1_1': text_to_code_binary, #Petrol/Diesel car
    'Q1_2': text_to_code_binary, # Electric Vehicle
    'Q1_3': text_to_code_binary, # Plug-in Hybrid vehicle
    'Q1_4': text_to_code_binary, # Hybrid vehicle
    'Q1_99': text_to_code_binary, # I don't own a car

    'Q2': text_to_code_q2,

    'Q3_1': text_to_code_q3_parking,  # Personal driveway
    'Q3_2': text_to_code_q3_parking,  # Personal garage
    'Q3_3': text_to_code_q3_parking,  # Carport
    'Q3_4': text_to_code_q3_parking,  # Street parking
    'Q3_5': text_to_code_q3_parking,   # Other parking

    'Q7_1': text_to_code_q7_likert, # Benefits V2G
    'Q7_2': text_to_code_q7_likert, # Benefits V2G
    'Q7_3': text_to_code_q7_likert, # Benefits V2G


    'Q8_1': text_to_code_q8_multi, # Solar Panels
    'Q8_2': text_to_code_q8_multi, # Home Battery
    'Q8_99': text_to_code_q8_multi, # Neither

    'Q9': text_to_code_q9,

    'Q10_2': text_to_code_q10_2,

    'Q14_1': text_to_code_binary,
    'Q14_2': text_to_code_binary,
    'Q14_3': text_to_code_binary,
    'Q14_4': text_to_code_binary,
    'Q14_5': text_to_code_binary,
    'Q14_6': text_to_code_binary,
    'Q14_7': text_to_code_binary,
    'Q14_8': text_to_code_binary,
    'Q14_99': text_to_code_binary,

    'Q15_1': text_to_code_binary,
    'Q15_2': text_to_code_binary,
    'Q15_3': text_to_code_binary,
    'Q15_4': text_to_code_binary,
    'Q15_5': text_to_code_binary,
    'Q15_6': text_to_code_binary,
    'Q15_7': text_to_code_binary,
    'Q15_8': text_to_code_binary,
    'Q15_9': text_to_code_binary,
    'Q15_10': text_to_code_binary,
    'Q15_99': text_to_code_binary,



}


####################
## MODEL BUILDING ##
####################

def build_v2g_model(csvfile, input_variables, target_variable):
    """
    Builds a logistic regression model given:
      - csvfile: path to your CSV dataset
      - input_variables: list of column names to use as predictors
      - target_variable: single column name representing the outcome
    """

    # Initialize the column map
    init_column_map(csvfile)

    # Gather the data row-wise
    question_list = input_variables + [target_variable]
    data = pull_data_rowwise(question_list, csvfile)

    # Set up a transformer dictionary to apply the correct numeric mapping for each variable. 
    # TBD - Improve on this
    transformers = {
        'Q9': text_to_code_q9,
        'Q2': text_to_code_q2,
        'Q8_1': text_to_code_q8_multi,
        'Q8_2': text_to_code_q8_multi,
        'Q8_99': text_to_code_q8_multi,
        'Q10_2': text_to_code_q10_2
        # We can keep increasing this list as we want
    }

    X = []
    y = []

    # Indices: first len(input_variables) columns are inputs, the last one is the target, analogous to 'data' up there
    n_inputs = len(input_variables)

    for row in data:
        row_x = []
        good_row = True  # we'll set False if any value is missing or invalid

        # Convert input variables
        for i, var_name in enumerate(input_variables):
            raw_val = row[i]
            if var_name in transformers:
                val = transformers[var_name](raw_val)
            else:
                # Default attempt: parse as float or int
                try:
                    val = float(raw_val)
                except:
                    val = -1 
            if val < 0:
                good_row = False  # indicates invalid or missing
            row_x.append(val)

        # Convert target variable
        raw_target = row[n_inputs]
        if target_variable in transformers:
            t_val = transformers[target_variable](raw_target)
        else:
            # Default parse as numeric
            try:
                t_val = float(raw_target)
            except:
                t_val = -1

        if t_val < 0:
            good_row = False

        # If everything is valid, append. Check if good_row is True
        if good_row:
            X.append(row_x)
            y.append(t_val)

    # Turn into numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # (Optional) Normalize or standardize the predictor matrix
    ## See here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    # TBD - Add this as a parameter to the function
    X = normalize(X, axis=0, norm='max')

    # Fit logistic regression with L1 penalty
    ## We can also consider L2 (Ridge regression)
    ## Here's a great article on it: https://medium.com/analytics-vidhya/regularization-understanding-l1-and-l2-regularization-for-deep-learning-a7b9e4a409bf
    model = LogisticRegression(
        penalty='l1', 
        # Note that liblinear works well for now, but SAGA might be able to handle  multinomial + L1 better.    
        solver='liblinear'
        )
    model.fit(X, y)

    # Evaluate
    accuracy = model.score(X, y)
    print("Accuracy on training data: ", accuracy)
    print("Coefficients shape:", model.coef_.shape)
    print("Coefficients:", model.coef_)
    print("Intercept(s):", model.intercept_)

    return model, X, y



def build_v2g_model_multinomial(csvfile, input_variables, target_variable,
                                do_normalize=True, test_split_ratio=0.0):
    """
    Builds a multinomial logistic regression with L1 penalty.
    - if a column is not in TRANSFORMERS, fallback_text_to_float is used
    - rows with any invalid value (=-1) are discarded
    - if no valid rows remain, we raise an Exception

    Returns (model, X, y, X_test, y_test)
    """
    init_column_map(csvfile)
    question_list = input_variables + [target_variable]
    data = pull_data_rowwise(question_list, csvfile)

    X_list = []
    y_list = []

    n_inputs = len(input_variables)

    for row_vals in data:
        row_x = []
        good_row = True

        # Convert each predictor
        for i, var_name in enumerate(input_variables):
            raw_val = row_vals[i]
            if var_name in TRANSFORMERS:
                val = TRANSFORMERS[var_name](raw_val)
            else:
                val = fallback_text_to_float(raw_val)
            if val < 0:
                good_row = False
            row_x.append(val)

        # Convert the target
        raw_target = row_vals[n_inputs]
        if target_variable in TRANSFORMERS:
            t_val = TRANSFORMERS[target_variable](raw_target)
        else:
            t_val = fallback_text_to_float(raw_target)

        if t_val < 0:
            good_row = False

        if good_row:
            X_list.append(row_x)
            y_list.append(t_val)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    # If no valid rows => can't fit
    if X.shape[0] == 0:
        raise ValueError("No valid rows found after transformation - all rows invalid or empty subset")

    # If only one dimension for X => we need to reshape
    # (But if at least 2 rows, shape will be (n,1) or (n,>1). scikit-learn needs (n_samples, n_features).
    if len(X.shape) == 1:
        # Means X is shape (n,)
        X = X.reshape(-1, 1)

    # Optionally normalize
    if do_normalize:
        X = normalize(X, axis=0, norm='max')

    # Split if needed
    X_test, y_test = None, None
    if test_split_ratio > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=42)
    else:
        X_train, y_train = X, y

    # Build a MULTINOMIAL logistic regression with L1
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        penalty='l1',
        solver='saga',          # can handle L1 + multi_class
        multi_class='multinomial',
        max_iter=1000
    )

    model.fit(X_train, y_train)

    # Evaluate on training
    train_acc = model.score(X_train, y_train)
    print("Train accuracy:", train_acc)

    if X_test is not None:
        test_acc = model.score(X_test, y_test)
        print("Test accuracy:", test_acc)

    return model, X, y, X_test, y_test

def build_v2g_model_multinomial_with_transformers(
    csvfile,
    input_variables,
    target_variable,
    transformers,
    do_normalize=True,
    test_split_ratio=0.0
):
    """
    Build a multinomial logistic regression with L1 penalty,
    using the `transformers` dictionary from auto_build_transformers.

    Steps:
      - For each column in input_variables + [target_variable], we look up 
        its mapping in `transformers`.
      - If it has {"__NUMERIC__": True}, parse as float.
      - Otherwise, we map string->int using the dict.
      - If any value is not recognized => row is invalid => skip.
      - Fit the logistic regression, return (model, X, y, X_test, y_test).
    """
    init_column_map(csvfile)
    question_list = input_variables + [target_variable]
    data = pull_data_rowwise(question_list, csvfile)

    X_list = []
    y_list = []

    n_inputs = len(input_variables)

    for row_vals in data:
        row_x = []
        good_row = True

        # 1) Convert input features
        for i, col_name in enumerate(input_variables):
            raw_val = row_vals[i].strip()
            if col_name not in transformers:
                # if no mapping => skip or parse as float?
                # let's skip
                good_row = False
                break

            mapping_or_num = transformers[col_name]
            if "__NUMERIC__" in mapping_or_num:
                # parse as float
                try:
                    val_f = float(raw_val)
                except:
                    val_f = -1
                if val_f < 0:
                    good_row = False
                row_x.append(val_f)
            else:
                # categorical
                if raw_val in mapping_or_num:
                    coded = mapping_or_num[raw_val]
                    row_x.append(coded)
                else:
                    # unknown text
                    good_row = False

        # 2) Convert target
        raw_target = row_vals[n_inputs].strip()
        t_val = None
        if target_variable not in transformers:
            # if no transformer for target => skip
            good_row = False
        else:
            map_t = transformers[target_variable]
            if "__NUMERIC__" in map_t:
                # parse float
                try:
                    t_val_f = float(raw_target)
                except:
                    t_val_f = -1
                if t_val_f < 0:
                    good_row = False
                t_val = t_val_f
            else:
                # categorical
                if raw_target in map_t:
                    t_val = map_t[raw_target]
                else:
                    good_row = False

        if good_row and (t_val is not None):
            X_list.append(row_x)
            y_list.append(t_val)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    # Edge case: if no valid rows or only 1 row => can't train
    if X.shape[0] <= 1:
        raise ValueError("Not enough valid rows to train the model.")

    # If X is 1D shape => reshape
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if do_normalize:
        X = normalize(X, axis=0, norm='max')

    X_test, y_test = None, None
    if test_split_ratio > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=42)
    else:
        X_train, y_train = X, y

    model = LogisticRegression(
        penalty='l1',
        solver='saga',          # supports L1 + multi-class
        multi_class='multinomial',
        max_iter=1000
    )
    model.fit(X_train, y_train)

    # Evaluate
    acc_train = model.score(X_train, y_train)
    print("Train accuracy:", acc_train)
    if X_test is not None:
        acc_test = model.score(X_test, y_test)
        print("Test accuracy:", acc_test)

    return model, X, y, X_test, y_test


def evaluate_subset(csvfile, subset_vars, target):
    """
    Build a logistic regression on subset_vars -> target,
    do cross-validation, return mean accuracy.
    """
    # We'll build the model with no train/test split 
    model, X, y, _, _ = build_v2g_model_multinomial(
        csvfile=csvfile,
        input_variables=list(subset_vars),
        target_variable=target,
        do_normalize=True,
        test_split_ratio=0.0
    )
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    return scores.mean()


def evaluate_subset_with_transformers(csvfile, subset_vars, target):
    """
    Build a logistic regression on subset_vars -> target,
    do cross-validation, return mean accuracy.
    """
    # We'll build the model with no train/test split 
    model, X, y, _, _ = build_v2g_model_multinomial_with_transformers(
        csvfile=csvfile,
        input_variables=list(subset_vars),
        target_variable=target,
        transformers=transformers,  
        do_normalize=True,
        test_split_ratio=0.0
    )
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    return scores.mean()




def build_v2g_model_multinomial_from_df(df, input_variables, target_variable, transformers, do_normalize=True, test_split_ratio=0.2):
    """
    Builds a multinomial logistic regression model from a DataFrame.
    Instead of reading from CSV, this function takes a pre-processed DataFrame.
    """
    X_list = []
    y_list = []

    n_inputs = len(input_variables)

    for _, row in df.iterrows():
        row_x = []
        good_row = True

        # Convert input variables
        for var_name in input_variables:
            if var_name in df.columns:
                raw_val = row[var_name]
                # Check if there's a specific transformer for this variable
                val = transformers.get(var_name, fallback_text_to_float)(raw_val)
            else:
                val = -1  # Treat missing values as invalid
            
            if val < 0:
                good_row = False
            row_x.append(val)

        # Convert target variable
        raw_target = row[target_variable] if target_variable in df.columns else -1
        t_val = transformers.get(target_variable, fallback_text_to_float)(raw_target)

        if t_val < 0:
            good_row = False

        if good_row:
            X_list.append(row_x)
            y_list.append(t_val)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    if X.shape[0] == 0:
        raise ValueError("No valid rows found after transformation.")

    # Normalize if required
    if do_normalize:
        X = normalize(X, axis=0, norm='max')

    # Split if needed
    X_test, y_test = None, None
    if test_split_ratio > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=42)
    else:
        X_train, y_train = X, y

    # Train model
    model = LogisticRegression(
        penalty='l1',
        solver='saga',  # supports L1 + multi-class
        multi_class='multinomial',
        max_iter=1000
    )
    model.fit(X_train, y_train)

    print("Train Accuracy:", model.score(X_train, y_train))
    if X_test is not None:
        print("Test Accuracy:", model.score(X_test, y_test))

    return model, X, y, X_test, y_test
