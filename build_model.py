import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from get_data import init_column_map, pull_data_rowwise

### More info on sklearn logistic regression here:
# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression



##################
## TEXT TO CODE ##
##################

# For now, whilst we explore the idiosyncracies of each question, I'm setting functions to convert the strings to code
# Later on, we can generalise these based on the types of questions.
# TBD - Move this to other file

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

def text_to_code_q8_multi(raw_ans):
    """
    Q8_1, Q8_2, Q8_99 are individual columns with '1' or '0' in the CSV.
    We just convert '1' -> 1 (True) and '0' -> 0 (False).
    """
    if raw_ans == '1':
        return 1
    else:
        return 0

def text_to_code_q10_2(raw_ans):
    mapping = {
       'Strongly disagree':1,
       'Somewhat disagree':2,
       'Neither agree nor disagree':3,
       'Somewhat agree':4,
       'Strongly agree':5
    }
    return mapping.get(raw_ans, -1)


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

    return model


def build_v2g_model_multinomial(csvfile, input_variables, target_variable, normalize_cols=True):
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
    if normalize_cols:
        X = normalize(X, axis=0, norm='max')

    # Fit logistic regression with L1 penalty
    ## We can also consider L2 (Ridge regression)
    ## Here's a great article on it: https://medium.com/analytics-vidhya/regularization-understanding-l1-and-l2-regularization-for-deep-learning-a7b9e4a409bf
    model = LogisticRegression(
        penalty='l1', 
        # Note that liblinear works well for now, but SAGA might be able to handle  multinomial + L1 better.    
        solver='saga',
        multi_class='multinomial',
        max_iter=1000,
        C=1.0
        )
    model.fit(X, y)

    # Evaluate
    accuracy = model.score(X, y)
    print("Accuracy on training data: ", accuracy)
    print("Coefficients shape:", model.coef_.shape)
    print("Coefficients:", model.coef_)
    print("Intercept(s):", model.intercept_)

    return model