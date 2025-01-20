import matplotlib.pyplot as plt
import numpy as np

from get_data import init_column_map, pull, pull_data_rowwise

def plot_metric(question_id, csv_file, xlabel='x label', ylabel='Count of respondents'):
    """
    Example: Read the distribution of Q10_2 (interest in V2G) and plot a bar chart
    """
    init_column_map(csv_file)  
    # Suppose Q10_2 is on a 5-pt Likert: "Strongly disagree" -> 1 ... "Strongly agree" -> 5
    # If your raw CSV codes them differently, define r_map accordingly:
    r_map = {
        'Strongly disagree':1,
        'Somewhat disagree':2,
        'Neither agree nor disagree':3,
        'Somewhat agree':4,
        'Strongly agree':5
    }
    data_dict = pull(question_id, csv_file, r_map=r_map)
    # data_dict might be {1: count, 2: count, ...}  

    # make a bar chart
    keys = sorted(data_dict.keys())
    vals = [data_dict[k] for k in keys]
    plt.bar(keys, vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    plot_metric()


###### Note ######
# The following function is quasi-hard-coded for Q10_2, we can change it to a generic target variable

def plot_demographic_vs_Q10_2(demo_var, possible_demo_answers, possible_q10_2, csvfile='dfc.csv'):
    """
    Creates a side-by-side bar chart of how Q10_2 (rows) vary by a single demographic variable (columns).

    :param demo_var: e.g., 'Q21'
    :param possible_demo_answers: list of known categories for Q21, or None to auto-detect
    :param possible_q10_2: list of known categories for Q10_2 (like 5 Likert levels)
    """
    init_column_map(csvfile)
    data_rows = pull_data_rowwise([demo_var, 'Q10_2'], csvfile)
    crosstab = {}
    for row in data_rows:
        dv_val, q10_val = row
        if dv_val == '' or q10_val == '':
            continue
        key = (dv_val, q10_val)
        crosstab[key] = crosstab.get(key, 0) + 1

    # If not provided, auto-detect possible demo answers
    if possible_demo_answers is None:
        possible_demo_answers = sorted(set([k[0] for k in crosstab.keys()]))

    # Build matrix
    matrix = np.zeros((len(possible_demo_answers), len(possible_q10_2)), dtype=int)
    for (dv_val, ans) in crosstab:
        if dv_val in possible_demo_answers and ans in possible_q10_2:
            r = possible_demo_answers.index(dv_val)
            c = possible_q10_2.index(ans)
            matrix[r, c] = crosstab[(dv_val, ans)]

    #    Each row in matrix is one home type, each column is count for a Q10_2 level
    xvals = np.arange(len(possible_demo_answers))  # one bar group per home type
    bar_width = 0.15  # small width for side-by-side bars of 5 answer levels
    fig, ax = plt.subplots(figsize=(10,6))

    for i, q10_label in enumerate(possible_q10_2):
        # matrix[:, i] is the count for that Q10_2 category across all home types
        offset = (i - 2) * bar_width  # shift bars left/right
        ax.bar(xvals + offset, matrix[:, i], width=bar_width, label=q10_label)

    ax.set_xticks(xvals)
    ax.set_xticklabels(possible_demo_answers, rotation=45, ha='right')
    ax.set_xlabel("Type of Home (Q21)")
    ax.set_ylabel("Number of Respondents")
    ax.set_title("Distribution of Q10_2 by Type of Home (Q21)")
    ax.legend(title="Q10_2 Answers", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


#if __name__ == "__main__":
#    analyze_demo_vs_Q10_2('dfc.csv')
