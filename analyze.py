import matplotlib.pyplot as plt

from get_data import init_column_map, pull

def plot_metric(question_id, csv_file, xlabel='x label', ylabel='Count of respondents'):
    """
    Example: Read the distribution of Q10_2 (interest in V2G) and plot a bar chart
    """
    init_column_map(csv_file)  
    # Suppose Q10_2 is on a 5-pt Likert: "Strongly disagree" -> 1 ... "Strongly agree" -> 5
    # If your raw CSV codes them differently, define r_map accordingly:
    r_map = {
        'Strongly disagree':1,
        'Disagree':2,
        'Neither agree nor disagree':3,
        'Agree':4,
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