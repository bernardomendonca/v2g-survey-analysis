import pandas as pd

def filter_data(df, conditions):
    """
    Filters a dataframe based on given conditions.
    
    conditions: list of strings, e.g. ["Q1_2 == '1'", "(Q3_1 == '1') | (Q3_2 == '1')"]
    """
    query_str = " & ".join(conditions)
    return df.query(query_str)


### Concatenate Segments with Labels
def prepare_segmented_data(segments, segment_names):
    """
    Concatenates multiple segmented DataFrames and adds a 'Segment' column.

    Args:
        segments (list of pd.DataFrame): List of DataFrames (each a segment).
        segment_names (list of str): List of names corresponding to each segment.

    Returns:
        pd.DataFrame: Concatenated DataFrame with 'Segment' column.
    """
    for df, name in zip(segments, segment_names):
        df["Segment"] = name  # Add segment label

    return pd.concat(segments, ignore_index=True)
