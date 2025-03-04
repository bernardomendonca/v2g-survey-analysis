


def filter_data(df, conditions):
    """
    Filters a dataframe based on given conditions.
    
    conditions: list of strings, e.g. ["Q1_2 == '1'", "(Q3_1 == '1') | (Q3_2 == '1')"]
    """
    query_str = " & ".join(conditions)
    return df.query(query_str)
