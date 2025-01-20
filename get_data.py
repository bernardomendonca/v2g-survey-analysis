import csv 

# A global dictionary to map column names to indices
column_map = {}

def init_column_map(csvfile):
    """
    Reads the header of the CSV and builds a dictionary
    from column-name -> column-index.
    """
    global column_map
    with open(csvfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for i, h in enumerate(headers):
            column_map[h] = i


def pull(question_id, csvfile, r_map=None):
    """
    Returns a dictionary {answer: count} for the specified column (question_id).
    If r_map is given, it will map each raw string to a transformed numeric value.
    """
    global column_map
    if question_id not in column_map:
        print(f"Question {question_id} not in column_map. Check spelling!")
        return {}

    counts = {}
    col_idx = column_map[question_id]

    with open(csvfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            raw_ans = row[col_idx]
            if raw_ans == '':
                continue

            # If a mapping is provided, convert
            if r_map is not None:
                if raw_ans not in r_map:
                    # skip or handle unknown answers
                    continue
                val = r_map[raw_ans]
            else:
                val = raw_ans

            counts[val] = counts.get(val, 0) + 1

    return counts

def pull_data_rowwise(question_ids, csvfile):
    """
    Returns a list of lists or list of dicts, each row containing 
    the values for the specified question_ids. 
    For example, question_ids might be ['gender','Q10_2','Q9','Q1_1'].
    """
    global column_map
    rows = []

    # Indices for each question_id
    indices = []
    for q in question_ids:
        if q not in column_map:
            raise ValueError(f"Question ID {q} not found in column_map.")
        indices.append(column_map[q])

    with open(csvfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            # Extract the relevant columns for this row
            selected = [line[i] for i in indices]
            rows.append(selected)

    return rows