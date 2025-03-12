import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap


#######################
### Contextual Data ###
#######################


def get_column_label(col_name, question_map):
    """
    Returns the 'pretty' text label for the column name, if it exists in the question_map.
    Otherwise, returns the original col_name.
    """
    if question_map is not None and col_name in question_map:
        return question_map[col_name]
    return col_name


def plot_bars(df, column, segment_name=None, figsize=(6, 4), rotation=30, palette="coolwarm", question_map=None):
    # Get a nicer label if available
    x_label = get_column_label(column, question_map)

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")  
    sns.countplot(x=column, data=df, palette=palette)

    plt.title(f"{x_label} \n segmented by {segment_name}", fontsize=14, fontweight="bold")
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=rotation)
    plt.show()


def plot_stacked_bars(df, col1, col2, segment_name=None, figsize=(8, 5), colormap="viridis",
                      question_map=None):
    grouped_data = df.groupby([col1, col2]).size().unstack()

    # Get nicer labels if available
    x_label = get_column_label(col1, question_map)
    legend_label = get_column_label(col2, question_map)

    plt.figure(figsize=figsize)
    grouped_data.plot(kind="bar", stacked=True, colormap=colormap, figsize=figsize)

    plt.title(f"{x_label} Distribution by {legend_label}\nSegmented by {segment_name}",
              fontsize=14, fontweight="bold")
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title=legend_label)
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_heatmap(df, col1, col2, segment_name="All", cmap="coolwarm", question_map=None):
    """
    Plots a heatmap of the frequency (cross-tabulation) between two categorical columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing your data.
    col1 (str): The first column name (rows of the heatmap).
    col2 (str): The second column name (columns of the heatmap).
    segment_name (str): Name/label for the chart title (default: "All").
    cmap (str): Colormap to use for the heatmap (default: "coolwarm").
    question_map (dict): Optional dictionary mapping column codes to descriptive labels.
    """
    heatmap_data = pd.crosstab(df[col1], df[col2])

    # Get nicer labels if available
    row_label = get_column_label(col1, question_map)
    col_label = get_column_label(col2, question_map)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap=cmap, fmt="d")

    plt.title(f"Heatmap of {row_label} vs {col_label}\nSegmented by {segment_name}",
              fontsize=14, fontweight="bold")
    plt.ylabel(row_label)
    plt.xlabel(col_label)
    plt.show()




################################
### Comparative Segmentation ###
################################


def plot_grouped_bar(df, x_column, title, xlabel, ylabel, figsize=(8, 5)):
    """
    Plots a grouped bar chart comparing different segments.

    Args:
        df (pd.DataFrame): Segmented DataFrame with 'Segment' column.
        x_column (str): Column to compare.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=x_column, hue="Segment", data=df, palette="coolwarm")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title="Segment")
    plt.show()

### Plot Stacked Bar Chart
def plot_stacked_bar(df, category_col, segment_col, title, xlabel, ylabel, figsize=(10, 6), colormap="viridis"):
    """
    Plots a stacked bar chart showing the proportion of a category across segments.

    Args:
        df (pd.DataFrame): Segmented DataFrame with 'Segment' column.
        category_col (str): Column with categorical values.
        segment_col (str): Column defining segments.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)

    # Get category distributions per segment
    category_dist = df.groupby([segment_col, category_col]).size().unstack()
    category_dist = category_dist.div(category_dist.sum(axis=1), axis=0)  # Convert to proportions

    category_dist.plot(kind="bar", stacked=True, colormap=colormap, figsize=figsize)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=category_col)
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

### Create a Facet Grid (Multiple Subplots for Each Segment)
def plot_facet_grid(df, x_column, title, xlabel, ylabel, figsize=(5, 5)):
    """
    Creates a FacetGrid of histograms per segment.

    Args:
        df (pd.DataFrame): Segmented DataFrame with 'Segment' column.
        x_column (str): Column for histogram.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Individual subplot size.
    """
    g = sns.FacetGrid(df, col="Segment", height=figsize[0], aspect=1)
    g.map_dataframe(sns.histplot, x=x_column, discrete=True)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(xlabel, ylabel)
    plt.show()

### Side-by-Side Heatmaps
def plot_side_by_side_heatmaps(segments, titles, row_var, col_var):
    """
    Plots heatmaps side by side for multiple segments, ensuring all categories are aligned.

    Args:
        segments (list of pd.DataFrame): List of DataFrames for each segment.
        titles (list of str): Titles for each subplot.
        row_var (str): Row variable for heatmap.
        col_var (str): Column variable for heatmap.
    """
    # Step 1: Get all unique categories across all segments
    all_rows = set()
    all_cols = set()

    for df_segment in segments:
        all_rows.update(df_segment[row_var].dropna().unique())  # Collect unique row values
        all_cols.update(df_segment[col_var].dropna().unique())  # Collect unique column values

    # Convert to sorted lists for consistent ordering
    all_rows = sorted(all_rows)
    all_cols = sorted(all_cols)

    # Step 2: Create subplots for heatmaps
    fig, axes = plt.subplots(1, len(segments), figsize=(6 * len(segments), 6))

    for i, df_segment in enumerate(segments):
        # Compute crosstab and reindex to ensure consistency
        heatmap_data = pd.crosstab(df_segment[row_var], df_segment[col_var])
        heatmap_data = heatmap_data.reindex(index=all_rows, columns=all_cols, fill_value=0)  # Fill missing categories with 0

        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt="d", ax=axes[i])
        axes[i].set_title(f"{titles[i]}")
        axes[i].set_xlabel(col_var)
        axes[i].set_ylabel(row_var)

    plt.tight_layout()
    plt.show()


def plot_stacked_percentage_bar(df, category_col, segment_col, title, xlabel, ylabel, figsize=(10, 6)):
    """
    Plots a 100% stacked bar chart showing the proportion of a category within each segment.

    Args:
        df (pd.DataFrame): Segmented DataFrame with 'Segment' column.
        category_col (str): Column with categorical values.
        segment_col (str): Column defining segments.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)

    # Compute proportions for each segment
    category_dist = df.groupby([segment_col, category_col]).size().unstack()
    category_dist = category_dist.div(category_dist.sum(axis=1), axis=0)  # Convert to percentages

    # Plot stacked bar chart
    category_dist.plot(kind="bar", stacked=True, colormap="viridis", figsize=figsize)
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=category_col, loc="upper right")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.show()


def plot_multi_column_counts(df, columns, segment_col, feature_label_map, title, xlabel, ylabel, figsize=(10, 6), stacked=True):
    """
    Plots a grouped or stacked bar chart for multiple columns (e.g., parking types) across segments.

    Args:
        df (pd.DataFrame): Segmented DataFrame with 'Segment' column.
        columns (list): List of column names to count.
        segment_col (str): Column that defines segments.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
        stacked (bool): If True, plot a stacked bar chart; otherwise, use grouped bars.
    """
    # Sum the counts for each column, grouped by segment
    counts = df.groupby(segment_col)[columns].sum()
    
    # Rename columns using feature_label_map for readability (if applicable)
    counts = counts.rename(columns=feature_label_map)

    # Plot the data
    counts.plot(kind="bar", stacked=stacked, colormap="viridis", figsize=figsize)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Parking Type", loc="upper right")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

def plot_multi_column_counts_percentage(df, columns, segment_col, feature_label_map, title, xlabel, ylabel, figsize=(10, 6), stacked=True, show_percent=True):
    """
    Plots a grouped or stacked bar chart for multiple columns (e.g., parking types) across segments.
    Adds percentage labels on top of bars within each segment.

    Args:
        df (pd.DataFrame): Segmented DataFrame with 'Segment' column.
        columns (list): List of column names to count.
        segment_col (str): Column that defines segments.
        feature_label_map (dict): Dictionary mapping column names to readable labels.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
        stacked (bool): If True, plot a stacked bar chart; otherwise, use grouped bars.
        show_percent (bool): If True, adds percentage values on top of bars.
    """
    # Sum the counts for each column, grouped by segment
    counts = df.groupby(segment_col)[columns].sum()

    # Rename columns using feature_label_map for readability (if applicable)
    counts = counts.rename(columns=feature_label_map)

    # Compute percentages **within each segment** (row-wise normalization)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100  # Row-wise division

    # Create plot
    ax = counts.plot(kind="bar", stacked=stacked, colormap="viridis", figsize=figsize)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Parking Type", loc="upper right")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add percentage text on bars
    if show_percent:
        for i, segment in enumerate(counts.index):  # Iterate over segments
            total_segment = counts.loc[segment].sum()  # Get total count for segment
            for j, category in enumerate(counts.columns):  # Iterate over categories
                value = counts.loc[segment, category]
                percent = percentages.loc[segment, category]  # Get percentage
                if value > 0:  # Only label bars with values
                    bar = ax.patches[i * len(counts.columns) + j]  # Locate the correct bar
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_y() + bar.get_height() / 2 if stacked else bar.get_height()

                    ax.text(x, y, f"{value:.0f} ({percent:.1f}%)", ha='center', va='center', fontsize=10, color="black")

    plt.show()



def plot_vehicle_presence_heatmaps(df, vehicle_type_columns, time_periods, segment_col, title_prefix=""):
    """
    Plots heatmaps for vehicle presence at home, categorized by time period and vehicle type.

    Args:
        df (pd.DataFrame): DataFrame containing vehicle presence data.
        vehicle_type_columns (dict): Dictionary mapping vehicle types to their corresponding column lists.
        time_periods (list): List of time period labels.
        segment_col (str): Column that defines segments.
        title_prefix (str): Prefix for heatmap titles (e.g., segment name).
    """
    # Ensure all time slots have data even if missing in the segment
    day_counts = range(6)  # 0 to 5 days

    # Prepare storage for heatmap data
    heatmap_data = []
    row_labels = []

    # Iterate over each vehicle type and time period
    for vehicle_label, vehicle_columns in vehicle_type_columns.items():
        for time_idx, time_label in enumerate(time_periods):
            # Convert to numeric, replace errors with NaN, fill NaNs with 0, then convert to int
            df[vehicle_columns[time_idx]] = df[vehicle_columns[time_idx]].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

            # Count occurrences of each value (0-5 days at home)
            counts = df[vehicle_columns[time_idx]].value_counts().reindex(day_counts, fill_value=0).values

            # Store heatmap data
            heatmap_data.append(counts)
            row_labels.append(f"{vehicle_label} - {time_label}")  # Proper row label for heatmap

    # Convert heatmap data into a DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=["0 - Weekdays", "1", "2", "3", "4", "5"])

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", fmt="d", linewidths=0.5, cbar_kws={"label": "Number of Respondents"})

    plt.title(f"{title_prefix}Car Presence at Home by Vehicle Type and Time Period")
    plt.xlabel("Days at Home")
    plt.ylabel("Vehicle Type & Time Period")
    plt.xticks(rotation=0)  # Keep labels readable
    plt.yticks(rotation=0)

    plt.show()


def plot_vehicle_presence_percentage_heatmaps(df, vehicle_type_columns, time_periods, segment_col, title_prefix=""):
    """
    Plots heatmaps showing the percentage of respondents whose cars are at home, 
    categorized by time period and vehicle type.

    Args:
        df (pd.DataFrame): DataFrame containing vehicle presence data.
        vehicle_type_columns (dict): Dictionary mapping vehicle types to their corresponding column lists.
        time_periods (list): List of time period labels.
        segment_col (str): Column that defines segments.
        title_prefix (str): Prefix for heatmap titles (e.g., segment name).
    """
    # Ensure all time slots have data even if missing in the segment
    day_counts = range(6)  # 0 to 5 days

    # Get total respondents in this segment (to compute percentages)
    total_respondents = len(df)

    # Prepare storage for heatmap data
    heatmap_data = []
    row_labels = []

    # Iterate over each vehicle type and time period
    for vehicle_label, vehicle_columns in vehicle_type_columns.items():
        for time_idx, time_label in enumerate(time_periods):
            # Convert to numeric, replace errors with NaN, fill NaNs with 0, then convert to int
            df[vehicle_columns[time_idx]] = df[vehicle_columns[time_idx]].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

            # Count occurrences of each value (0-5 days at home) and convert to percentages
            counts = df[vehicle_columns[time_idx]].value_counts().reindex(day_counts, fill_value=0)
            percentages = (counts / total_respondents) * 100  # Convert to percentage

            # Store heatmap data
            heatmap_data.append(percentages.values)
            row_labels.append(f"{vehicle_label} - {time_label}")  # Proper row label for heatmap

    # Convert heatmap data into a DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=["0 - Weekdays", "1", "2", "3", "4", "5"])

    # Plot the heatmap with percentages
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5, cbar_kws={"label": "Percentage of Respondents"})

    plt.title(f"{title_prefix}Car Presence at Home by Vehicle Type and Time Period (Percentage)")
    plt.xlabel("Days at Home")
    plt.ylabel("Vehicle Type & Time Period")
    plt.xticks(rotation=0)  # Keep labels readable
    plt.yticks(rotation=0)

    plt.show()


def plot_vehicle_presence_percentage_side_by_side(segments, segment_names, vehicle_type_columns, time_periods):
    """
    Plots side-by-side heatmaps showing the percentage of respondents whose cars are at home, 
    categorized by time period and vehicle type for multiple segments.

    Args:
        segments (list of pd.DataFrame): List of DataFrames, one per segment.
        segment_names (list of str): List of segment names (titles for each subplot).
        vehicle_type_columns (dict): Dictionary mapping vehicle types to their corresponding column lists.
        time_periods (list): List of time period labels.
    """
    num_segments = len(segments)
    fig, axes = plt.subplots(1, num_segments, figsize=(6 * num_segments, 10), sharey=True)

    # Ensure all time slots have data even if missing in some segments
    day_counts = range(6)  # 0 to 5 days

    for i, (df, segment_name) in enumerate(zip(segments, segment_names)):
        heatmap_data = []
        row_labels = []

        # Get total respondents in this segment
        total_respondents = len(df)

        for vehicle_label, vehicle_columns in vehicle_type_columns.items():
            for time_idx, time_label in enumerate(time_periods):
                # Convert to numeric, replace errors with NaN, fill NaNs with 0, then convert to int
                df[vehicle_columns[time_idx]] = df[vehicle_columns[time_idx]].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

                # Count occurrences and convert to percentage
                counts = df[vehicle_columns[time_idx]].value_counts().reindex(day_counts, fill_value=0)
                percentages = (counts / total_respondents) * 100  # Convert to percentage

                # Store data
                heatmap_data.append(percentages.values)
                row_labels.append(f"{vehicle_label} - {time_label}")

        # Convert heatmap data into a DataFrame
        heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=["0 - Weekdays", "1", "2", "3", "4", "5"])

        # Plot the heatmap in the current subplot
        sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5, 
                    cbar=(i == num_segments - 1),  # Show colorbar only on the last subplot
                    ax=axes[i])

        axes[i].set_title(f"{segment_name} - Car Presence (%)")
        axes[i].set_xlabel("Days at Home")
        if i == 0:
            axes[i].set_ylabel("Vehicle Type & Time Period")

    plt.tight_layout()
    plt.show()

def plot_likert_heatmap(df, likert_columns, segment_col, category_order, feature_label_map, title_prefix=""):
    """
    Plots a heatmap for Likert-scale responses across multiple segments.

    Args:
        df (pd.DataFrame): Combined DataFrame with a segment identifier column.
        likert_columns (list): List of Likert-scale columns.
        segment_col (str): Name of the column identifying segments.
        category_order (list): Ordered list of Likert-scale response categories.
        feature_label_map (dict): Dictionary mapping column names to descriptive labels.
        title_prefix (str): Prefix for heatmap title.
    """
    # Melt the dataframe for visualization
    df_melted = df.melt(id_vars=[segment_col], value_vars=likert_columns, 
                         var_name='Question', value_name='Answer')

    # Convert to categorical with ordered responses
    df_melted['Answer'] = pd.Categorical(df_melted['Answer'], categories=category_order, ordered=True)

    # Replace short codes with descriptive labels
    df_melted['Question_label'] = df_melted['Question'].replace(feature_label_map)

    # Crosstab to compute frequency
    freq_table = pd.crosstab([df_melted['Question_label'], df_melted[segment_col]], df_melted['Answer'])

    # Normalize by row for percentages
    freq_table_percentage = freq_table.div(freq_table.sum(axis=1), axis=0) * 100

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(freq_table_percentage, annot=True, cmap='Blues', fmt=".1f")
    plt.title(f"{title_prefix} Likert-scale Attitude Heatmap")
    plt.ylabel("Question")
    plt.xlabel("Likert-scale Response")
    plt.show()


def plot_grouped_bar_chart(df, columns, segment_col, feature_label_map, title, ylabel, figsize=(12, 8), colormap="coolwarm", wrap_length=25):
    """
    Plots a grouped bar chart for binary Yes/No responses across multiple segments with wrapped x-axis labels.

    Args:
        df (pd.DataFrame): Combined DataFrame with a segment identifier column.
        columns (list): List of binary columns to analyze.
        segment_col (str): Name of the column identifying segments.
        feature_label_map (dict): Dictionary mapping column names to descriptive labels.
        title (str): Title of the plot.
        ylabel (str): Y-axis label.
        wrap_length (int): Maximum number of characters before wrapping x-axis labels.
    """

    # Convert only the binary columns to numeric 0/1
    df_subset = df[columns].replace({"0": 0, "1": 1}).astype(float)
    df_subset[segment_col] = df[segment_col]  # Keep the segment column

    # Compute percentage Yes per segment
    grouped = df_subset.groupby(segment_col).sum()
    total_responses = df_subset.groupby(segment_col).count()
    prop_yes = (grouped / total_responses) * 100  # Convert to percentage

    # Convert column names to descriptive labels
    prop_yes.index.name = "Segment"
    prop_yes = prop_yes.rename(columns=feature_label_map)

    # **Wrap x-axis labels** using textwrap
    wrapped_labels = [textwrap.fill(label, wrap_length) for label in prop_yes.columns]

    # Plot as grouped bars (side by side)
    ax = prop_yes.T.plot(kind="bar", figsize=figsize, width=0.8, colormap=colormap, edgecolor="black")

    # **Legend inside the plot, top right**
    ax.legend(title="Segment", loc="upper right", bbox_to_anchor=(1, 1), frameon=True)

    plt.xlabel("Benefits of V2G")
    plt.ylabel(ylabel)
    plt.title(title)

    # **Set wrapped labels**
    ax.set_xticklabels(wrapped_labels, rotation=90, ha="right")

    plt.tight_layout()
    plt.show()


def plot_co_occurrence_heatmap(df, columns, segment_col, feature_label_map, title_prefix=""):
    """
    Plots a heatmap of co-occurrence percentages for binary Yes/No responses across segments.

    Args:
        df (pd.DataFrame): Combined DataFrame with a segment identifier column.
        columns (list): List of binary columns to analyze.
        segment_col (str): Name of the column identifying segments.
        feature_label_map (dict): Dictionary mapping column names to descriptive labels.
        title_prefix (str): Prefix for heatmap title.
    """
    for segment in df[segment_col].unique():
        df_segment = df[df[segment_col] == segment][columns].replace({"0": 0, "1": 1}).astype(float)

        # Compute co-occurrence matrix
        co_occurrence = df_segment.T.dot(df_segment)
        co_occurrence_frac = (co_occurrence / len(df_segment)) * 100

        # Rename rows & columns
        co_occurrence_frac.index = co_occurrence_frac.index.to_series().replace(feature_label_map)
        co_occurrence_frac.columns = co_occurrence_frac.columns.to_series().replace(feature_label_map)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(co_occurrence_frac, annot=True, cmap="Blues", fmt=".1f")
        plt.title(f"{title_prefix}{segment} - Co-occurrence of Responses (%)")
        plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_horizontal_stacked_bar_chart(df, columns, segment_col, feature_label_map, title, ylabel):
    """
    Plots a horizontal stacked bar chart for binary Yes/No responses across multiple segments.

    Args:
        df (pd.DataFrame): Combined DataFrame with a segment identifier column.
        columns (list): List of binary columns to analyze.
        segment_col (str): Name of the column identifying segments.
        feature_label_map (dict): Dictionary mapping column names to descriptive labels.
        title (str): Title of the plot.
        ylabel (str): Y-axis label.
    """

    # Convert only the binary columns to numeric 0/1
    df_subset = df[columns].replace({"0": 0, "1": 1}).astype(float)
    df_subset[segment_col] = df[segment_col]  # Keep segment column

    # Compute percentage Yes/No for each segment
    grouped = df_subset.groupby(segment_col).sum()
    total_responses = df_subset.groupby(segment_col).count()
    prop_yes = (grouped / total_responses) * 100  # Convert to percentage
    prop_no = 100 - prop_yes

    # Convert column names to descriptive labels
    prop_yes.index.name = "Segment"
    prop_yes = prop_yes.rename(columns=feature_label_map)
    prop_no = prop_no.rename(columns=feature_label_map)

    # Create stacked DataFrame
    stacked_df = pd.DataFrame({
        "Yes": prop_yes.stack(),
        "No": prop_no.stack()
    }).reset_index()
    stacked_df.rename(columns={'level_1': 'Feature'}, inplace=True)

    # Pivot for plotting
    stacked_pivot = stacked_df.pivot(index="Feature", columns="Segment", values=["Yes", "No"])

    # Plot as horizontal stacked bars
    ax = stacked_pivot["Yes"].plot(kind="barh", stacked=True, color=["cornflowerblue", "lightgray"], figsize=(12, 8), edgecolor="black")

    ax.legend(title="Segment", loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=len(stacked_pivot.columns))
    plt.xlabel("Percentage of respondents")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
