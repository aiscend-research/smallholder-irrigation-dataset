import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_num_images(df, title, by_survey=False):
    counts = df['operator_initials'].value_counts().sort_index()

    # if by_survey is True, we need to divide the image count by the number of unique source_file values for each operator
    if by_survey:
        surveys_per_op = (
            df
            .groupby('operator_initials')['source_file']
            .nunique()
            .sort_index()
        )
        counts = counts.div(surveys_per_op)
        

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.ylabel("Number of Images")
    plt.xlabel("Labeler")
    plt.ylim(0, max(counts.values) + 10)
    plt.show()

def plot_irrigation_distribution(df, title):
    # 1) Count how many times each operator gives each irrigation label
    counts = (
        df
        .groupby(['operator_initials','irrigation'])
        .size()
        .unstack(fill_value=0)       # rows: operator, cols: irrigation levels
    )

    # 2) Turn those into fractions *per operator*
    fracs = counts.div(counts.sum(axis=1), axis=0)  # divide each row by its row‐sum

    # 3) Transpose so x-axis is irrigation, columns are operators
    to_plot = fracs.T  # now rows: irrigation, cols: operator_initials

    # 4) Plot
    ax = to_plot.plot(kind='bar', figsize=(10,6))
    ax.set_xlabel("Irrigation Label (1–5)")
    ax.set_ylabel("Fraction of labels")
    ax.set_title(title)
    ax.legend(title="Labeler")
    plt.tight_layout()
    plt.show()

def plot_percent_coverage(df, title, certain_only=False, ymax=None):
    plt.figure(figsize=(10, 6))
    if certain_only == True:
        yvar = "percent_coverage_high_certainty"
    else:
        yvar = "percent_coverage"
    sns.boxplot(data=df, x="operator_initials", y=yvar)
    plt.title(title)
    plt.xlabel("Labeler")
    plt.ylabel("Percent Area Covered")
    if ymax:
        plt.ylim(0, ymax)
    plt.show()

def plot_polygon_size(df, title, stat="avg", certain_only=False, ymax=None):
    plt.figure(figsize=(10, 6))
    if certain_only == True:
        yvar = df[f"poly_{stat}_size_high_certainty"].apply(np.sqrt)
    else:
        yvar = df[f"poly_{stat}_size"].apply(np.sqrt)
    sns.boxplot(data=df, x="operator_initials", y=yvar)
    plt.title(title)
    plt.xlabel("Labeler")
    plt.ylabel(f"Polygon size sqrt(m^2) ({stat})")
    if ymax:
        plt.ylim(0, ymax)
    plt.show()

def plot_coverage_outliers(df, title, threshold=.35, certain_only=False):
    
    if certain_only == True:
        yvar = df["percent_coverage_high_certainty"]
    else:
        yvar = df["percent_coverage"]

    # 1) Count how many times each operator goes over the threshold
    counts = (
        df
        .loc[yvar > threshold]
        .groupby('operator_initials')
        .size()  # rows: operator, number of outliers
    )

    # 2) Turn those into fractions *per operator*
    fracs = counts / counts.sum()  # divide each count by the total sum

    # 3) Transpose so x-axis is irrigation, columns are operators
    to_plot = fracs.T  # now rows: irrigation, cols: operator_initials

    # 4) Plot
    plt.figure(figsize=(10, 6))
    ax = to_plot.plot(kind='bar', figsize=(10,6))
    plt.title(title)
    plt.xlabel("Labeler")
    plt.ylabel(f"Fraction of Outliers (> {threshold})")
    plt.ylim(0, 1)
    plt.show()

def count_surveys_locations_images(df):
    df = df.copy()  # guarantees you’re working on a real object
    surveys   = df['plot_file'].nunique()
    locations = df['site_id'].nunique()

    df['image_id'] = df['site_id'].astype(str) + '_' + df['image_number'].astype(str)
    images = df['image_id'].nunique()

    return {"Surveys" : surveys, 
            "Locations": locations, 
            "Images": images}

def label_count_table(df):
    df = df.copy()  # guarantees you’re working on a real object

    # Count the surveys, locations and images for each operator using "count_surveys_locations_images"
    operator_counts = df.groupby('operator_initials').apply(count_surveys_locations_images).to_dict()

    # Create a DataFrame from the dictionary
    counts_df = pd.DataFrame(operator_counts)

    # Add a total row
    total_counts = count_surveys_locations_images(df)
    total_counts_df = pd.DataFrame(total_counts, index=['Total (unique)']).T

    # Add total not unique row
    total_counts_not_unique = counts_df.sum(axis=1)
    total_counts_not_unique_df = pd.DataFrame(total_counts_not_unique, columns=['Total (not unique)'])

    counts_df = pd.concat([counts_df, total_counts_df, total_counts_not_unique_df], axis=1)

    return counts_df

# Confusion matrix and false positive and false negative rates of marking irrigation for each operator

def confusion_matrix(df, threshold=3):
    # Prepare image identifier
    df['date'] = pd.to_datetime(df[['year','month','day']])
    df['image'] = df['site_id'] + '_' + df['date'].dt.strftime('%Y-%m-%d')

    # Pivot to get whether the irrigation ratings are above the threshold for each operator and image
    pivot_irr = df.pivot(index='image', columns='operator_initials', values='irrigation') >= threshold

    # Identify all operators except AB
    operators = [op for op in pivot_irr.columns if op != 'AB']

    # Create a confusion matrix for each operator
    confusion_matrices = {}
    for op in operators:
        # Create a confusion matrix
        cm = pd.crosstab(pivot_irr['AB'], pivot_irr[op], rownames=['AB'], colnames=[op])
        confusion_matrices[op] = cm

        # Calculate false positive and false negative rates
        false_positive = cm.loc[False, True] / cm.loc[False].sum()
        false_negative = cm.loc[True, False] / cm.loc[True].sum()

        # Make nice plots of the confusion matrices
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap='Blues')
        plt.colorbar(cax)

        # Add numbers inside the confusion matrix
        for (i, j), val in np.ndenumerate(cm.values):
            ax.text(j, i, f'{val}', ha='center', va='center', color='black')

        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns.tolist())
        ax.set_yticklabels(cm.index.tolist())
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual (AB)')
        ax.set_title(f'Confusion Matrix for {op}')

        # Add FN/FP rates below the figure
        plt.figtext(0.5, 0.01, f'FP Rate: {false_positive:.2f}, FN Rate: {false_negative:.2f}', ha='center', fontsize=10)

        plt.show()
    return confusion_matrices

def compare_to_AB(df, df_description, column, jitter=False):
    # Prepare image identifier
    df['date'] = pd.to_datetime(df[['year','month','day']])
    df['image'] = df['site_id'] + '_' + df['date'].dt.strftime('%Y-%m-%d')

    # Pivot to get the target column by operator
    pivot = df.pivot(index='image', columns='operator_initials', values=column)
    # Pivot to get the irrigation ratings by operator
    pivot_irr = df.pivot(index='image', columns='operator_initials', values='irrigation')

    # Identify all operators except AB
    operators = [op for op in pivot.columns if op != 'AB']

    # Assign each operator a distinct color automatically
    cmap = plt.get_cmap('tab10')
    colors = {op: cmap(i % 10) for i, op in enumerate(operators)}

    # Define marker shapes for the other operator’s irrigation rating
    shapes = {
        1: 'o',   # circle
        2: 's',   # square
        3: '^',   # triangle-up
        4: 'D',   # diamond
        5: 'X'    # X
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, op in enumerate(operators):
        data = pivot[['AB', op]].dropna()
        col = colors[op]

        # jitter if requested
        if jitter:
            jitter_x = np.random.normal(0, 0.1, size=len(data))
            jitter_y = np.random.normal(0, 0.1, size=len(data))
        else:
            jitter_x = np.zeros(len(data))
            jitter_y = np.zeros(len(data))

        # get the OTHER operator's irrigation rating for each point
        irr_op = pivot_irr.loc[data.index, op].astype(int)

        # scatter by other-operator rating → shape, and color by operator
        for rating in sorted(irr_op.unique()):
            mask = (irr_op == rating)
            ax.scatter(
                data['AB'][mask] + jitter_x[mask],
                data[op][mask] + jitter_y[mask],
                marker=shapes[rating],
                color=col,
                label=op if rating == sorted(irr_op.unique())[0] else None,
                alpha=0.7,
                edgecolor='k',
                linewidth=0.5
            )

        # fit and draw trend line in the same color
        m, b = np.polyfit(data['AB'], data[op], 1)
        x_vals = np.array([data['AB'].min(), data['AB'].max()])
        y_vals = m * x_vals + b
        ax.plot(x_vals, y_vals, color=col, linewidth=2)

        # compute and annotate R²
        y_pred = m * data['AB'] + b
        ss_res = ((data[op] - y_pred) ** 2).sum()
        ss_tot = ((data[op] - data[op].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        ax.text(
            0.05, 0.95 - i * 0.05,
            f"{op} $R^2$ = {r2:.2f}",
            transform=ax.transAxes,
            color=col,
            fontsize=10
        )

    ax.set_xlabel("AB's " + column)
    ax.set_ylabel("Other operator's " + column)
    ax.set_title("Jittered Operator " + column + " Comparison: " + df_description)
    ax.legend(title='Operator')
    plt.tight_layout()
    plt.show()


def plot_image_counts(df, df_description):
    """
    Plot the number of times each image was found by each operator.
    """

    # Combine date components and site ID into a single image identifier
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['image'] = df['site_id'] + '_' + df['date'].dt.strftime('%Y-%m-%d')

    # Count occurrences per image and operator
    grouped = df.groupby(['image', 'operator_initials']).size().unstack(fill_value=0)

    # Plot a stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Image (site_id_date)')
    ax.set_ylabel('Count of Detections')
    ax.set_title('Number of Times Each Image Was Found by Operator: ' + df_description)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()