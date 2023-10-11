import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import textwrap
import numpy as np


def split_title(title, max_line_length=30):
    # Use textwrap to split the title into lines
    lines = textwrap.wrap(title, width=max_line_length)
    # Join the lines with line breaks
    return '\n'.join(lines)

def compute_grouped_mean_std(df):
    # Group the DataFrame by the 'Iteration' column
    grouped = df.groupby('Iteration')

    # Compute the mean and standard deviation of 'MCD Mean' for each group
    result = grouped['MCD Mean'].agg(['mean', 'std']).reset_index()

    return result

def plot_mean_std_MCD(df, title=None):
    # Extract 'Iteration', 'mean', and 'std' columns from the DataFrame
    iterations = df['Iteration']
    mean_values = df['mean']
    std_values = df['std']

    # Set the width of the bars
    bar_width = 0.35

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar positions for 'mean' and 'std'
    bar_positions = range(len(iterations))
    bar_positions_std = [pos + bar_width for pos in bar_positions]

    # Create the bar plots
    plt.bar(bar_positions, mean_values, bar_width, label='Mean MCD', yerr=std_values)

    # Set the x-axis labels to 'Iteration' values
    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(iterations, rotation=90)  # Rotate labels 90 degrees

    # Split the title into multiple lines with a maximum line length
    title = split_title(title, max_line_length=70)

    # Set labels and title
    plt.xlabel('Iteration')
    plt.ylabel('MCD Mean')
    plt.title('MCD Mean of ' + str(title))

    # Add a legend
    plt.legend()

    # Adjust margins to ensure labels are fully visible
    plt.subplots_adjust(bottom=0.2)

    # Use tight_layout to improve the plot layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def plot_grouped_comparison(df_dict, keys, bar_width=1, group_spacing=0.7):
    # In case list contains lists
    keys = flatten_list(keys)

    # Create a list of unique Iteration values from the DataFrames
    iteration_values = sorted(set(iteration for df in df_dict.values() for iteration in df['Iteration']))

    # Calculate the number of keys and the number of groups
    num_keys = len(keys)
    num_groups = len(iteration_values)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a color map for the bars
    colors = plt.cm.get_cmap('tab10', num_keys)

    for i, key in enumerate(keys):
        key_df = df_dict[key]
        mean_values_key = key_df['mean']

        # Calculate the bar positions for each group, accounting for spacing
        group_positions = np.arange(num_groups) * (num_keys * (bar_width + group_spacing)) + i * (bar_width + group_spacing)

        plt.bar(group_positions, mean_values_key, bar_width, label=f'{key}', color=colors(i))

    # Set the x-axis labels to 'Iteration' values
    ax.set_xticks(np.arange(num_groups) * (num_keys * (bar_width + group_spacing)) + ((num_keys - 1) * (bar_width + group_spacing)) / 2)
    ax.set_xticklabels(iteration_values, rotation=90)

    # Set labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Mean Value MCD')

    # Add a legend outside the graph
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15))

    # Set the title with an appropriate pad value
    plt.title("Comparison betwen runs", pad=20)

    # Use tight_layout to improve the plot layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def main(config):


    ev_results_csvs = os.listdir(experiment_ev_results_folder)

    dfs_dict ={}
    for ev_results_csv in ev_results_csvs:

        ev_results_csv_path = os.path.join(experiment_ev_results_folder, ev_results_csv)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(ev_results_csv_path)
        df_grouped_by_n_iter = compute_grouped_mean_std(df)
        #plot_mean_std_MCD(df_grouped_by_n_iter, title=ev_results_csv.split("_metrics")[0])

        dfs_dict[ev_results_csv.split("_metrics")[0]] = df_grouped_by_n_iter

    # Compare plots
    key1 = 'sgv_v1_topk_False_topk_g_0.9999_topk_v_0.5_topk_fi_25000_lbd_rec_10.0_lbd_rec_10.0_lambda_id_5_lbd_cl'
    key2 = 'sgv_v1_topk_True_topk_g_0.9999_topk_v_0.5_topk_fi_25000_lbd_rec_10.0_lbd_rec_10.0_lambda_id_5_lbd_cls'
    plot_grouped_comparison(dfs_dict, keys = [list(dfs_dict.keys())[0], list(dfs_dict.keys())[5:]])

    print("Holu")






if __name__ == '__main__':

    # Folder storing the samples from the
    experiment_ev_results_folder = "./evaluation_results/[8_10_2023]_TopK_v2"

    parser = argparse.ArgumentParser()

    # Directories.
    parser.add_argument('--experiment_ev_results_folder', type=str, default=experiment_ev_results_folder)

    config = parser.parse_args()
    main(config)
