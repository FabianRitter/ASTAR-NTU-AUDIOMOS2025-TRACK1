# prepare_inner_split.py
import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append('./code') # If utils.py is in ./code
from utils import systemID # Assuming systemID is in utils.py

def analyze_and_split_data(input_list_path, output_train_path, output_dev_path, 
                           dev_split_ratio=0.1, random_seed=1984,
                           analysis_reference_list_path=None, plot_dir=None):
    """
    Analyzes system ID distribution and performs a stratified split of the input list.
    """
    print(f"Processing list: {input_list_path}")
    with open(input_list_path, 'r') as f:
        all_lines = f.readlines()

    if len(all_lines) < 2:
        print(f"ERROR: Not enough data in {input_list_path} to perform a split.")
        return False

    # Extract system IDs and scores for stratification and analysis
    system_ids = []
    # scores_overall = [] # Optional: for score distribution analysis
    
    parsed_lines_data = [] # To store (line, system_id) for easier splitting

    for line_idx, line in enumerate(all_lines):
        parts = line.strip().split(',')
        if len(parts) < 1:
            print(f"Skipping malformed line {line_idx+1}: {line.strip()}")
            continue
        wav_filename = parts[0]
        sid = systemID(wav_filename)
        if sid is None:
            print(f"Could not parse system ID for {wav_filename}. Using 'unknown_system'.")
            sid = 'unknown_system'
        system_ids.append(sid)
        parsed_lines_data.append({'line': line, 'system_id': sid})
        # if len(parts) > 1: scores_overall.append(float(parts[1]))


    # --- Analysis ---
    print(f"\n--- Analysis for {os.path.basename(input_list_path)} ({len(all_lines)} samples) ---")
    sys_counts_input = Counter(system_ids)
    print("System ID Distribution (Input):")
    for sid, count in sorted(sys_counts_input.items()):
        print(f"  {sid}: {count} ({count/len(all_lines)*100:.2f}%)")

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_distribution(sys_counts_input, f"System Distribution - {os.path.basename(input_list_path)}", 
                          os.path.join(plot_dir, f"sys_dist_{os.path.basename(input_list_path)}.png"))

    if analysis_reference_list_path:
        print(f"\n--- Analysis for Reference List {os.path.basename(analysis_reference_list_path)} ---")
        with open(analysis_reference_list_path, 'r') as f:
            ref_lines = f.readlines()
        ref_system_ids = [systemID(line.strip().split(',')[0]) for line in ref_lines if line.strip()]
        ref_sys_counts = Counter(ref_system_ids)
        print("System ID Distribution (Reference):")
        for sid, count in sorted(ref_sys_counts.items()):
            print(f"  {sid}: {count} ({count/len(ref_lines)*100:.2f}%)")
        if plot_dir:
            plot_distribution(ref_sys_counts, f"System Distribution - {os.path.basename(analysis_reference_list_path)}",
                              os.path.join(plot_dir, f"sys_dist_{os.path.basename(analysis_reference_list_path)}.png"))
    
    # --- Stratified Splitting ---
    # We need indices to pass to train_test_split along with the lines and stratification key
    indices = np.arange(len(parsed_lines_data))

    # Ensure that all system IDs used for stratification have at least 2 samples if test_size > 0
    # or if test_size is a float (which implies at least 1 sample per class for the test set).
    # sklearn's train_test_split with stratify needs at least 2 members per class for a split.
    # If a system has only 1 sample, it cannot be split and will cause an error.
    # We can filter out such systems for stratification or handle them by putting them all in train.
    
    stratify_labels = [data['system_id'] for data in parsed_lines_data]
    unique_labels, counts = np.unique(stratify_labels, return_counts=True)
    
    # For systems with only 1 sample, they can't be stratified correctly if we split.
    # A simple strategy: if a system has only 1 sample, it always goes to the training set.
    # This requires a more custom split. For now, let's rely on train_test_split's behavior.
    # If it errors due to single-member classes, those classes may need to be handled.
    # Check if any class has fewer than 2 samples, which is problematic for stratification.
    problematic_classes = unique_labels[counts < 2]
    if len(problematic_classes) > 0 and dev_split_ratio > 0:
        print(f"WARNING: The following system IDs have only 1 sample and cannot be reliably stratified: {problematic_classes}")
        print("Stratification might not be perfect for these. Consider putting them all in the training set manually or adjusting split.")

    try:
        train_indices, dev_indices = train_test_split(
            indices,
            test_size=dev_split_ratio,
            random_state=random_seed,
            stratify=stratify_labels  # Stratify by system ID
        )
    except ValueError as e:
        print(f"ERROR during stratified split: {e}")
        print("This often happens if a class in 'stratify' has only 1 member. Trying without stratification for problematic classes.")
        # Fallback: if stratification fails, do a random split (less ideal)
        # Or, a more complex handling of single-member classes would be needed.
        # For now, let's just do a random split if stratified fails.
        print("Falling back to random split due to stratification error.")
        train_indices, dev_indices = train_test_split(
            indices,
            test_size=dev_split_ratio,
            random_state=random_seed
        )


    train_lines_split = [parsed_lines_data[i]['line'] for i in train_indices]
    dev_lines_split = [parsed_lines_data[i]['line'] for i in dev_indices]

    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_dev_path), exist_ok=True)

    with open(output_train_path, 'w') as f:
        f.writelines(train_lines_split)
    with open(output_dev_path, 'w') as f:
        f.writelines(dev_lines_split)

    print(f"\nSuccessfully split {input_list_path}:")
    print(f"  New training partition ({len(train_lines_split)} lines) saved to: {output_train_path}")
    print(f"  New inner dev partition ({len(dev_lines_split)} lines) saved to: {output_dev_path}")

    # Analysis of the created splits
    train_split_sids = [systemID(line.strip().split(',')[0]) for line in train_lines_split if line.strip()]
    dev_split_sids = [systemID(line.strip().split(',')[0]) for line in dev_lines_split if line.strip()]
    
    print("\nSystem ID Distribution (New Train Split):")
    for sid, count in sorted(Counter(train_split_sids).items()):
        print(f"  {sid}: {count} ({count/len(train_lines_split)*100:.2f}%)")
    
    print("\nSystem ID Distribution (New Inner Dev Split):")
    for sid, count in sorted(Counter(dev_split_sids).items()):
        print(f"  {sid}: {count} ({count/len(dev_lines_split)*100:.2f}%)")
    
    if plot_dir:
        plot_distribution(Counter(train_split_sids), f"System Distribution - {os.path.basename(output_train_path)}",
                          os.path.join(plot_dir, f"sys_dist_{os.path.basename(output_train_path)}.png"))
        plot_distribution(Counter(dev_split_sids), f"System Distribution - {os.path.basename(output_dev_path)}",
                          os.path.join(plot_dir, f"sys_dist_{os.path.basename(output_dev_path)}.png"))
    return True

def plot_distribution(counts, title, save_path):
    sids = list(counts.keys())
    values = list(counts.values())
    
    # Sort by SIDs for consistent plotting order
    sorted_indices = np.argsort(sids)
    sids = np.array(sids)[sorted_indices]
    values = np.array(values)[sorted_indices]

    plt.figure(figsize=(12, 7))
    plt.bar(sids, values)
    plt.xlabel("System ID")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and perform stratified split of MOS lists.")
    parser.add_argument('--input_list', type=str, required=True, help='Path to the input MOS list file (e.g., original train_mos_list.txt).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the new train and dev split files and plots.')
    parser.add_argument('--output_train_name', type=str, default='train_stratified_split.txt', help='Name for the new training partition file.')
    parser.add_argument('--output_dev_name', type=str, default='dev_stratified_split.txt', help='Name for the new inner dev partition file.')
    parser.add_argument('--dev_split_ratio', type=float, default=0.1, help='Ratio of data for the inner dev set (e.g., 0.1 for 10%).')
    parser.add_argument('--random_seed', type=int, default=1984, help='Random seed for the split.')
    parser.add_argument('--analysis_reference_list', type=str, default=None, help='Optional: Path to a reference list (e.g., original dev_mos_list.txt) for distribution comparison.')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_output_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_output_dir, exist_ok=True)

    output_train_path = os.path.join(args.output_dir, args.output_train_name)
    output_dev_path = os.path.join(args.output_dir, args.output_dev_name)

    analyze_and_split_data(
        args.input_list,
        output_train_path,
        output_dev_path,
        dev_split_ratio=args.dev_split_ratio,
        random_seed=args.random_seed,
        analysis_reference_list_path=args.analysis_reference_list,
        plot_dir=plot_output_dir
    )