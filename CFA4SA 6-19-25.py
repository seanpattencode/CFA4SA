import numpy as np
import torch
import random
import os
import shutil
from scipy.sparse import save_npz, load_npz
import time
import threading
import json
from joblib import dump, load
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Set seeds for reproducibility
seed_value = 32
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# --- ETA TRACKING HELPERS ---
class ETATracker:
    """A simple progress tracker for iterable operations."""
    def __init__(self, total, description="Processing", print_interval_seconds=10):
        self.total = total
        self.description = description
        self.print_interval_seconds = print_interval_seconds
        self.count = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        print(f"Starting: {self.description} ({self.total} items total)...")

    def step(self, increment=1):
        """Call this after each item is processed."""
        self.count += increment
        current_time = time.time()
        if (current_time - self.last_print_time >= self.print_interval_seconds) or (self.count == self.total):
            self.print_status()
            self.last_print_time = current_time

    def print_status(self):
        elapsed_time = time.time() - self.start_time
        if self.count == 0 or elapsed_time < 1e-6:
            return
        items_per_second = self.count / elapsed_time
        remaining_items = self.total - self.count
        if items_per_second > 0 and remaining_items > 0:
            eta_seconds = remaining_items / items_per_second
            eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        else:
            eta_formatted = "00:00:00"
        progress_percent = (self.count / self.total) * 100
        elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        print(f"  > {self.description}: {self.count}/{self.total} ({progress_percent:.1f}%) | "
              f"Elapsed: {elapsed_formatted} | ETA: {eta_formatted}")

    def finish(self):
        """Call this after the loop is complete."""
        total_time = time.time() - self.start_time
        total_time_formatted = time.strftime('%H:%M:%S', time.gmtime(total_time))
        print(f"Finished: {self.description} in {total_time_formatted}.")


def monitor_long_operation(func, description, *args, **kwargs):
    """
    Executes a blocking function and prints a 'still running' message every 10 seconds.
    This is for non-iterable operations like model.fit().
    """
    result_container = {}
    event = threading.Event()

    def target():
        try:
            result_container['result'] = func(*args, **kwargs)
        except Exception as e:
            result_container['exception'] = e
        finally:
            event.set()

    print(f"Starting: {description} (this may take a while)...")
    start_time = time.time()
    thread = threading.Thread(target=target)
    thread.start()
    seconds_elapsed = 0
    while not event.is_set():
        event.wait(10)
        if not event.is_set():
            seconds_elapsed += 10
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(seconds_elapsed))
            print(f"  > ... {description} still running for {elapsed_str}")

    thread.join()
    end_time = time.time()
    total_time_str = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    if 'exception' in result_container:
        print(f"Finished: {description} with an ERROR after {total_time_str}.")
        raise result_container['exception']
    else:
        print(f"Finished: {description} in {total_time_str}.")
        return result_container.get('result')

# --- Configuration and Cache Setup ---
SUBSET_SIZE = 25000

CACHE_DIR = f"cache_results_subset_{SUBSET_SIZE}"
MODEL_DIR = os.path.join(CACHE_DIR, 'models')
TOKENIZER_DIR = os.path.join(CACHE_DIR, 'tokenizers')
DATA_DIR = os.path.join(CACHE_DIR, 'data')
PREDICTION_DIR = os.path.join(CACHE_DIR, 'predictions')
RESULTS_DIR = os.path.join(CACHE_DIR, 'results')

for dir_path in [CACHE_DIR, MODEL_DIR, TOKENIZER_DIR, DATA_DIR, PREDICTION_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"Using cache directory: {CACHE_DIR}")
print(f"Results will be saved to: {RESULTS_DIR}")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

# --- PIECEWISE SAVING FOR RANDOM FOREST ---
def save_random_forest_piecewise(model, path):
    """
    Saves a RandomForestClassifier in pieces without using pickle for the main object.
    Each tree is saved individually using joblib. Model parameters are saved in a JSON file.
    """
    os.makedirs(path, exist_ok=True)
    
    params = model.get_params()
    if 'estimators' in params:
        del params['estimators']
    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
        
    np.save(os.path.join(path, 'classes_.npy'), model.classes_)
    if hasattr(model, 'n_features_in_'):
        with open(os.path.join(path, 'n_features_in_.txt'), 'w') as f:
            f.write(str(model.n_features_in_))
    if hasattr(model, 'n_outputs_'):
        with open(os.path.join(path, 'n_outputs_.txt'), 'w') as f:
            f.write(str(model.n_outputs_))
            
    estimators_path = os.path.join(path, 'estimators')
    os.makedirs(estimators_path, exist_ok=True)
    
    print(f"Saving {len(model.estimators_)} trees to {estimators_path}...")
    tracker = ETATracker(len(model.estimators_), description="Saving individual trees")
    for i, estimator in enumerate(model.estimators_):
        dump(estimator, os.path.join(estimators_path, f'tree_{i}.joblib'))
        tracker.step()
    tracker.finish()

def load_random_forest_piecewise(path):
    """
    Loads a RandomForestClassifier from a directory of saved pieces.
    Returns None if the model cache is incomplete or does not exist.
    """
    if not os.path.isdir(path):
        return None
        
    params_file = os.path.join(path, 'params.json')
    estimators_path = os.path.join(path, 'estimators')
    if not os.path.isfile(params_file) or not os.path.isdir(estimators_path):
        return None

    try:
        print(f"Loading Random Forest from pieces at {path}...")
        
        with open(params_file, 'r') as f:
            params = json.load(f)
            
        model = RandomForestClassifier(**params)
        
        model.classes_ = np.load(os.path.join(path, 'classes_.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'n_features_in_.txt')):
            with open(os.path.join(path, 'n_features_in_.txt'), 'r') as f:
                model.n_features_in_ = int(f.read())
        if os.path.exists(os.path.join(path, 'n_outputs_.txt')):
            with open(os.path.join(path, 'n_outputs_.txt'), 'r') as f:
                model.n_outputs_ = int(f.read())

        if hasattr(model, 'n_outputs_') and model.n_outputs_ > 1:
            model.n_classes_ = [len(c) for c in model.classes_]
        else:
            model.n_classes_ = len(model.classes_)

        estimator_files = sorted(os.listdir(estimators_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if model.n_estimators != len(estimator_files):
            print(f"Warning: Mismatch in estimator count. Expected {model.n_estimators}, found {len(estimator_files)}. Cache is invalid.")
            return None

        model.estimators_ = []
        tracker = ETATracker(len(estimator_files), description="Loading individual trees")
        for f in estimator_files:
            estimator = load(os.path.join(estimators_path, f))
            model.estimators_.append(estimator)
            tracker.step()
        tracker.finish()
        
        print("Random Forest loaded successfully.")
        return model

    except Exception as e:
        print(f"Warning: Failed to load model from {path} due to an error: {e}. Cache is likely corrupt.")
        return None

# --- Plotting Configuration ---
# Set matplotlib parameters for better quality plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Create output directory for plots
output_dir = os.path.join(RESULTS_DIR, "model_performance_plots")
os.makedirs(output_dir, exist_ok=True)

# Method Name Mapping
method_mapping = {
    'Div Score': 'WSCDS',
    'Div Rank': 'WRCDS',
    'AVG Score': 'ASC',
    'AVG Rank': 'ARC',
    'Perf Score': 'WSCP',
    'Perf Rank': 'WRCP'
}

# Define Score-based and Rank-based methods
score_based_methods = ['WSCDS', 'ASC', 'WSCP']
rank_based_methods = ['WRCDS', 'ARC', 'WRCP']

# Simplified color scheme based on Score vs. Rank
method_colors = {
    # Score-based methods (BLUE)
    'WSCDS': 'blue',
    'ASC':   'blue',
    'WSCP':  'blue',

    # Rank-based methods (RED)
    'WRCDS': 'red',
    'ARC':   'red',
    'WRCP':  'red'
}

# Patterns for differentiating methods within the same color
method_patterns = {
    'WSCDS': None,      # solid
    'WRCDS': '///',     # diagonal lines
    'ASC':   None,      # solid
    'ARC':   '...',     # dots
    'WSCP':  None,      # solid
    'WRCP':  'xxx'      # crosses
}

# --- Plotting Function ---
def create_performance_plot(df_original, score_col, metric_name, method_category, output_filename):
    """
    Creates a matplotlib bar plot where combinations are grouped by model
    count (1, 2, 3...) and sorted by performance within each group.
    """
    df_plot_full = df_original.copy()
    df_plot_full['Num_Models'] = df_plot_full['Models'].apply(len)

    # Calculate Baseline
    best_individual_score = None
    best_individual_model_name = ""
    individual_models_df = df_plot_full[df_plot_full['Num_Models'] == 1]
    if not individual_models_df.empty:
        best_individual_score = individual_models_df[score_col].max()
        best_models = individual_models_df[individual_models_df[score_col] == best_individual_score]['Models'].unique()
        best_individual_model_name = ", ".join(best_models)

    # Filter Data based on method_category
    df_plot = df_original.copy()
    title_suffix = ""
    category_keyword = ""
    sc_methods_in_category = []

    if method_category == 'Average':
        category_keyword = "AVG"
        title_suffix = "(Average Methods - ASC, ARC)"
        sc_methods_in_category = ['ASC']
    elif method_category == 'Performance':
        category_keyword = "Perf"
        title_suffix = "(Performance Methods - WSCP, WRCP)"
        sc_methods_in_category = ['WSCP']
    elif method_category == 'Diversity':
        category_keyword = "Div"
        title_suffix = "(Diversity Methods - WSCDS, WRCDS)"
        sc_methods_in_category = ['WSCDS']

    if category_keyword:
        df_plot = df_plot[df_plot['Method'].str.contains(category_keyword, case=False, na=False)]

    if df_plot.empty:
        print(f"Warning: No data found for metric '{metric_name}' and category '{method_category}'. Skipping plot.")
        return None

    # Map Method names to Abbreviations
    df_plot['Method'] = df_plot['Method'].map(method_mapping).fillna(df_plot['Method'])

    # Get all model combinations in the filtered data
    model_combinations = df_plot['Models'].unique()
    
    # Calculate the mean score for Score Combination methods specific to this category
    df_all = df_original.copy()
    df_all['Method'] = df_all['Method'].map(method_mapping).fillna(df_all['Method'])
    
    # Filter for Score Combination methods in this specific category
    df_sc = df_all[df_all['Method'].isin(sc_methods_in_category)]
    
    # Calculate mean SC score for each model combination
    sc_mean_scores = df_sc.groupby('Models')[score_col].mean().reset_index()
    sc_mean_scores.rename(columns={score_col: 'SC_Mean_Score'}, inplace=True)
    sc_mean_scores['Num_Models'] = sc_mean_scores['Models'].apply(len)
    
    # Sort by number of models first, then by SC mean score (increasing)
    sorted_combinations_df = sc_mean_scores.sort_values(
        by=['Num_Models', 'SC_Mean_Score'],
        ascending=[True, True]
    )
    
    # Filter to only include combinations that are in our current plot data
    category_order = [combo for combo in sorted_combinations_df['Models'].tolist() 
                      if combo in model_combinations]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for grouped bars
    methods = sorted(df_plot['Method'].unique())
    score_methods_in_plot = [m for m in methods if m in score_based_methods]
    rank_methods_in_plot = [m for m in methods if m in rank_based_methods]
    
    # Order methods: Score-based first (left), then Rank-based (right)
    ordered_methods = sorted(score_methods_in_plot) + sorted(rank_methods_in_plot)
    
    n_methods = len(ordered_methods)
    n_combinations = len(category_order)
    bar_width = 0.8 / n_methods
    x_base = np.arange(n_combinations)
    
    # Plot each method as a set of bars
    for i, method in enumerate(ordered_methods):
        method_df = df_plot[df_plot['Method'] == method].copy()
        
        values = []
        for model_combo in category_order:
            combo_data = method_df[method_df['Models'] == model_combo]
            if not combo_data.empty:
                values.append(combo_data[score_col].values[0])
            else:
                values.append(0)
        
        x_positions = x_base + (i - n_methods/2 + 0.5) * bar_width
        
        edgecolor = 'black'
        linewidth = 1.5
        if method in rank_based_methods:
            edgecolor = method_colors.get(method, 'black')
            linewidth = 2.5
        
        bars = ax.bar(x_positions, values,
                      bar_width,
                      label=method,
                      color=method_colors.get(method, 'gray'),
                      hatch=method_patterns.get(method, None),
                      edgecolor=edgecolor,
                      linewidth=linewidth,
                      alpha=0.8)

    # Add baseline
    if best_individual_score is not None:
        ax.axhline(y=best_individual_score, color='grey', linestyle=':', linewidth=2,
                  label=f'Best Single ({best_individual_model_name}): {best_individual_score:.4f}')

    # Add vertical lines to separate the groups
    group_sizes = [len(combo) for combo in category_order]
    if len(group_sizes) > 1:
        last_size = group_sizes[0]
        for i, size in enumerate(group_sizes[1:], 1):
            if size != last_size:
                ax.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
                last_size = size

    # Customize plot
    ax.set_xticks(x_base)
    ax.set_xticklabels(category_order, rotation=45, ha='right')
    ax.set_xlabel('Model Combination (Grouped by Size, then Sorted by SC Performance)')
    ax.set_ylabel(f'{metric_name} Score')
    ax.set_title(f'{metric_name} Performance {title_suffix}')
    ax.legend(title='Method / Baseline', loc='best', framealpha=0.9)
    ax.grid(True, which='major', axis='y', alpha=0.3)
    ax.grid(False, which='major', axis='x')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Set y-axis limits to better show differences
    if len(values) > 0:
        all_values = [v for v in values if v > 0]
        if all_values:
            y_min = min(all_values) * 0.995
            y_max = max(all_values) * 1.005
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(output_dir, output_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}")
    
    return True

def create_metric_table_variable_length(df_metric, score_col):
    placeholder = "---"
    df_metric['Num_Models'] = df_metric['Models'].apply(len)

    individuals = df_metric[df_metric['Num_Models'] == 1].copy()
    best_individuals = individuals.loc[individuals.groupby('Models')[score_col].idxmax()]
    best_individuals = best_individuals.sort_values(score_col, ascending=False)
    individual_list = [f"{row['Models']}: {row[score_col]:.5f}" for _, row in best_individuals.iterrows()]
    
    model_order = ['A', 'B', 'C', 'D']
    temp_dict = {item.split(':')[0]: item for item in individual_list}
    individual_list_sorted = [temp_dict.get(model, f"{model}: N/A") for model in model_order]
    individual_list_sorted.sort(key=lambda x: float(x.split(': ')[1]) if 'N/A' not in x else -1, reverse=True)

    best_single_score = best_individuals[score_col].max() if not best_individuals.empty else -1

    score_method_list = ['WSCP', 'ASC', 'WSCDS']
    rank_method_list = ['WRCP', 'ARC', 'WRCDS']

    score_combos = df_metric[
        (df_metric['Num_Models'] > 1) &
        (df_metric['Method'].isin(score_method_list)) &
        (df_metric[score_col] > best_single_score)
    ].copy()
    score_combos = score_combos.sort_values(score_col, ascending=False)
    sc_list = [f"{row['Models']} ({row['Method']}): {row[score_col]:.5f}" for _, row in score_combos.iterrows()]

    rank_combos = df_metric[
        (df_metric['Num_Models'] > 1) &
        (df_metric['Method'].isin(rank_method_list)) &
        (df_metric[score_col] > best_single_score)
    ].copy()
    rank_combos = rank_combos.sort_values(score_col, ascending=False)
    rc_list = [f"{row['Models']} ({row['Method']}): {row[score_col]:.5f}" for _, row in rank_combos.iterrows()]

    max_len = max(len(individual_list_sorted), len(sc_list), len(rc_list))
    while len(individual_list_sorted) < max_len:
        individual_list_sorted.append(placeholder)
    while len(sc_list) < max_len:
        sc_list.append(placeholder)
    while len(rc_list) < max_len:
        rc_list.append(placeholder)

    table_df = pd.DataFrame({
        'Individual Model': individual_list_sorted,
        'Top Score Combinations (SC)': sc_list,
        'Top Rank Combinations (RC)': rc_list
    })
    return table_df

# === ENSEMBLE METRICS CALCULATION ===
def _process_single_combination(combo_keys, strengths, actual_labels, rank_score_data, key_to_idx, use_ranks, median_ranks_map):
    """
    Worker function to calculate metrics for a single ensemble combination.
    This function is designed to be called in parallel.
    """
    combo_name = "+".join([k.split('_')[1] for k in combo_keys])
    total_strength = sum(strengths[k] for k in combo_keys)
    
    if total_strength == 0:
        return None

    weights = np.array([strengths[k] / total_strength for k in combo_keys])
    data_key = 'ranks' if use_ranks else 'normalized'
    
    data_arrays = np.vstack([rank_score_data[key_to_idx[k]][data_key] for k in combo_keys])
    ensemble_values = np.average(data_arrays, axis=0, weights=weights)

    if use_ranks:
        median_val = median_ranks_map[combo_name]
        predictions = (ensemble_values <= median_val).astype(int)
    else:
        predictions = (ensemble_values >= 0.5).astype(int)

    metrics = {
        'Accuracy': accuracy_score(actual_labels, predictions),
        'Precision': precision_score(actual_labels, predictions, zero_division=0),
        'Recall': recall_score(actual_labels, predictions, zero_division=0),
        'F1 Score': f1_score(actual_labels, predictions, zero_division=0)
    }
    
    return combo_name, metrics

def calculate_ensemble_metrics_parallel(strengths, actual_labels, rank_score_data, use_ranks=False, description=""):
    """
    Calculates ensemble metrics in parallel for all possible combinations.
    Uses joblib to distribute work across CPU cores and tqdm for a progress bar.
    """
    model_keys = list(strengths.keys())
    key_to_idx = {key: i for i, key in enumerate(model_keys)}
    
    all_combos = []
    for r in range(1, len(model_keys) + 1):
        all_combos.extend(combinations(model_keys, r))

    median_ranks_map = {}
    if use_ranks:
        print(f"Pre-calculating median ranks for {len(all_combos)} combinations...")
        for combo_keys in tqdm(all_combos, desc="Calculating Medians"):
            combo_name = "+".join([k.split('_')[1] for k in combo_keys])
            total_strength = sum(strengths[k] for k in combo_keys)
            if total_strength > 0:
                weights = np.array([strengths[k] / total_strength for k in combo_keys])
                data_arrays = np.vstack([rank_score_data[key_to_idx[k]]['ranks'] for k in combo_keys])
                ensemble_values = np.average(data_arrays, axis=0, weights=weights)
                median_ranks_map[combo_name] = np.median(ensemble_values)

    results = Parallel(n_jobs=-1)(
        delayed(_process_single_combination)(
            combo_keys, strengths, actual_labels, rank_score_data, key_to_idx, use_ranks, median_ranks_map
        ) 
        for combo_keys in tqdm(all_combos, desc=f"Calculating Ensembles ({description})")
    )
    
    scores_dict = {name: metrics for name, metrics in results if name is not None}
    return scores_dict

# === MAIN ANALYSIS SCRIPT ===
dataset = load_dataset('imdb')
train_dataset = dataset['train']
test_dataset = dataset['test']

base_test_predictions = []
base_test_predictions_raw = []
base_train_predictions = []
base_train_predictions_raw = []
base_model_names = []
base_models = []

# --- 1. Handle Transformer Model, Tokenizer, and Predictions ---
transformer_model_names = ['aychang/roberta-base-imdb']
transformer_model_name_safe = transformer_model_names[0].replace('/', '_')

model_path = os.path.join(MODEL_DIR, transformer_model_name_safe)
tokenizer_path = os.path.join(TOKENIZER_DIR, transformer_model_name_safe)
test_preds_path = os.path.join(PREDICTION_DIR, f"{transformer_model_name_safe}_test_preds.npy")
test_raw_preds_path = os.path.join(PREDICTION_DIR, f"{transformer_model_name_safe}_test_raw.npy")
train_preds_path = os.path.join(PREDICTION_DIR, f"{transformer_model_name_safe}_train_preds.npy")
train_raw_preds_path = os.path.join(PREDICTION_DIR, f"{transformer_model_name_safe}_train_raw.npy")

if all(os.path.exists(p) for p in [model_path, tokenizer_path, test_preds_path, train_preds_path]):
    print(f"Loading CACHED Transformer model, tokenizer, and predictions for {transformer_model_names[0]}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    base_test_predictions.append(np.load(test_preds_path))
    base_test_predictions_raw.append(np.load(test_raw_preds_path))
    base_train_predictions.append(np.load(train_preds_path))
    base_train_predictions_raw.append(np.load(train_raw_preds_path))
    base_models.append(model)
    base_model_names.append(transformer_model_names[0])
else:
    print(f"GENERATING Transformer predictions for {transformer_model_names[0]}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model_names[0])
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_names[0])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    test_texts, train_texts = test_dataset['text'][:SUBSET_SIZE], train_dataset['text'][:SUBSET_SIZE]
    test_encoding = tokenizer(test_texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    train_encoding = tokenizer(train_texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    model.to(device); model.eval()
    
    batch_size = 20
    test_preds, test_preds_raw, train_preds, train_preds_raw = [], [], [], []
    with torch.no_grad():
        test_loader = DataLoader(list(zip(test_encoding['input_ids'], test_encoding['attention_mask'])), batch_size=batch_size)
        tracker = ETATracker(len(test_loader), f"Transformer predicting on test set")
        for batch in test_loader:
            outputs = model(batch[0].to(device), attention_mask=batch[1].to(device))[0]
            test_preds.extend(outputs.argmax(dim=-1).cpu().numpy()); test_preds_raw.extend(outputs[:, 1].cpu().numpy())
            tracker.step()
        tracker.finish()

        train_loader = DataLoader(list(zip(train_encoding['input_ids'], train_encoding['attention_mask'])), batch_size=batch_size)
        tracker = ETATracker(len(train_loader), f"Transformer predicting on train set")
        for batch in train_loader:
            outputs = model(batch[0].to(device), attention_mask=batch[1].to(device))[0]
            train_preds.extend(outputs.argmax(dim=-1).cpu().numpy()); train_preds_raw.extend(outputs[:, 1].cpu().numpy())
            tracker.step()
        tracker.finish()

    print("SAVING Transformer model, tokenizer, and predictions...")
    model.save_pretrained(model_path); tokenizer.save_pretrained(tokenizer_path)
    np.save(test_preds_path, np.array(test_preds)); np.save(test_raw_preds_path, np.array(test_preds_raw))
    np.save(train_preds_path, np.array(train_preds)); np.save(train_raw_preds_path, np.array(train_preds_raw))

    base_test_predictions.append(np.array(test_preds)); base_test_predictions_raw.append(np.array(test_preds_raw))
    base_train_predictions.append(np.array(train_preds)); base_train_predictions_raw.append(np.array(train_preds_raw))
    base_models.append(model); base_model_names.append(transformer_model_names[0])
    model.to('cpu'); del model; torch.cuda.empty_cache()

# --- 2. Handle Data Preparation for Traditional ML Models ---
train_texts, train_labels = shuffle(train_dataset['text'], train_dataset['label'], random_state=42)
test_texts, test_labels = shuffle(test_dataset['text'], test_dataset['label'], random_state=42)

def get_balanced_subset(texts, labels, size):
    unique_classes, subset_texts, subset_labels = np.unique(labels), [], []
    per_class = size // len(unique_classes)
    for cls in unique_classes:
        cls_indices = np.where(np.array(labels) == cls)[0][:per_class]
        subset_texts.extend(np.array(texts)[cls_indices].tolist())
        subset_labels.extend(np.array(labels)[cls_indices].tolist())
    return subset_texts, subset_labels

subset_train_texts, subset_train_labels = get_balanced_subset(train_texts, train_labels, SUBSET_SIZE)
subset_test_texts, subset_test_labels = get_balanced_subset(test_texts, test_labels, SUBSET_SIZE)

X_train_path, X_test_path = os.path.join(DATA_DIR, 'X_train.npz'), os.path.join(DATA_DIR, 'X_test.npz')
y_train_path, y_test_path = os.path.join(DATA_DIR, 'y_train.npy'), os.path.join(DATA_DIR, 'y_test.npy')
vectorizer_path = os.path.join(TOKENIZER_DIR, 'count_vectorizer.joblib')

if all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path, vectorizer_path]):
    print("Loading CACHED pre-processed data (X_train, X_test, etc.)...")
    X_train, X_test = load_npz(X_train_path), load_npz(X_test_path)
    y_train, y_test = np.load(y_train_path), np.load(y_test_path)
    vectorizer = load(vectorizer_path)
else:
    print("GENERATING and vectorizing ML data...")
    vectorizer = CountVectorizer(max_features=20000)
    X_train = monitor_long_operation(vectorizer.fit_transform, "Vectorizing train data", subset_train_texts)
    X_test = monitor_long_operation(vectorizer.transform, "Vectorizing test data", subset_test_texts)
    y_train, y_test = np.array(subset_train_labels), np.array(subset_test_labels)

    print("SAVING processed data to cache...")
    save_npz(X_train_path, X_train); save_npz(X_test_path, X_test)
    np.save(y_train_path, y_train); np.save(y_test_path, y_test)
    dump(vectorizer, vectorizer_path)

# --- 3. Handle SVC and XGBoost Models (Cache Predictions Only) ---
other_ml_models_defs = {
    'SVC': SVC(probability=True, kernel='linear', random_state=42),
    'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for model_key in other_ml_models_defs.keys():
    test_preds_path = os.path.join(PREDICTION_DIR, f"{model_key}_test_preds.npy")
    test_raw_preds_path = os.path.join(PREDICTION_DIR, f"{model_key}_test_raw.npy")
    train_preds_path = os.path.join(PREDICTION_DIR, f"{model_key}_train_preds.npy")
    train_raw_preds_path = os.path.join(PREDICTION_DIR, f"{model_key}_train_raw.npy")

    if all(os.path.exists(p) for p in [test_preds_path, test_raw_preds_path, train_preds_path, train_raw_preds_path]):
        print(f"Loading CACHED predictions for {model_key}...")
        test_preds, test_preds_raw = np.load(test_preds_path), np.load(test_raw_preds_path)
        train_preds, train_preds_raw = np.load(train_preds_path), np.load(train_raw_preds_path)
    else:
        print(f"GENERATING predictions for {model_key}...")
        model = other_ml_models_defs[model_key]
        monitor_long_operation(model.fit, f"Fitting {model_key}", X_train, y_train)
        
        test_preds = monitor_long_operation(model.predict, f"Predicting with {model_key} on test set", X_test)
        test_preds_raw = monitor_long_operation(model.predict_proba, f"Predicting probs with {model_key} on test", X_test)[:, 1]
        train_preds = monitor_long_operation(model.predict, f"Predicting with {model_key} on train set", X_train)
        train_preds_raw = monitor_long_operation(model.predict_proba, f"Predicting probs with {model_key} on train", X_train)[:, 1]

        print(f"SAVING predictions for {model_key}...")
        np.save(test_preds_path, test_preds); np.save(test_raw_preds_path, test_preds_raw)
        np.save(train_preds_path, train_preds); np.save(train_raw_preds_path, train_preds_raw)

    base_test_predictions.append(test_preds); base_test_predictions_raw.append(test_preds_raw)
    base_train_predictions.append(train_preds); base_train_predictions_raw.append(train_preds_raw)
    base_model_names.append(model_key); base_models.append(f"<{model_key} object - not saved>")

# Store data for plots generation later
all_results_for_plots = {}

# --- 4. Main Loop for Random Forest Configurations ---
rf_configs = [
    {'n_estimators': 100, 'name': 'RandomForest_100'},
    {'n_estimators': 1000, 'name': 'RandomForest_1000'},
    {'n_estimators': 25000, 'name': 'RandomForest_25000'}, #larger models will take longer, but the ensemble does improve as random forest scales. This is simply the maximum size reachable given time and resources.
]

for config in rf_configs:
    target_estimators = config['n_estimators']
    model_name_short = config['name']
    model_name_full = f"RandomForestClassifier(n_estimators={target_estimators})"
    rf_model_path = os.path.join(MODEL_DIR, model_name_short)

    print("\n" + "="*80); print(f"STARTING ANALYSIS FOR {model_name_full}"); print("="*80 + "\n")
    
    test_predictions = [np.copy(p) for p in base_test_predictions]; test_predictions_raw = [np.copy(p) for p in base_test_predictions_raw]
    train_predictions = [np.copy(p) for p in base_train_predictions]; train_predictions_raw = [np.copy(p) for p in base_train_predictions_raw]
    model_names = list(base_model_names); models = list(base_models)

    # --- ROBUST MODEL GENERATION WITH RETRY LOGIC ---
    rf_model = None
    retries = 1
    while retries >= 0:
        try:
            loaded_model = load_random_forest_piecewise(rf_model_path)
            
            if loaded_model is None:
                print(f"No cached model found for {model_name_short}. Training from scratch...")
                rf_model = RandomForestClassifier(n_estimators=target_estimators, random_state=42, n_jobs=-1)
                monitor_long_operation(rf_model.fit, f"Fitting {model_name_short}", X_train, y_train)

                print(f"Saving {model_name_short} piecewise...")
                save_random_forest_piecewise(rf_model, rf_model_path)
            else:
                rf_model = loaded_model

            print(f"Generating predictions for {model_name_short}...")
            test_preds = monitor_long_operation(rf_model.predict, f"Predicting with {model_name_short} on test set", X_test)
            test_preds_raw = monitor_long_operation(rf_model.predict_proba, f"Predicting probs with {model_name_short} on test", X_test)[:, 1]
            train_preds = monitor_long_operation(rf_model.predict, f"Predicting with {model_name_short} on train set", X_train)
            train_preds_raw = monitor_long_operation(rf_model.predict_proba, f"Predicting probs with {model_name_short} on train", X_train)[:, 1]

            break

        except Exception as e:
            print("\n" + "!"*20 + " CACHE/MODEL ERROR " + "!"*20)
            print(f"An error occurred while processing {model_name_short}: {e}")
            rf_model = None
            
            if retries > 0:
                print("Assuming the cache is corrupt. Deleting it and forcing regeneration...")
                if os.path.isdir(rf_model_path):
                    shutil.rmtree(rf_model_path)
                    print(f"Deleted cache directory: {rf_model_path}")
                retries -= 1
            else:
                print(f"Regeneration failed for {model_name_short}. Skipping this model configuration.")
                break
    
    if rf_model is None:
        print("\n" + "="*80); print(f"SKIPPING ANALYSIS FOR {model_name_full} DUE TO ERRORS"); print("="*80 + "\n")
        continue

    test_predictions.append(test_preds); test_predictions_raw.append(test_preds_raw)
    train_predictions.append(train_preds); train_predictions_raw.append(train_preds_raw)
    model_names.append(model_name_full); models.append(rf_model)
    
    print("\n--- Model Roster for this Run ---")
    model_roster_text = []
    for i, (model_name, model) in enumerate(zip(model_names, models)):
        model_str = str(model)
        if len(model_str) > 100: model_str = model_str[:100] + '...>'
        roster_line = f'Model {chr(ord("A") + i)} ({model_name}): {model_str}'
        print(roster_line)
        model_roster_text.append(roster_line)
    
    roster_filename = os.path.join(RESULTS_DIR, f"model_roster_RF_{target_estimators}.txt")
    with open(roster_filename, 'w') as f:
        f.write("Model Roster for this Run\n")
        f.write("="*50 + "\n")
        for line in model_roster_text:
            f.write(line + "\n")
    print(f"Model roster saved to: {roster_filename}")

    # Normalize predictions
    normalized_test_raw_predictions, normalized_train_raw_predictions = [], []
    for test_raw, train_raw in zip(test_predictions_raw, train_predictions_raw):
        train_min, train_max = train_raw.min(), train_raw.max()
        
        norm_train = (train_raw - train_min) / (train_max - train_min) if (train_max - train_min) > 0 else np.zeros_like(train_raw)
        norm_test = (test_raw - train_min) / (train_max - train_min) if (train_max - train_min) > 0 else np.zeros_like(test_raw)
        
        normalized_train_raw_predictions.append(norm_train)
        normalized_test_raw_predictions.append(norm_test)

    # Generate Ranks
    test_rank_and_score, train_rank_and_score = [], []
    for test_orig, test_norm, train_orig, train_norm in zip(test_predictions_raw, normalized_test_raw_predictions, train_predictions_raw, normalized_train_raw_predictions):
        test_ranks = np.empty_like(test_norm.argsort()); test_ranks[test_norm.argsort()] = np.arange(len(test_norm), 0, -1)
        test_rank_and_score.append({'original': test_orig, 'normalized': test_norm, 'ranks': test_ranks})
        train_ranks = np.empty_like(train_norm.argsort()); train_ranks[train_norm.argsort()] = np.arange(len(train_norm), 0, -1)
        train_rank_and_score.append({'original': train_orig, 'normalized': train_norm, 'ranks': train_ranks})
    
    # Diversity Calculation
    def cognitive_diversity(model_a, model_b):
        scores_model_a = {rank: score for rank, score in zip(model_a['ranks'], model_a['normalized'])}
        scores_model_b = {rank: score for rank, score in zip(model_b['ranks'], model_b['normalized'])}
        
        n = len(scores_model_a)
        diversity_sum = 0
        
        for rank in scores_model_a.keys():
            term = (scores_model_a[rank] - scores_model_b[rank]) ** 2
            diversity_sum += term
        
        return sqrt(diversity_sum / n) if n > 0 else 0

    test_div_strengths, train_div_strengths = {}, {}
    num_models = len(test_rank_and_score)
    div_model_keys = [f"Model_{chr(ord('A') + i)}_{name.split('(')[0]}" for i, name in enumerate(model_names)]

    for i in range(num_models):
        test_div_sum, train_div_sum = 0, 0
        for j in range(num_models):
            if i == j: continue
            test_div_sum += cognitive_diversity(test_rank_and_score[i], test_rank_and_score[j])
            train_div_sum += cognitive_diversity(train_rank_and_score[i], train_rank_and_score[j])
        
        test_div_strengths[div_model_keys[i]] = (test_div_sum / (num_models - 1)) if num_models > 1 else 0
        train_div_strengths[div_model_keys[i]] = (train_div_sum / (num_models - 1)) if num_models > 1 else 0

    print("\n--- Calculated Diversity Strengths (Test) ---")
    print("NOTE: Test dataset diversity is calculated without using actual labels - based only on model predictions")
    diversity_test_dict = {k: round(v, 4) for k, v in test_div_strengths.items()}
    print(diversity_test_dict)
    
    print("\n--- Calculated Diversity Strengths (Train) ---")
    diversity_train_dict = {k: round(v, 4) for k, v in train_div_strengths.items()}
    print(diversity_train_dict)
    
    diversity_filename = os.path.join(RESULTS_DIR, f"diversity_strengths_RF_{target_estimators}.txt")
    with open(diversity_filename, 'w') as f:
        f.write("Calculated Diversity Strengths\n")
        f.write("="*50 + "\n\n")
        f.write("Test Dataset Diversity:\n")
        f.write("NOTE: Test dataset diversity is calculated without using actual labels - based only on model predictions\n")
        for k, v in diversity_test_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\n\nTrain Dataset Diversity:\n")
        for k, v in diversity_train_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"Diversity strengths saved to: {diversity_filename}")

    # Plotting Section
    test_ranked_scores_by_model = {}
    
    for i, (model_name, data) in enumerate(zip(model_names, test_rank_and_score)):
        ranks = data['ranks']
        scores = data['normalized']
        
        rank_to_scores = {}
        for rank, score in zip(ranks, scores):
            if rank not in rank_to_scores:
                rank_to_scores[rank] = []
            rank_to_scores[rank].append(score)
        
        test_ranked_scores_by_model[model_name] = rank_to_scores
    
    plt.figure(figsize=(10, 6))
    interval = 250
    marker_size = 100
    markers = ['o', '+', 'x', '*', 's', 'D', '^', 'v']
    
    for i, (model_name, ranked_scores) in enumerate(test_ranked_scores_by_model.items()):
        all_ranks = sorted(ranked_scores.keys())
        selected_ranks = all_ranks[::interval]
        aggregated_x = []
        aggregated_y = []
        for rank in selected_ranks:
            scores = ranked_scores[rank]
            aggregated_x.extend([rank] * len(scores))
            aggregated_y.extend(scores)
        marker = markers[i % len(markers)]
        plt.scatter(aggregated_x, aggregated_y, label=model_name, marker=marker, s=marker_size)
        plt.plot(aggregated_x, aggregated_y)
    
    plt.legend(title='Model Names', loc='upper right')
    plt.xlabel('Ranks')
    plt.ylabel('Normalized Prediction Scores')
    plt.title('Rank Score Graph by Model (Test Dataset)')
    
    plot_filename = os.path.join(RESULTS_DIR, f"rank_score_graph_rf_{target_estimators}.png")
    plt.savefig(plot_filename)
    print(f"\n--- Rank Score Graph saved to: {plot_filename} ---")
    
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Metrics
    test_actual_labels, train_actual_labels = y_test, y_train
    
    train_accuracy_scores, train_f1_scores, train_precision_scores, train_recall_scores = [], [], [], []
    
    print("\n--- Individual Model Metrics (Test Dataset) ---")
    test_metrics_text = []
    for i, model_name in enumerate(model_names):
        metrics = {
            'Acc': accuracy_score(test_actual_labels, test_predictions[i]),
            'Prec': precision_score(test_actual_labels, test_predictions[i], zero_division=0),
            'F1': f1_score(test_actual_labels, test_predictions[i], zero_division=0),
            'Rec': recall_score(test_actual_labels, test_predictions[i], zero_division=0)
        }
        metrics_line = f"Model: {model_name:<35} | " + " | ".join([f"{k}: {v:.4f}" for k,v in metrics.items()])
        print(metrics_line)
        test_metrics_text.append(metrics_line)
    
    print("\n--- Individual Model Metrics (Train Dataset) ---")
    train_metrics_text = []
    for i, model_name in enumerate(model_names):
        metrics = {
            'Acc': accuracy_score(train_actual_labels, train_predictions[i]),
            'Prec': precision_score(train_actual_labels, train_predictions[i], zero_division=0),
            'F1': f1_score(train_actual_labels, train_predictions[i], zero_division=0),
            'Rec': recall_score(train_actual_labels, train_predictions[i], zero_division=0)
        }
        train_accuracy_scores.append(metrics['Acc'])
        train_f1_scores.append(metrics['F1'])
        train_precision_scores.append(metrics['Prec'])
        train_recall_scores.append(metrics['Rec'])
        
        metrics_line = f"Model: {model_name:<35} | " + " | ".join([f"{k}: {v:.4f}" for k,v in metrics.items()])
        print(metrics_line)
        train_metrics_text.append(metrics_line)
    
    metrics_filename = os.path.join(RESULTS_DIR, f"individual_model_metrics_RF_{target_estimators}.txt")
    with open(metrics_filename, 'w') as f:
        f.write("Individual Model Metrics\n")
        f.write("="*50 + "\n\n")
        f.write("Test Dataset Metrics:\n")
        for line in test_metrics_text:
            f.write(line + "\n")
        f.write("\n\nTrain Dataset Metrics:\n")
        for line in train_metrics_text:
            f.write(line + "\n")
    print(f"Individual model metrics saved to: {metrics_filename}")
    
    # Ensemble Analysis
    total_acc_train = sum(train_accuracy_scores)
    perf_strengths_accuracy = {key: train_accuracy_scores[i] / total_acc_train if total_acc_train > 0 else 0 for i, key in enumerate(div_model_keys)}
    
    total_f1_train = sum(train_f1_scores)
    perf_strengths_f1 = {key: train_f1_scores[i] / total_f1_train if total_f1_train > 0 else 0 for i, key in enumerate(div_model_keys)}
    
    total_prec_train = sum(train_precision_scores)
    perf_strengths_precision = {key: train_precision_scores[i] / total_prec_train if total_prec_train > 0 else 0 for i, key in enumerate(div_model_keys)}
    
    total_rec_train = sum(train_recall_scores)
    perf_strengths_recall = {key: train_recall_scores[i] / total_rec_train if total_rec_train > 0 else 0 for i, key in enumerate(div_model_keys)}
    
    avg_strengths = {key: 1.0 / len(div_model_keys) for key in div_model_keys}
    
    # Score Combination (SC) calculations
    perf_scores_acc_sc = calculate_ensemble_metrics_parallel(perf_strengths_accuracy, test_actual_labels, test_rank_and_score, use_ranks=False, description="Perf Acc SC")
    perf_scores_f1_sc = calculate_ensemble_metrics_parallel(perf_strengths_f1, test_actual_labels, test_rank_and_score, use_ranks=False, description="Perf F1 SC")
    perf_scores_prec_sc = calculate_ensemble_metrics_parallel(perf_strengths_precision, test_actual_labels, test_rank_and_score, use_ranks=False, description="Perf Prec SC")
    perf_scores_rec_sc = calculate_ensemble_metrics_parallel(perf_strengths_recall, test_actual_labels, test_rank_and_score, use_ranks=False, description="Perf Rec SC")
    
    div_test_scores_sc = calculate_ensemble_metrics_parallel(test_div_strengths, test_actual_labels, test_rank_and_score, use_ranks=False, description="Div-Test SC")
    div_train_scores_sc = calculate_ensemble_metrics_parallel(train_div_strengths, test_actual_labels, test_rank_and_score, use_ranks=False, description="Div-Train SC")
    avg_scores_sc = calculate_ensemble_metrics_parallel(avg_strengths, test_actual_labels, test_rank_and_score, use_ranks=False, description="Avg SC")

    # Rank Combination (RC) calculations
    perf_scores_acc_rc = calculate_ensemble_metrics_parallel(perf_strengths_accuracy, test_actual_labels, test_rank_and_score, use_ranks=True, description="Perf Acc RC")
    perf_scores_f1_rc = calculate_ensemble_metrics_parallel(perf_strengths_f1, test_actual_labels, test_rank_and_score, use_ranks=True, description="Perf F1 RC")
    perf_scores_prec_rc = calculate_ensemble_metrics_parallel(perf_strengths_precision, test_actual_labels, test_rank_and_score, use_ranks=True, description="Perf Prec RC")
    perf_scores_rec_rc = calculate_ensemble_metrics_parallel(perf_strengths_recall, test_actual_labels, test_rank_and_score, use_ranks=True, description="Perf Rec RC")
    
    div_test_scores_rc = calculate_ensemble_metrics_parallel(test_div_strengths, test_actual_labels, test_rank_and_score, use_ranks=True, description="Div-Test RC")
    div_train_scores_rc = calculate_ensemble_metrics_parallel(train_div_strengths, test_actual_labels, test_rank_and_score, use_ranks=True, description="Div-Train RC")
    avg_scores_rc = calculate_ensemble_metrics_parallel(avg_strengths, test_actual_labels, test_rank_and_score, use_ranks=True, description="Avg RC")

    # Combine results into a single list for DataFrame creation
    results_list = []
    all_scores = {
        ('Performance (Train Accuracy)', 'SC'): perf_scores_acc_sc,
        ('Performance (Train F1)',       'SC'): perf_scores_f1_sc,
        ('Performance (Train Precision)','SC'): perf_scores_prec_sc,
        ('Performance (Train Recall)',   'SC'): perf_scores_rec_sc,
        ('Diversity (Test)',             'SC'): div_test_scores_sc,
        ('Diversity (Train)',            'SC'): div_train_scores_sc,
        ('Average',                      'SC'): avg_scores_sc,
        ('Performance (Train Accuracy)', 'RC'): perf_scores_acc_rc,
        ('Performance (Train F1)',       'RC'): perf_scores_f1_rc,
        ('Performance (Train Precision)','RC'): perf_scores_prec_rc,
        ('Performance (Train Recall)',   'RC'): perf_scores_rec_rc,
        ('Diversity (Test)',             'RC'): div_test_scores_rc,
        ('Diversity (Train)',            'RC'): div_train_scores_rc,
        ('Average',                      'RC'): avg_scores_rc,
    }

    for (weighting_type, combo_type), scores_dict in all_scores.items():
        for ensemble_name, metrics in scores_dict.items():
            record = {'Ensemble': ensemble_name, 'Weighting': weighting_type, 'Type': combo_type, **metrics}
            results_list.append(record)

    df = pd.DataFrame(results_list)
    df = df[['Ensemble', 'Weighting', 'Type', 'Accuracy', 'F1 Score', 'Precision', 'Recall']]
    sorted_df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
    print(f"\n--- Top Ensemble Scores by Metric (RF={target_estimators}) ---")
    print("NOTE: Performance weighting now uses all four Train metrics (Accuracy, F1, Precision, Recall)")
    print("NOTE: Diversity (Test) is calculated without using actual test labels")
    print(sorted_df)
    
    csv_filename = os.path.join(RESULTS_DIR, f"ensemble_scores_RF_{target_estimators}.csv")
    sorted_df.to_csv(csv_filename, index=False)
    print(f"Ensemble scores saved to CSV: {csv_filename}")
    
    txt_filename = os.path.join(RESULTS_DIR, f"ensemble_scores_RF_{target_estimators}.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"Top Ensemble Scores by Metric (RF={target_estimators})\n")
        f.write("="*80 + "\n")
        f.write("NOTE: Performance weighting now uses all four Train metrics (Accuracy, F1, Precision, Recall)\n")
        f.write("METHODOLOGY NOTE:\n")
        f.write("NOTE: Diversity (Test) is calculated without using actual test labels\n\n")
        f.write("Test diversity calculation uses only model predictions (not labels) on the test set.\n")
        f.write("This represents a valid batch prediction scenario where diversity among predictions\n")
        f.write("can inform ensemble combination without using ground truth labels.\n")
        f.write(sorted_df.to_string(index=False))
    print(f"Ensemble scores saved to text: {txt_filename}")
    
    # Store results for plotting
    all_results_for_plots[target_estimators] = sorted_df.copy()
    
    print("\n" + "="*80); print(f"COMPLETED ANALYSIS FOR {model_name_full}"); print("="*80 + "\n")

    del rf_model, models

# === GENERATE PLOTS FROM REAL DATA ===
print("\n" + "="*80)
print("GENERATING PERFORMANCE PLOTS FROM REAL DATA")
print("="*80 + "\n")

# Convert real data to plotting format
def convert_results_to_plot_format(df_results):
    """Convert the ensemble results DataFrame to the format expected by plotting functions"""
    plot_data = []
    
    for _, row in df_results.iterrows():
        # Map model combinations
        models = row['Ensemble']
        
        # Map weighting type and combination type to method names
        weighting = row['Weighting']
        combo_type = row['Type']
        
        # Determine method name based on mapping
        if 'Diversity' in weighting and combo_type == 'SC':
            method = 'Div Score'
        elif 'Diversity' in weighting and combo_type == 'RC':
            method = 'Div Rank'
        elif 'Average' in weighting and combo_type == 'SC':
            method = 'AVG Score'
        elif 'Average' in weighting and combo_type == 'RC':
            method = 'AVG Rank'
        elif 'Performance' in weighting and combo_type == 'SC':
            method = 'Perf Score'
        elif 'Performance' in weighting and combo_type == 'RC':
            method = 'Perf Rank'
        else:
            continue  # Skip if cannot map
        
        # Create entries for each metric
        plot_data.append([models, method, row['Accuracy']])
        plot_data.append([models, method, row['Precision']])
        plot_data.append([models, method, row['Recall']])
        plot_data.append([models, method, row['F1 Score']])
    
    return plot_data

# Generate plots for each RF configuration
for target_estimators, df_results in all_results_for_plots.items():
    print(f"\nGenerating plots for RF_{target_estimators}...")
    
    # Convert data
    plot_data = convert_results_to_plot_format(df_results)
    
    # Create DataFrames for each metric
    data_accuracy = [[models, method, score] for models, method, score in plot_data if models and method]
    data_precision = [[models, method, score] for models, method, score in plot_data if models and method]
    data_recall = [[models, method, score] for models, method, score in plot_data if models and method]
    data_f1 = [[models, method, score] for models, method, score in plot_data if models and method]
    
    # Filter by metric (using the fact that we know the order)
    data_accuracy = [d for i, d in enumerate(plot_data) if i % 4 == 0]
    data_precision = [d for i, d in enumerate(plot_data) if i % 4 == 1]
    data_recall = [d for i, d in enumerate(plot_data) if i % 4 == 2]
    data_f1 = [d for i, d in enumerate(plot_data) if i % 4 == 3]
    
    df_accuracy = pd.DataFrame(data_accuracy, columns=['Models', 'Method', 'Accuracy Score'])
    df_precision = pd.DataFrame(data_precision, columns=['Models', 'Method', 'Precision Score'])
    df_recall = pd.DataFrame(data_recall, columns=['Models', 'Method', 'Recall Score'])
    df_f1 = pd.DataFrame(data_f1, columns=['Models', 'Method', 'F1 Score'])

    # Apply Model Combination Label Formatting
    for df in [df_accuracy, df_precision, df_recall, df_f1]:
        df['Models'] = df['Models'].str.replace('+', '', regex=False)

    metrics_data = {
        'accuracy': (df_accuracy, 'Accuracy Score', 'Accuracy'),
        'precision': (df_precision, 'Precision Score', 'Precision'),
        'recall': (df_recall, 'Recall Score', 'Recall'),
        'f1': (df_f1, 'F1 Score', 'F1')
    }

    categories = ['average', 'performance', 'diversity']
    category_map = {'average': 'Average', 'performance': 'Performance', 'diversity': 'Diversity'}

    print(f"\nGenerating plots for RF_{target_estimators}...")
    
    for metric_key, (df, score_col, metric_display_name) in metrics_data.items():
        print(f"\n--- Generating {metric_display_name} Plots for RF_{target_estimators} ---")
        
        df_prepared = df.copy()
        
        for category_key in categories:
            filename = f"{metric_key}_{category_key}_RF_{target_estimators}.png"
            category_display = category_map[category_key]
            
            create_performance_plot(df_prepared, score_col, metric_display_name, category_display, filename)

# Generate Performance Summary Tables
print("\n\nGenerating Performance Summary Tables...")
print("=" * 60)

for target_estimators, df_results in all_results_for_plots.items():
    # Convert data for tables
    plot_data = convert_results_to_plot_format(df_results)
    
    # Create DataFrames for each metric
    data_accuracy = [d for i, d in enumerate(plot_data) if i % 4 == 0]
    data_precision = [d for i, d in enumerate(plot_data) if i % 4 == 1]
    data_recall = [d for i, d in enumerate(plot_data) if i % 4 == 2]
    data_f1 = [d for i, d in enumerate(plot_data) if i % 4 == 3]
    
    df_accuracy = pd.DataFrame(data_accuracy, columns=['Models', 'Method', 'Accuracy Score'])
    df_precision = pd.DataFrame(data_precision, columns=['Models', 'Method', 'Precision Score'])
    df_recall = pd.DataFrame(data_recall, columns=['Models', 'Method', 'Recall Score'])
    df_f1 = pd.DataFrame(data_f1, columns=['Models', 'Method', 'F1 Score'])

    # Apply formatting
    for df in [df_accuracy, df_precision, df_recall, df_f1]:
        df['Models'] = df['Models'].str.replace('+', '', regex=False)
        df['Method'] = df['Method'].map(method_mapping).fillna(df['Method'])

    metrics_data_tables = {
        'Accuracy': (df_accuracy, 'Accuracy Score'),
        'Precision': (df_precision, 'Precision Score'),
        'Recall': (df_recall, 'Recall Score'),
        'F1': (df_f1, 'F1 Score')
    }

    tables_filepath = os.path.join(output_dir, f"performance_summary_tables_RF_{target_estimators}.txt")
    with open(tables_filepath, 'w') as f:
        f.write(f"Performance Summary Tables for RF_{target_estimators}\n")
        f.write("="*80 + "\n\n")
        
        for metric, (df, score_col) in metrics_data_tables.items():
            f.write(f"## {metric} Performance Summary\n\n")
            table_df = create_metric_table_variable_length(df, score_col)
            f.write(table_df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n\n")
            
            print(f"\n## {metric} Performance Summary (RF_{target_estimators})\n")
            print(table_df.to_string(index=False))
            print("\n" + "="*40)

    print(f"\nTables saved to: {os.path.abspath(tables_filepath)}")

print("\n" + "=" * 60)
print(f"All plots have been saved to: {os.path.abspath(output_dir)}")
print("=" * 60)

       # --- IMPORTANT NOTE ON ENSEMBLE WEIGHTING STRATEGIES ---
    #
    # The following results table evaluates several ensemble weighting strategies. It is crucial
    # to understand the theoretical difference between them, particularly "Diversity (Test)".
    #
    # 1.  **Inductive Methods (Standard Evaluation):**
    #     - 'Performance (Train)': Weights are derived from how well models performed on the training set.
    #     - 'Diversity (Train)': Weights are derived from how much models disagreed on the training set.
    #     - 'Average': All models are weighted equally.
    #     These methods create a fixed model based *only* on training data. Their evaluation on the
    #     test set is a true measure of their ability to generalize to new, unseen data points, one by one.
    #
    # 2.  **Transductive Method (Batch-Adaptive Evaluation):**
    #     - 'Diversity (Test)': Weights are derived from how much models disagree with each other on the
    #       **entire test set**. This is a valid and powerful technique when making predictions for a
    #       large, known batch of unlabeled data. It allows the ensemble to adapt its weights based on
    #       the specific characteristics of the data it needs to predict.
    #
    #     **Why this is not "data contamination" in a practical context:** This method does not use
    #     the test set *labels*. It only uses the structural information within the test set's predictions,
    #     which is available at prediction time in a batch-processing scenario, which is extremely common
    #     in real-world applications.
    #
    #     **How to interpret its results:** The performance of 'Diversity (Test)' should be seen as the
    #     result of a specialized, batch-aware system. It is included for comparison.
    #
    # The labels 'Diversity (Train)' and 'Diversity (Test)' are used for clarity to help the reader
    # easily identify the source of data used for each weighting scheme.
    # --- END NOTE ---