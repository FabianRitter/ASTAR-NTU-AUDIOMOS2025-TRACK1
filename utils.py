import pandas as pd
import os
import numpy as np
import scipy
from scipy import stats
import random
import json

SPECIAL_S_IDS = {"S001", "S006", "S013", "S017", "S018", "S033"}    # DEMO system use special prompts list

def get_texts_from_filename(data_dir, filenames):
    prompt_ids = []
    system_ids = []
    for fn in filenames:     # audiomos2025-track1-S002_P044.wav
        fn = fn.replace("audiomos2025-track1-","")
        s_id = fn.split("_")[0]
        p_id = fn.split("_")[1].split(".")[0]
        system_ids.append(s_id)
        prompt_ids.append(p_id)

    df = pd.read_csv(f'{data_dir}/prompt_info.txt', sep='	')
    demo_df = pd.read_csv(f'{data_dir}/demo_prompt_info.txt', sep='	')
    texts = []
    for s_id, p_id in zip(system_ids, prompt_ids):
        if s_id in SPECIAL_S_IDS:   # demo_prompt_info
            demo_id = 'audiomos2025-track1-' + s_id + '_' + p_id + '.wav'
            text = demo_df.loc[demo_df['id'] == demo_id, 'text'].values
        else:   # prompt_info
            text = df.loc[df['id'] == p_id, 'text'].values
        
        if len(text) > 0:
            texts.append(text[0])
        else:
            texts.append(None)
    return texts

def compute_metrics(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true_np = np.array(y_true).squeeze() # .squeeze() to remove any single dimensions like (N,1) -> (N,)
    y_pred_np = np.array(y_pred).squeeze()

    # Check for empty or insufficient data after conversion
    if y_true_np.size == 0 or y_pred_np.size == 0 or len(y_true_np) < 2: # Check .size for numpy arrays
        print("compute_metrics received empty or insufficient data. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan

    mse  = np.mean((y_true_np - y_pred_np)**2) # Use the numpy arrays
    
    if np.std(y_true_np) == 0 or np.std(y_pred_np) == 0:
        # If one has stddev 0 and the other doesn't, LCC is undefined (NaN).
        # If both have stddev 0, they are constant. If they are the same constant, LCC could be 1 (perfectly correlated),
        # but np.corrcoef might return NaN or raise warning. Let's return NaN to be safe if any std is 0.
        lcc = np.nan
    else:
        lcc  = np.corrcoef(y_true_np, y_pred_np)[0,1]
    
    try:
        srcc = stats.spearmanr(y_true_np, y_pred_np).correlation
    except (ValueError, TypeError, ZeroDivisionError): # Added ZeroDivisionError for robustness
        srcc = np.nan
    try:
        ktau = stats.kendalltau(y_true_np, y_pred_np).correlation
    except (ValueError, TypeError, ZeroDivisionError): # Added ZeroDivisionError for robustness
        ktau = np.nan
        
    return mse, lcc, srcc, ktau

# systemID function as provided in Yi-Cheng branch
def systemID(wavID):
    try:
        # Example wavID: "audiomos2025-track1-S002_P044.wav"
        return wavID.replace("audiomos2025-track1-","").split('_')[0]
    except Exception as e:
        # print(f"Error parsing system ID from {wavID}: {e}") # for debugging
        return "unknown_system" # return a placeholder


