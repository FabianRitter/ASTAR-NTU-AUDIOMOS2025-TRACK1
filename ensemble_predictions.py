import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn.linear_model import Ridge # Option for meta-model
from sklearn.model_selection import train_test_split
import numpy as np # Ensure numpy is imported if not already for array operations
import lightgbm as lgb # Option for meta-model
# Assuming your main script's utils and augment are in the path
# If not, adjust sys.path or copy functions
sys.path.append('./code') # Or the correct path to your project code
from utils import systemID # For potential future use, not directly for MOS list
# The scores_to_gaussian_target function from your augment.py
# Copied here for completeness, ensure it's correctly imported or defined
def scores_to_gaussian_target(scores, num_bins, device, sigma=0.5):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.float().to(device)
    clamped_scores = torch.clamp(scores, 1.0, 5.0)
    bin_centers = torch.linspace(1.0, 5.0, num_bins, device=device)
    scores_expanded = clamped_scores.unsqueeze(-1) # Ensure it's (N, 1) for broadcasting
    bin_centers_expanded = bin_centers.unsqueeze(0) # Ensure it's (1, num_bins)
    distances = scores_expanded - bin_centers_expanded
    soft_targets = torch.exp(- (distances ** 2) / (2 * (sigma ** 2)))
    soft_targets = soft_targets / torch.sum(soft_targets, dim=1, keepdim=True)
    return soft_targets

# This function should be in your main script's utils.py or similar
def generate_answer_file_ensemble(wavnames_no_ext, pred_overall_scores, pred_textual_scores, output_filepath):
    logging.info(f"Generating answer file: {output_filepath}")
    predictions_map = {}
    for wav_no_ext, o_score, t_score in zip(wavnames_no_ext, pred_overall_scores, pred_textual_scores):
        predictions_map[wav_no_ext] = (o_score, t_score)

    with open(output_filepath, 'w') as ans:
        sorted_wavIDs_no_ext = sorted(predictions_map.keys())
        for wavID_no_ext in sorted_wavIDs_no_ext:
            overall_score, textual_score = predictions_map[wavID_no_ext]
            outl = f"{wavID_no_ext},{float(overall_score):.8f},{float(textual_score):.8f}\n"
            ans.write(outl)
    logging.info(f"Answer file {output_filepath} generated successfully.")


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 20 # As used by your _dist models
SIGMA_FOR_SCALAR_SMOOTHING = 0.35 # Hyperparameter for Gaussian smoothing

# Define your base models and paths to their predictions
# USER: Fill these paths accurately!
MODEL_INFO = [
    {
        "id": "transformer_standard_gaussian_bias_srcc",
        "type": "dist",
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_eval_list.pt",
    },
    {
        "id": "transformer_standard_gaussian_bias_seed1990_combined",
        "type": "dist",
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    {
        "id": "transformer_standard_gaussian_bias_seed1990_srcc",
        "type": "dist",
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_eval_list.pt",
    },
    {
        "id": "transformer_standard_gaussian_bias_seed1960_combined",
        "type": "dist",
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    {
        "id": "transformer_standard_gaussian_bias_seed1960_srcc",
        "type": "dist",
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_eval_list.pt",
    },
    {
        "id": "transformer_standard_gaussian_bias_seed1910_combined",
        "type": "dist",
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1910/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1910/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_standard_gaussian_bias_seed1510_combined",
        "type": "dist", # Based on bash script: muq_roberta_transformer_dist;gaussian;20
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_standard_gaussian_bias_seed1510_srcc",
        "type": "dist", # Based on bash script: muq_roberta_transformer_dist;gaussian;20
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_decoupled_nolstm_gaussian_bias_whole_combined",
        "type": "dist", # Based on bash script: muq_roberta_transformer_decoupled_dist;gaussian;20
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_decoupled_nolstm_gaussian_bias_whole_srcc",
        "type": "dist", # Based on bash script: muq_roberta_transformer_decoupled_dist;gaussian;20
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_decoupled_lstm_gaussian_bias_whole_combined",
        "type": "dist", # Based on bash script: muq_roberta_transformer_decoupled_and_lst_dist;gaussian;20
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_LSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_LSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_ordinal_mono_wholedata_combined",
        "type": "scalar", # Based on bash script: muq_roberta_transformer_dist_coral;coral;5
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_ordinal_mono_wholedata_srcc",
        "type": "scalar", # Based on bash script: muq_roberta_transformer_dist_coral;coral;5
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_srcc_eval_list.pt",
    },
    { # From your logs
        "id": "transformer_ordinal_mono_wholedata_seed1510_combined",
        "type": "scalar", # Based on bash script: muq_roberta_transformer_dist_coral;coral;5
        "val_pred_file": "./track1_ckpt/transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata_seed_1510/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_dev_mos_list.pt",
        "test_pred_file": "./track1_ckpt/transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata_seed_1510/model_predictions_for_ensemble/detailed_preds_ckpt_best_val_combined_eval_list.pt",
    }
]

# Paths to ground truth files (validation for training meta-model, test for ordering final output)
# This should be the list used to generate the validation predictions
DEV_MOS_LIST_PATH = "/dataset/speech_and_audio_datasets/MusicEval-phase1/sets/dev_mos_list.txt" # Or your custom validation_list_path
# This is the list for the final submission, needed for wavname ordering
#TEST_LIST_PATH = "/dataset/speech_and_audio_datasets/MusicEval-phase1/sets/dev_mos_list.txt" # Or your actual test list
TEST_LIST_PATH = "/dataset/speech_and_audio_datasets/MusicEval-phase1/audiomos2025-track1-eval-phase/DATA/sets/eval_list.txt"

TEST_GT_MOS_LIST_PATH = "/dataset/speech_and_audio_datasets/MusicEval-full/sets/test_mos_list.txt" # This is the labeled test set


OUTPUT_DIR = "./ensemble_outputs_23JUN_for_paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def load_predictions(pred_file_path):
    """ Loads predictions saved by torch.save() """
    if not os.path.exists(pred_file_path):
        logging.error(f"Prediction file not found: {pred_file_path}")
        return None
    try:
        preds = torch.load(pred_file_path, map_location='cpu', weights_only=False)
        # Ensure predictions are numpy arrays for sklearn
        # And ensure distributions are float32
        for wav_id, p_data in preds.items():
            if 'overall_dist' in p_data and p_data['overall_dist'] is not None:
                preds[wav_id]['overall_dist'] = np.array(p_data['overall_dist'], dtype=np.float32)
            if 'textual_dist' in p_data and p_data['textual_dist'] is not None:
                preds[wav_id]['textual_dist'] = np.array(p_data['textual_dist'], dtype=np.float32)
            if 'overall_score' in p_data and p_data['overall_score'] is not None:
                 preds[wav_id]['overall_score'] = float(p_data['overall_score'])
            if 'textual_score' in p_data and p_data['textual_score'] is not None:
                 preds[wav_id]['textual_score'] = float(p_data['textual_score'])
        return preds
    except Exception as e:
        logging.error(f"Error loading prediction file {pred_file_path}: {e}")
        return None

def get_true_mos_scores_and_order(mos_list_file):
    """ Reads a MOS list file and returns true scores and wav_id order. """
    wav_data = {}
    wav_order = [] # To maintain the order of samples as in the list file
    with open(mos_list_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            wav_filename_ext = parts[0]
            # wav_id_no_ext = wav_filename_ext.split('.')[0] # Or wav_filename_ext[:-4]
            if wav_filename_ext.lower().endswith('.wav'):
                 wav_id_no_ext = wav_filename_ext[:-4]
            else:
                 wav_id_no_ext = wav_filename_ext # Should not happen with challenge data

            overall_mos = float(parts[1])
            textual_mos = float(parts[2])
            wav_data[wav_id_no_ext] = {
                'overall_true': overall_mos,
                'textual_true': textual_mos,
                'wavname_with_ext': wav_filename_ext
            }
            wav_order.append(wav_id_no_ext)
    return wav_data, wav_order

def get_mos_list_data(mos_list_file, has_labels=True):
    """
    Reads a MOS list file. If has_labels is False, MOS scores are ignored/dummied.
    Returns: dict {wavID_no_ext: {'true_overall': float, 'true_textual': float, 'wav_filename_ext': str}} (scores are dummy if no_labels)
             list: [wavID_no_ext_in_order]
    """
    wav_data = {}
    wav_order = []
    if not os.path.exists(mos_list_file):
        logging.error(f"MOS list file not found: {mos_list_file}")
        # For official eval list, if not found, it's a critical error for prediction.
        raise FileNotFoundError(f"Required MOS list file not found: {mos_list_file}")

    with open(mos_list_file, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            
            wav_filename_part = parts[0] # This is wavID_no_ext for eval_list, or wavfile_with_ext for dev_list
            wav_id_no_ext = wav_filename_part
            wav_filename_ext = wav_filename_part
            
            if has_labels:
                if not wav_filename_part.lower().endswith('.wav'): # Should be filename for dev
                    logging.error(f"Expected .wav extension in labeled list: {mos_list_file}, line {line_num+1}")
                    # Decide how to handle this, maybe skip or try to append
                    # For now, assume dev_list has .wav
                    wav_id_no_ext = wav_filename_part # if it was already no_ext
                else:
                    wav_id_no_ext = wav_filename_part[:-4]
            else: # Official eval_list.txt only has basename
                wav_filename_ext = wav_filename_part + ".wav" # Assume .wav for internal processing

            if has_labels:
                if len(parts) < 3:
                    logging.warning(f"Skipping malformed line in labeled MOS list {mos_list_file} (line {line_num+1}): {line}")
                    continue
                try:
                    overall_mos = float(parts[1])
                    textual_mos = float(parts[2])
                except ValueError:
                    logging.warning(f"Skipping line with non-float MOS scores in {mos_list_file} (line {line_num+1}): {line}")
                    continue
            else: # No labels for official eval list
                overall_mos, textual_mos = 0.0, 0.0 # Dummy values

            wav_data[wav_id_no_ext] = {
                'true_overall': overall_mos, # Will be dummy for official eval list
                'true_textual': textual_mos, # Will be dummy for official eval list
                'wav_filename_ext': wav_filename_ext
            }
            wav_order.append(wav_id_no_ext)
            
    if not wav_data:
        logging.error(f"No data loaded from {mos_list_file}. Please check the file format and path.")
    return wav_data, wav_order

def prepare_meta_features(model_infos, prediction_set_type, wav_order, num_bins, device_tensor):
    """
    Loads predictions from all models for a given set (val or test),
    converts scalars to distributions, and concatenates them.
    """
    all_overall_dists = []
    all_textual_dists = []
    num_models = len(model_infos)
    logging.info(f"Preparing meta-features for {prediction_set_type} set from {num_models} models.")

    # Pre-load all model predictions for the current set
    loaded_model_preds = []
    for i, model_info in enumerate(model_infos):
        pred_file = model_info[f'{prediction_set_type}_pred_file']
        logging.info(f"Loading {prediction_set_type} predictions for model {model_info['id']} from {pred_file}")
        preds = load_predictions(pred_file)
        if preds is None:
            raise ValueError(f"Could not load predictions for model {model_info['id']}")
        loaded_model_preds.append(preds)

    # Iterate through each wav file in the specified order
    for wav_id in tqdm(wav_order, desc=f"Processing wavs for {prediction_set_type} meta-features"):
        sample_overall_dists = []
        sample_textual_dists = []

        for i, model_info in enumerate(model_infos):
            model_preds_for_wav = loaded_model_preds[i].get(wav_id)
            if model_preds_for_wav is None:
                logging.warning(f"Missing prediction for wav_id {wav_id} in model {model_info['id']}. Using zeros.")
                # Fallback to zeros, or handle more gracefully (e.g., skip sample, mean imputation)
                # For stacking, consistent feature length is crucial.
                overall_d = np.zeros(num_bins, dtype=np.float32)
                textual_d = np.zeros(num_bins, dtype=np.float32)
            else:
                # Overall
                if model_info['type'] == 'dist':
                    overall_d = model_preds_for_wav.get('overall_dist')
                    if overall_d is None or len(overall_d) != num_bins:
                        logging.warning(f"Missing/malformed overall_dist for {wav_id}, model {model_info['id']}. Using zeros.")
                        overall_d = np.zeros(num_bins, dtype=np.float32)
                elif model_info['type'] == 'scalar':
                    scalar_val = model_preds_for_wav.get('overall_score')
                    if scalar_val is None:
                        logging.warning(f"Missing overall_score for {wav_id}, model {model_info['id']}. Using zeros for dist.")
                        overall_d = np.zeros(num_bins, dtype=np.float32)
                    else:
                        overall_d_tensor = scores_to_gaussian_target(torch.tensor([scalar_val]), num_bins, device_tensor, sigma=SIGMA_FOR_SCALAR_SMOOTHING)
                        overall_d = overall_d_tensor.squeeze().cpu().numpy()
                else:
                    raise ValueError(f"Unknown model type: {model_info['type']}")
                sample_overall_dists.append(overall_d)

                # Textual
                if model_info['type'] == 'dist':
                    textual_d = model_preds_for_wav.get('textual_dist')
                    if textual_d is None or len(textual_d) != num_bins:
                        logging.warning(f"Missing/malformed textual_dist for {wav_id}, model {model_info['id']}. Using zeros.")
                        textual_d = np.zeros(num_bins, dtype=np.float32)
                elif model_info['type'] == 'scalar':
                    scalar_val = model_preds_for_wav.get('textual_score')
                    if scalar_val is None:
                        logging.warning(f"Missing textual_score for {wav_id}, model {model_info['id']}. Using zeros for dist.")
                        textual_d = np.zeros(num_bins, dtype=np.float32)
                    else:
                        textual_d_tensor = scores_to_gaussian_target(torch.tensor([scalar_val]), num_bins, device_tensor, sigma=SIGMA_FOR_SCALAR_SMOOTHING)
                        textual_d = textual_d_tensor.squeeze().cpu().numpy()
                else:
                    raise ValueError(f"Unknown model type: {model_info['type']}")
                sample_textual_dists.append(textual_d)
        
        # Concatenate distributions from all models for this sample
        all_overall_dists.append(np.concatenate(sample_overall_dists)) # Shape: (num_models * num_bins)
        all_textual_dists.append(np.concatenate(sample_textual_dists)) # Shape: (num_models * num_bins)

    return np.array(all_overall_dists, dtype=np.float32), np.array(all_textual_dists, dtype=np.float32)


# --- Main Ensemble Logic ---
if __name__ == "__main__":
    logging.info("Starting Stacking Ensemble Process...")
     # 1. Load true MOS scores and wav order for the validation set (used to train meta-model)
    val_true_data, val_wav_order_full = get_true_mos_scores_and_order(DEV_MOS_LIST_PATH)
    val_y_overall_true_full = np.array([val_true_data[wid]['overall_true'] for wid in val_wav_order_full], dtype=np.float32)
    val_y_textual_true_full = np.array([val_true_data[wid]['textual_true'] for wid in val_wav_order_full], dtype=np.float32)
    logging.info(f"Loaded {len(val_wav_order_full)} DEV samples for meta-model training data preparation.")

    # 2. Prepare meta-features for the ENTIRE validation (DEV) set
    val_X_overall_meta_full, val_X_textual_meta_full = prepare_meta_features(MODEL_INFO, "val", val_wav_order_full, NUM_BINS, DEVICE)
    logging.info(f"Full DEV meta-features shape: Overall {val_X_overall_meta_full.shape}, Textual {val_X_textual_meta_full.shape}")

    # === MODIFICATION START: Split DEV meta-data for meta-model training and validation ===
    logging.info("Splitting DEV meta-data into 60% train / 40% validation for meta-model sanity check...")
    
    dev_indices = np.arange(val_X_overall_meta_full.shape[0])

    # Split overall data and indices
    X_meta_train_overall, X_meta_val_overall, \
    y_meta_train_overall, y_meta_val_overall, \
    indices_meta_train, indices_meta_val = train_test_split( # Now expects 6 return values
        val_X_overall_meta_full,   # Array 1
        val_y_overall_true_full,   # Array 2
        dev_indices,               # Array 3
        test_size=0.40,
        random_state=42,
        shuffle=True
    )
    # X_meta_train_overall, X_meta_val_overall are the feature splits for overall
    # y_meta_train_overall, y_meta_val_overall are the target splits for overall
    # indices_meta_train, indices_meta_val are the splits of the original dev_indices

    # Use the split indices to get the corresponding textual data parts
    X_meta_train_textual = val_X_textual_meta_full[indices_meta_train]
    y_meta_train_textual = val_y_textual_true_full[indices_meta_train] # Target for textual train
    
    X_meta_val_textual = val_X_textual_meta_full[indices_meta_val]
    y_meta_val_textual = val_y_textual_true_full[indices_meta_val] # Target for textual val

    # Use the split indices to get the corresponding wav order for the meta-validation set
    val_wav_order_full_np = np.array(val_wav_order_full)
    wav_order_meta_val = val_wav_order_full_np[indices_meta_val].tolist()

    logging.info(f"Split DEV meta-data: {X_meta_train_overall.shape[0]} for meta-training, {X_meta_val_overall.shape[0]} for meta-validation.")
    # === MODIFICATION END ===

    meta_model_configs = [
        # ... (rest of your meta_model_configs definition)
        {
            "name": "ridge",
            "overall_model_class": Ridge, "overall_model_params": {"alpha": 1.0},
            "textual_model_class": Ridge, "textual_model_params": {"alpha": 1.0}
        },
        {
            "name": "lgbm",
            "overall_model_class": lgb.LGBMRegressor, "overall_model_params": {"random_state": 42},
            "textual_model_class": lgb.LGBMRegressor, "textual_model_params": {"random_state": 42}
        }
    ]

    # Prepare meta-features for the FINAL TEST set (do this once outside the loop)
    # ... (rest of your test set preparation)
    _, test_wav_order = get_mos_list_data(TEST_LIST_PATH, has_labels=False)
    logging.info(f"Loaded {len(test_wav_order)} test samples for final prediction.")
    test_X_overall_meta, test_X_textual_meta = prepare_meta_features(MODEL_INFO, "test", test_wav_order, NUM_BINS, DEVICE)
    logging.info(f"Test meta-features shape: Overall {test_X_overall_meta.shape}, Textual {test_X_textual_meta.shape}")


    for config in meta_model_configs:
        config_name = config["name"]
        logging.info(f"\n--- Processing ensemble with meta-model: {config_name} ---")
        
        # === SCENARIO A (Sanity Check on held-out dev data) - NO CHANGE NEEDED ===
        # This part trains on 60% of dev and predicts on 40% of dev. Your code for this is correct.
        logging.info(f"--- SCENARIO A: Training on 60% DEV, Evaluating on 40% DEV ---")
        
        # 3a. Train Meta-Models on the META-TRAIN split of DEV data
        meta_model_overall_cv = config['overall_model_class'](**config['overall_model_params'])
        meta_model_overall_cv.fit(X_meta_train_overall, y_meta_train_overall) 

        meta_model_textual_cv = config['textual_model_class'](**config['textual_model_params'])
        meta_model_textual_cv.fit(X_meta_train_textual, y_meta_train_textual)
        
        # 3b. Predict and Evaluate on META-VALIDATION split
        meta_val_pred_overall = meta_model_overall_cv.predict(X_meta_val_overall)
        meta_val_pred_textual = meta_model_textual_cv.predict(X_meta_val_textual)
        # ... (clipping and saving logic remains the same)
        meta_val_answer_file = os.path.join(OUTPUT_DIR, f"answer_ensemble_dev_split_eval_{config_name}.txt")
        generate_answer_file_ensemble(wav_order_meta_val, meta_val_pred_overall.tolist(), meta_val_pred_textual.tolist(), meta_val_answer_file)
        logging.info(f"Dev (split) evaluation file for {config_name} saved. To evaluate: python eval_ensemble_model.py --submission_file {meta_val_answer_file} --gt_file {DEV_MOS_LIST_PATH}")

        # === PREPARE FOR SCENARIOS B and C ===
        # 4. RETRAIN Meta-Models on FULL DEV meta-data
        logging.info(f"--- Retraining on FULL DEV data for Scenarios B and C ---")
        final_meta_model_overall = config['overall_model_class'](**config['overall_model_params'])
        final_meta_model_overall.fit(val_X_overall_meta_full, val_y_overall_true_full)

        final_meta_model_textual = config['textual_model_class'](**config['textual_model_params'])
        final_meta_model_textual.fit(val_X_textual_meta_full, val_y_textual_true_full)
        logging.info(f"Retrained Meta-Models ({config_name}) on full DEV meta-data.")
        
        # === SCENARIO B (Full Dev Performance) - NEW BLOCK OF CODE ===
        logging.info(f"--- SCENARIO B: Predicting on FULL DEV set (using model trained on full DEV set) ---")
        dev_full_pred_overall = final_meta_model_overall.predict(val_X_overall_meta_full)
        dev_full_pred_textual = final_meta_model_textual.predict(val_X_textual_meta_full)

        dev_full_pred_overall = np.clip(dev_full_pred_overall, 1.0, 5.0)
        dev_full_pred_textual = np.clip(dev_full_pred_textual, 1.0, 5.0)

        # Generate answer file for the full dev set
        dev_full_answer_file = os.path.join(OUTPUT_DIR, f"answer_ensemble_dev_full_eval_{config_name}.txt")
        generate_answer_file_ensemble(
            val_wav_order_full, # Use the full dev wav order
            dev_full_pred_overall.tolist(),
            dev_full_pred_textual.tolist(),
            dev_full_answer_file
        )
        logging.info(f"Full DEV evaluation file for {config_name} saved. To evaluate: python eval_ensemble_model.py --submission_file {dev_full_answer_file} --gt_file {DEV_MOS_LIST_PATH}")
        
        # === SCENARIO C (Final Test Set Submission) - NO CHANGE NEEDED ===
        logging.info(f"--- SCENARIO C: Predicting on TEST set (using model trained on full DEV set) ---")
        test_pred_overall_ensemble = final_meta_model_overall.predict(test_X_overall_meta)
        test_pred_textual_ensemble = final_meta_model_textual.predict(test_X_textual_meta)

        test_pred_overall_ensemble = np.clip(test_pred_overall_ensemble, 1.0, 5.0)
        test_pred_textual_ensemble = np.clip(test_pred_textual_ensemble, 1.0, 5.0)

        # Generate Submission File for the test set
        submission_answer_file = os.path.join(OUTPUT_DIR, f"answer_ensemble_submission_{config_name}.txt")
        generate_answer_file_ensemble(
            test_wav_order, 
            test_pred_overall_ensemble.tolist(),
            test_pred_textual_ensemble.tolist(),
            submission_answer_file
        )
        logging.info(f"Final TEST submission file for {config_name} generated: {submission_answer_file}")
        # You can evaluate this file once the official test ground truth is released.
        logging.info(f"To evaluate final test submission (once GT is available): python eval_ensemble_model.py --submission_file {submission_answer_file} --gt_file {TEST_GT_MOS_LIST_PATH}")


    logging.info(f"\nEnsemble script finished for all configurations.")
    logging.info(f"\nStacking ensemble finished for all configurations.")
    logging.info("Evaluation on meta-validation splits (40% of DEV) provides a sanity check.")
    logging.info("Submission files were generated using meta-models retrained on the FULL DEV meta-data.")