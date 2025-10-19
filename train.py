
"""
Largely adapt codes from:
https://github.com/nii-yamagishilab/mos-finetune-ssl
"""
import os
import sys
sys.path.append('./dataset') 

import argparse
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from transformers import RobertaModel
from tqdm import tqdm
import random
from muq import MuQ
# from torch.utils.data.dataset import Dataset # Not directly used if using custom datasets
from torch.utils.data import DataLoader
from utils import get_texts_from_filename, compute_metrics, systemID
from dataset_mos import MosDataset, PersonMosDataset
from augment import mixup_data, scores_to_one_hot, scores_to_gaussian_target


# new for visualizations:

import matplotlib.pyplot as plt
import seaborn as sns


from models.roberta_transformer import ( MuQRoBERTaTransformerDistributionPredictor,MuQRoBERTaTransformerScalarPredictor, MuQRoBERTaTransformerDistributionPredictorCORALPredictor, MuQRoBERTaTransformerLSTMHeadCrossAttnDecoupledPredictor, MuQMulanRoBERTaTransformerDistributionPredictor, MuQRoBERTaTransformerDecoupledDist)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

# Define your model's hyperparameters (must match your trained model)
model_hyperparams = {
    "num_bins": 20,
    "audio_transformer_layers": 1,
    "audio_transformer_heads": 4,
    "audio_transformer_dim": 1024,
    "common_embed_dim": 768,
    "cross_attention_heads": 4,
    "dropout_rate": 0.3
}



def generate_answer_file(wavnames_with_ext, pred_overall_scores, pred_textual_scores, output_filepath):
    """
    Generates an answer file in the format: wavID_no_ext,overall_score,textual_score
    Args:
        wavnames_with_ext (list): List of wav filenames (e.g., 'file.wav')
        pred_overall_scores (list): List of predicted overall scores.
        pred_textual_scores (list): List of predicted textual scores.
        output_filepath (str): Path to save the answer file.
    """
    logging.info(f"Generating answer file: {output_filepath}")
    
    predictions_overall_dict = {}
    predictions_textual_dict = {}

    if not (len(wavnames_with_ext) == len(pred_overall_scores) == len(pred_textual_scores)):
        logging.error(f"Mismatched lengths in generate_answer_file: wavs({len(wavnames_with_ext)}), overall({len(pred_overall_scores)}), textual({len(pred_textual_scores)})")
        # Fallback: try to process common minimum length if possible, or raise error
        # For now, we'll proceed but this indicates an issue upstream
        min_len = min(len(wavnames_with_ext), len(pred_overall_scores), len(pred_textual_scores))
        wavnames_with_ext = wavnames_with_ext[:min_len]
        pred_overall_scores = pred_overall_scores[:min_len]
        pred_textual_scores = pred_textual_scores[:min_len]


    for i in range(len(wavnames_with_ext)):
        wavID_with_ext = wavnames_with_ext[i]
        # Ensure wavID_no_ext is consistently derived
        if isinstance(wavID_with_ext, str) and wavID_with_ext.lower().endswith('.wav'):
            wavID_no_ext = wavID_with_ext[:-4] # More robust than split
        else:
            wavID_no_ext = wavID_with_ext # Assume it's already without extension if not ending with .wav
            logging.warning(f"Filename '{wavID_with_ext}' does not end with .wav. Using as is for wavID_no_ext.")
            
        predictions_overall_dict[wavID_no_ext] = pred_overall_scores[i]
        predictions_textual_dict[wavID_no_ext] = pred_textual_scores[i]

    with open(output_filepath, 'w') as ans:
        sorted_wavIDs_no_ext = sorted(predictions_overall_dict.keys())
        for wavID_no_ext in sorted_wavIDs_no_ext:
            overall_score = predictions_overall_dict[wavID_no_ext]
            # It's possible that for some reason textual_score might not be there if lists were mismatched
            # but given the input lists are expected to be aligned, this check is more of a safeguard.
            if wavID_no_ext in predictions_textual_dict:
                textual_score = predictions_textual_dict[wavID_no_ext]
                outl = f"{wavID_no_ext},{float(overall_score):.8f},{float(textual_score):.8f}\n" # Ensure float formatting
                ans.write(outl)
            else:
                # This case should ideally not happen if input lists are correct
                logging.warning(f"Critical: Missing textual prediction for {wavID_no_ext} when writing to {output_filepath}. This might indicate a bug in data handling.")
                # Decide on fallback: skip, or write with a placeholder e.g. 0.0 or NaN
                # For now, skipping the line if textual score is missing
    logging.info(f"Answer file {output_filepath} generated successfully.")

def coral_loss_function(logits, true_mos_scores, num_ranks, reduction='mean'):
    """
    Args:
        logits (torch.Tensor): Predicted logits, shape (batch_size, num_ranks - 1).
                               These are for P(score > rank_k).
        true_mos_scores (torch.Tensor): True MOS scores, shape (batch_size,). Values in [1, num_ranks].
        num_ranks (int): Number of distinct rank categories (e.g., 5 for MOS 1-5).
    """
    # Ensure true_mos_scores are on the same device and float
    true_mos_scores = true_mos_scores.float().to(logits.device)
    
    # Create ordinal labels {0, 1} for each of the num_ranks-1 tasks
    # P(Y > k) where k are the thresholds. If ranks are 1,2,3,4,5 (num_ranks=5)
    # Thresholds can be considered 1, 2, 3, 4.
    # So, logit_j corresponds to P(MOS > j+1) for j=0 to num_ranks-2
    
    # rank_thresholds are [1, 2, ..., num_ranks-1]
    rank_thresholds = torch.arange(1, num_ranks, device=logits.device, dtype=torch.float32) # (num_ranks-1,)
    
    # labels_for_bce[i,j] = 1 if true_mos_scores[i] > rank_thresholds[j] else 0
    # true_mos_scores: (batch_size, 1)
    # rank_thresholds: (1, num_ranks-1)
    labels_for_bce = (true_mos_scores.unsqueeze(1) > rank_thresholds.unsqueeze(0)).float() # (batch_size, num_ranks-1)
    
    loss = F.binary_cross_entropy_with_logits(logits, labels_for_bce, reduction='none').sum(dim=1) # Sum over the K-1 tasks
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# --- Helper function to evaluate a loaded model (from previous response) ---
def evaluate_model(model, dataloader, criterion_eval, args_eval, model_type_eval, data_dir_eval, is_dist_model_eval, is_coral_eval, device_eval):
    model.eval()
    eval_loss = 0.0; eval_loss1 = 0.0; eval_loss2 = 0.0
    eval_total_samples = 0
    utt_true_overall, utt_pred_overall = [], []
    utt_true_textual, utt_pred_textual = [], []
    utt_wavnames_eval = []
    # added for ensembling info
    utt_pred_overall_dist_list = []
    utt_pred_textual_dist_list = []

    pbar_eval = tqdm(dataloader, desc="Evaluating", ncols=100, leave=False)
    for i, data in enumerate(pbar_eval):
        if model_type_eval == 'muq_roberta_transformer_beta_pmf': # Or a more general check
            wavs, mean_scores_q_eval, mean_scores_a_eval, filenames, target_pmf_q_eval, target_pmf_a_eval = data
            labels1_metric_target = mean_scores_q_eval.float().to(device_eval) # For PCC/SRCC/MSE
            labels2_metric_target = mean_scores_a_eval.float().to(device_eval)
            labels1_loss_target = target_pmf_q_eval.float().to(device_eval)   # For KLDiv Loss
            labels2_loss_target = target_pmf_a_eval.float().to(device_eval)
        else:
            wavs, labels1_orig, labels2_orig, filenames = data
            if model_type_eval == 'muq_roberta_annotator_dist':
                labels1_loss_target = convert_annotator_scores_to_distribution(labels1_orig).to(device_eval)
                labels2_loss_target = convert_annotator_scores_to_distribution(labels2_orig).to(device_eval)
                labels1_metric_target = torch.tensor([np.mean(s) for s in labels1_orig], dtype=torch.float32, device=device_eval)
                labels2_metric_target = torch.tensor([np.mean(s) for s in labels2_orig], dtype=torch.float32, device=device_eval)
            else:
                labels1_loss_target = torch.as_tensor(labels1_orig, dtype=torch.float32, device=device_eval)
                labels2_loss_target = torch.as_tensor(labels2_orig, dtype=torch.float32, device=device_eval)
                labels1_metric_target = labels1_loss_target.clone()
                labels2_metric_target = labels2_loss_target.clone()
            
        current_batch_size = wavs.size(0); eval_total_samples += current_batch_size
        
        # Waveform preprocessing for evaluation (ensure consistency with training)
        # The original `mos_muq.py` validation loop had `wavs = wavs.squeeze(1).to(device)`
        # We need to replicate that if models expect squeezed input.
        # Assuming MosDataset provides (B, 1, T) for mono audio.
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs_input = wavs.squeeze(1).to(device_eval)
        elif wavs.ndim == 2: # Already (B, T)
            wavs_input = wavs.to(device_eval)
        else: # (B, C, T) C>1 or other
            wavs_input = wavs.to(device_eval) # Pass as is, model must handle

        texts = get_texts_from_filename(data_dir_eval, filenames)

        

        with torch.no_grad():
            current_pred_overall_dist = None # For current batch
            current_pred_textual_dist = None # For current batch

            if is_coral_eval: # NEW BRANCH for CORAL
                # Model returns: overall_logits, coherence_logits, overall_mos_pred, coherence_mos_pred
                overall_logits_eval, coherence_logits_eval, output1_score_eval, output2_score_eval = model(wavs_input, texts)
                
                # Loss calculation (criterion_eval is coral_loss_function)
                # labels1_metric_target and labels2_metric_target are the raw MOS scores here
                loss1 = criterion_eval(overall_logits_eval, labels1_metric_target, num_ranks=model.num_ranks) # Assumes model has num_ranks attribute
                loss2 = criterion_eval(coherence_logits_eval, labels2_metric_target, num_ranks=model.num_ranks)
                
                pred1_metric, pred2_metric = output1_score_eval, output2_score_eval
                pred1_scalar_metric, pred2_scalar_metric = output1_score_eval, output2_score_eval

            
            elif is_dist_model_eval:
                dist_output1, dist_output2, scalar_score1, scalar_score2 = model(wavs_input, texts)
                pred1_scalar_metric, pred2_scalar_metric = scalar_score1, scalar_score2
                current_pred_overall_dist = dist_output1.detach().cpu().numpy() # Store distribution
                current_pred_textual_dist = dist_output2.detach().cpu().numpy() # Store distribution
                
                if model_type_eval == 'muq_roberta_annotator_dist':
                    # labels1_loss_target IS the target distribution for annotator_dist
                    target1_for_loss = labels1_loss_target 
                    target2_for_loss = labels2_loss_target
                elif model_type_eval == 'muq_roberta_transformer_beta_pmf':
                    # labels1_loss_target IS the target PMF for beta_pmf
                    target1_for_loss = labels1_loss_target
                    target2_for_loss = labels2_loss_target
                else: # Other KLDiv-based _dist models using one_hot or gaussian from scalar labels
                    # Here, labels1_loss_target is still a scalar (mean MOS)
                    if args_eval.dist_prediction_score_style == 'one_hot':
                        target1_for_loss = scores_to_one_hot(labels1_loss_target, args_eval.num_bins, device_eval)
                        target2_for_loss = scores_to_one_hot(labels2_loss_target, args_eval.num_bins, device_eval)
                    elif args_eval.dist_prediction_score_style == 'gaussian':
                        # Ensure scores_to_gaussian_target is correctly imported and used
                        target1_for_loss = scores_to_gaussian_target(labels1_loss_target, args_eval.num_bins, device_eval, sigma=0.25) # Add sigma or make it an arg
                        target2_for_loss = scores_to_gaussian_target(labels2_loss_target, args_eval.num_bins, device_eval, sigma=0.25)
                    else: # Fallback or error if style not recognized for older dist models
                        raise ValueError(f"Unsupported dist_prediction_score_style '{args_eval.dist_prediction_score_style}' for model {model_type_eval}")
                
                # Loss calculation (applies to all is_dist_model_eval types using KLDiv)
                loss1 = criterion_eval(torch.log(dist_output1 + 1e-10), target1_for_loss)
                loss2 = criterion_eval(torch.log(dist_output2 + 1e-10), target2_for_loss)
                

                
                pred1_metric, pred2_metric = scalar_score1, scalar_score2
            else: # Regression
                output1, output2 = model(wavs_input, texts)
                pred1_scalar_metric, pred2_scalar_metric = output1, output2
                l1_loss_shaped = labels1_loss_target.unsqueeze(1) if output1.ndim == 2 and output1.size(1) == 1 and labels1_loss_target.ndim == 1 else labels1_loss_target
                l2_loss_shaped = labels2_loss_target.unsqueeze(1) if output2.ndim == 2 and output2.size(1) == 1 and labels2_loss_target.ndim == 1 else labels2_loss_target
                loss1 = criterion_eval(output1, l1_loss_shaped)
                loss2 = criterion_eval(output2, l2_loss_shaped)
                pred1_metric, pred2_metric = output1, output2
        
        utt_pred_overall.extend(np.array(pred1_metric.detach().cpu()).flatten())
        utt_true_overall.extend(np.array(labels1_metric_target.cpu()).flatten())
        utt_pred_textual.extend(np.array(pred2_metric.detach().cpu()).flatten())
        utt_true_textual.extend(np.array(labels2_metric_target.cpu()).flatten())
        utt_wavnames_eval.extend(filenames)

        # Append distributions batch by batch
        if current_pred_overall_dist is not None:
            utt_pred_overall_dist_list.extend(list(current_pred_overall_dist)) # list of arrays
        else: # If no dist, append None or a placeholder for each sample in batch
            utt_pred_overall_dist_list.extend([None] * current_batch_size)

        if current_pred_textual_dist is not None:
            utt_pred_textual_dist_list.extend(list(current_pred_textual_dist)) # list of arrays
        else:
            utt_pred_textual_dist_list.extend([None] * current_batch_size)

        eval_loss += ((loss1 + loss2) / 2).item() * current_batch_size
        eval_loss1 += loss1.item() * current_batch_size
        eval_loss2 += loss2.item() * current_batch_size
    
    avg_eval_loss = eval_loss / eval_total_samples if eval_total_samples > 0 else 0
    avg_eval_loss1 = eval_loss1 / eval_total_samples if eval_total_samples > 0 else 0
    avg_eval_loss2 = eval_loss2 / eval_total_samples if eval_total_samples > 0 else 0

    mse_o, lcc_o, srcc_o, ktau_o = compute_metrics(utt_true_overall, utt_pred_overall)
    mse_t, lcc_t, srcc_t, ktau_t = compute_metrics(utt_true_textual, utt_pred_textual)
    true_sys_o, true_sys_t, pred_sys_o, pred_sys_t = {}, {}, {}, {}
    for wav, to, po, tt, pt in zip(utt_wavnames_eval, utt_true_overall, utt_pred_overall, utt_true_textual, utt_pred_textual):
        sid = systemID(wav);
        if sid is None: continue
        pred_sys_o.setdefault(sid, []).append(po); true_sys_o.setdefault(sid, []).append(to)
        pred_sys_t.setdefault(sid, []).append(pt); true_sys_t.setdefault(sid, []).append(tt)
    s_true_o, s_pred_o, s_true_t, s_pred_t = [], [], [], []
    if pred_sys_o:
        for sid in sorted(pred_sys_o.keys()): s_true_o.append(np.mean(true_sys_o[sid])); s_pred_o.append(np.mean(pred_sys_o[sid]))
    if pred_sys_t:
        for sid in sorted(pred_sys_t.keys()): s_true_t.append(np.mean(true_sys_t[sid])); s_pred_t.append(np.mean(pred_sys_t[sid]))
    sys_mse_o, sys_lcc_o, sys_srcc_o, sys_ktau_o = compute_metrics(s_true_o, s_pred_o)
    sys_mse_t, sys_lcc_t, sys_srcc_t, sys_ktau_t = compute_metrics(s_true_t, s_pred_t)
    
    results_metrics = {
        "loss": avg_eval_loss, "loss1": avg_eval_loss1, "loss2": avg_eval_loss2,
        "utt_mse_o": mse_o, "utt_lcc_o": lcc_o, "utt_srcc_o": srcc_o, "utt_ktau_o": ktau_o,
        "utt_mse_t": mse_t, "utt_lcc_t": lcc_t, "utt_srcc_t": srcc_t, "utt_ktau_t": ktau_t,
        "sys_mse_o": sys_mse_o, "sys_lcc_o": sys_lcc_o, "sys_srcc_o": sys_srcc_o, "sys_ktau_o": sys_ktau_o,
        "sys_mse_t": sys_mse_t, "sys_lcc_t": sys_lcc_t, "sys_srcc_t": sys_srcc_t, "sys_ktau_t": sys_ktau_t,
    }
    detailed_predictions_per_sample = []
    for i in range(len(utt_wavnames_eval)):
        wav_pred_data = {
            "wavname_with_ext": utt_wavnames_eval[i],
            "overall_score": utt_pred_overall[i],
            "textual_score": utt_pred_textual[i],
            "overall_dist": utt_pred_overall_dist_list[i], # This will be a numpy array or None
            "textual_dist": utt_pred_textual_dist_list[i]  # This will be a numpy array or None
        }
        detailed_predictions_per_sample.append(wav_pred_data)

    predictions_for_file = {
        "wavnames": utt_wavnames_eval,        # List of original filenames (e.g., 'file.wav')
        "overall_scores": utt_pred_overall, # List of predicted overall scores
        "textual_scores": utt_pred_textual  # List of predicted textual scores
    }
    return results_metrics, predictions_for_file, detailed_predictions_per_sample
# --- End of helper function ---


def main() -> None: # Added type hint for clarity
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="../data/MusicEval-phase1", required=False, help='Path of musiceval dataset root')
    parser.add_argument('--expname', type=str, required=False, default='exp1_custom_split', help='ckpt will be saved in track1_ckpt/EXPNAME')
    parser.add_argument('--ablation_model', action='store_true', help='Use the flexible ablation model instead of a specific model_type.')
    parser.add_argument('--audio_encoder_type', type=str, default='transformer', choices=['transformer', 'mamba', 'bilstm'], help='Sequence encoder for the ablation model.')
    parser.add_argument('--pooling_type', type=str, default='attention', choices=['attention', 'mean'], help='Pooling method for the ablation model.')
    # --- NEW ARGS for custom splits ---
    parser.add_argument('--train_list_path', type=str, required=True, help='Path to the training list file.')
    parser.add_argument('--validation_list_path', type=str, required=True, help='Path to the validation list file (used during training epochs).')
    parser.add_argument('--test_list_path', type=str, required=True, help='Path to the final test list file (e.g., original dev_mos_list.txt).')
    # --- End NEW ARGS ---
    parser.add_argument('--model_type', type=str, 
                       choices=['ablation_dist','clap','mulan', 'muq_roberta', 'muq_roberta_cnn', 'muq_roberta_lstm', 
                               'muq_roberta_cnn_lstm', 'muq_roberta_attention', 'muq_roberta_dist',
                               'muq_roberta_lstm_dist',
                               'muq_roberta_annotator_dist', 'muq_roberta_mamba','muq_mulan_roberta_transformer_dist', 
                               'muq_roberta_dual_mamba', 'muq_roberta_dual_mamba_3layer','muq_roberta_dual_mamba_3layer_dist','muq_roberta_finetune',
                               'muq_roberta_transformer_dist','muq_roberta_transformer_scalar','muq_roberta_transformer_dist_coral','muq_roberta_transformer_beta_pmf','muq_roberta_transformer_and_mamba_dist','muq_roberta_transformer_decoupled_dist','muq_roberta_transformer_decoupled_and_lst_dist'],
                       default='mulan', 
                       help='Model type to use')
    parser.add_argument('--num_ranks', type=int, default=5, help='Number of ranks for PMF/Ordinal models (e.g., 5 for MOS 1-5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='Batch size for validation and testing') # Clarified
    parser.add_argument('--num_bins', type=int, default=20, help='Number of bins for distribution prediction')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (AdamW default changed)') # Changed default to match prev suggestion
    parser.add_argument('--loss1_type', type=str, choices=['l1', 'dcq'], default='l1',
                       help='Loss function type for overall quality prediction (loss1): l1 or dcq')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Alpha parameter for Mixup. 0.0 to disable.')
    parser.add_argument('--mixup_type', type=str, choices=['none', 'standard', 'c_mixup'], default='none',
                       help='Type of mixup augmentation to use.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of training epochs.') # Adjusted default
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience.') # Adjusted default
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default="sgd", help='Early stopping patience.')
    parser.add_argument('--dist_prediction_score_style', type=str, choices=['one_hot', 'gaussian','coral'], default="one_hot", help='Early stopping patience.')
    parser.add_argument('--predict_only_ckpt_path', type=str, default=None, 
                        help='Path to a checkpoint for direct prediction. Skips training. Requires --test_list_path.')
    parser.add_argument('--predict_output_filename_base', type=str, default='answer', 
                        help='Base name for the output prediction file if --predict_only_ckpt_path is used (e.g., answer_MODEL).')
    parser.add_argument('--seed', type=int, default=1984, help='Random seed for reproducibility')

    args = parser.parse_args()

    # --- Seeding ---
    SEED = args.seed if hasattr(args, 'seed') else 1984 # Default seed if not provided
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # Potentially slows down but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    DATA_DIR = args.datadir
    MODEL_TYPE = args.model_type
    EXP_NAME = args.expname
    CKPT_DIR = os.path.join('./track1_ckpt', EXP_NAME) # Using ../track1_ckpt consistent with previous suggestions
    
    target_sr = 16000 if args.model_type == "clap" else 24000
    max_audio_seconds = 30 
    
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR, exist_ok=True) # Use makedirs

    # Setup Logging (log file name and tensorboard dir updated)
    log_file_name = f'train_eval_custom_split.log'
    if args.predict_only_ckpt_path:
        log_file_name = f'predict_only_{os.path.splitext(os.path.basename(args.predict_only_ckpt_path))[0]}.log'
    
    logging.basicConfig(
        filename=os.path.join(CKPT_DIR, log_file_name), # Log will be in CKPT_DIR
        filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

    logging.info(f"Running script: {os.path.basename(__file__)}")
    logging.info(f"Script arguments: {args}")
    logging.info(f"DEVICE: {device}")
    logging.info(f"Using target_sr={target_sr} Hz and max_duration={max_audio_seconds}s for datasets.")
    logging.info(f"Using seed: {SEED}")


    is_coral_model = (args.dist_prediction_score_style == 'coral' and 
                  args.model_type == 'muq_roberta_transformer_dist_coral') # More specific
    
    if is_coral_model:
        is_distribution_model = False # CORAL is not a distribution model in the old sense
    else:
        is_distribution_model = args.model_type.lower().endswith("_dist")
    
    if MODEL_TYPE == 'clap':
        is_distribution_model = False
        is_coral_model = False
    else:
        muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()
        roberta = RobertaModel.from_pretrained('roberta-base').to(device).eval()
        
        if MODEL_TYPE == 'muq_roberta_transformer_decoupled_and_lst_dist':
            net = MuQRoBERTaTransformerLSTMHeadCrossAttnDecoupledPredictor(muq, roberta, num_bins=args.num_bins).to(device)
            is_distribution_model = True
        elif MODEL_TYPE == 'muq_roberta_transformer_dist' or MODEL_TYPE == 'muq_roberta_transformer_dist_coral': 
            if args.dist_prediction_score_style == 'coral':
                NUM_RANKS_FOR_CORAL = 5 # Or args.num_ranks
                net = MuQRoBERTaTransformerDistributionPredictorCORALPredictor(muq, roberta, num_ranks=NUM_RANKS_FOR_CORAL).to(device)
                is_coral_model = True # Add a flag for clarity if needed
            else:
                net = MuQRoBERTaTransformerDistributionPredictor(muq, roberta, num_bins=args.num_bins).to(device)
        
        elif MODEL_TYPE == 'muq_roberta_transformer_scalar':
            net = MuQRoBERTaTransformerScalarPredictor(muq, roberta).to(device)
        
        elif MODEL_TYPE == 'muq_roberta_transformer_decoupled_dist':
            net = MuQRoBERTaTransformerDecoupledDist(muq, roberta, num_bins=args.num_bins).to(device)
        elif MODEL_TYPE == "muq_mulan_roberta_transformer_dist":
            net = MuQMulanRoBERTaTransformerDistributionPredictor(model_mulan_instance, roberta, num_bins=args.num_bins).to(device)

        else: raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # *** BRANCH FOR PREDICT_ONLY MODE ***
    if args.predict_only_ckpt_path:
        logging.info(f"--- PREDICTION ONLY MODE ACTIVATED ---")
        logging.info(f"Loading checkpoint from: {args.predict_only_ckpt_path}")
        if not os.path.exists(args.predict_only_ckpt_path):
            logging.error(f"FATAL: Checkpoint file not found: {args.predict_only_ckpt_path}")
            sys.exit(1)
        
        try:
            ckpt = torch.load(args.predict_only_ckpt_path, map_location=device)
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt: net.load_state_dict(ckpt['state_dict'])
                elif 'model_state_dict' in ckpt: net.load_state_dict(ckpt['model_state_dict'])
                else: net.load_state_dict(ckpt)
            else: net.load_state_dict(ckpt)
            logging.info("Checkpoint loaded successfully into model for prediction.")
        except Exception as e:
            logging.error(f"FATAL: Failed to load checkpoint {args.predict_only_ckpt_path}: {e}")
            sys.exit(1)
        
        net.eval() # Ensure model is in evaluation mode

        # Initialize test dataloader for prediction
        wavdir_pred = os.path.join(args.datadir, 'wav')
        if not args.test_list_path or not os.path.exists(args.test_list_path):
            logging.error(f"FATAL: --test_list_path ('{args.test_list_path}') is required and was not found for --predict_only mode.")
            sys.exit(1)
        
        logging.info(f"Preparing test dataset from: {args.test_list_path}")
        is_official_eval_set = os.path.basename(args.test_list_path) == "eval_list.txt"
        current_wavdir_pred = wavdir_pred # Default wavdir
        if is_official_eval_set:
            # Path: /dataset/speech_and_audio_datasets/MusicEval-phase1/audiomos2025-track1-eval-phase/DATA/wav
            current_wavdir_pred = os.path.join(args.datadir, "audiomos2025-track1-eval-phase", "DATA", "wav")
            logging.info(f"Using special WAV directory for eval_list: {current_wavdir_pred}")

        if args.model_type == 'muq_roberta_transformer_beta_pmf': # Use MODEL_TYPE not args.model_type for consistency if changed
            testset_pred =  PersonMosDataset(current_wavdir_pred, args.test_list_path, 
                                                target_sr=target_sr, max_duration_seconds=max_audio_seconds,
                                                num_ranks=args.num_ranks)
        else:
            testset_pred = MosDataset(wavdir=current_wavdir_pred, mos_list=args.test_list_path, target_sr=target_sr, max_duration_seconds=max_audio_seconds, is_eval_mode=is_official_eval_set)
        
        testloader_pred = DataLoader(testset_pred, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, collate_fn=testset_pred.collate_fn, pin_memory=True)
        logging.info(f"Test dataloader created with {len(testset_pred)} samples.")

        # Determine criterion for evaluate_model (loss value itself isn't primary output here but evaluate_model needs it)
        if is_coral_model: criterion_pred = coral_loss_function
        elif is_distribution_model: criterion_pred = nn.KLDivLoss(reduction='batchmean')
        else: criterion_pred = nn.L1Loss()

        logging.info(f"Evaluating model from {args.predict_only_ckpt_path} on test set: {args.test_list_path}")
        # Note: is_distribution_model and is_coral_model are already correctly set globally
        test_metrics_direct, test_predictions_direct, detailed_predictions_list = evaluate_model(
            net, testloader_pred, criterion_pred, args, MODEL_TYPE, args.datadir, 
            is_distribution_model, is_coral_model, device
        )
        
        logging.info(f"TEST SET METRICS (Direct Prediction from {os.path.basename(args.predict_only_ckpt_path)}):")
        for key, value in test_metrics_direct.items(): 
            logging.info(f"  {key.replace('_o','_overall').replace('_t','_textual')}: {value:.4f}")

        output_filename = f"{args.predict_output_filename_base}_{os.path.splitext(os.path.basename(args.predict_only_ckpt_path))[0]}.txt"
        answer_file_path_direct = os.path.join(CKPT_DIR, output_filename) # Save in experiment directory
        
        generate_answer_file(
            test_predictions_direct["wavnames"],
            test_predictions_direct["overall_scores"],
            test_predictions_direct["textual_scores"],
            answer_file_path_direct
        )
         # --- NEW: Save detailed predictions for ensembling ---
        # Create a dictionary mapping wav_id_no_ext to prediction data
        predictions_to_save = {}
        for pred_data in detailed_predictions_list:
            wav_filename_ext = pred_data["wavname_with_ext"]
            if wav_filename_ext.lower().endswith('.wav'):
                wav_id_no_ext = wav_filename_ext[:-4]
            else:
                wav_id_no_ext = wav_filename_ext # Should ideally not happen
            
            predictions_to_save[wav_id_no_ext] = {
                'overall_score': float(pred_data['overall_score']), # Ensure basic Python float
                'textual_score': float(pred_data['textual_score']), # Ensure basic Python float
                'overall_dist': pred_data['overall_dist'], # numpy array or None
                'textual_dist': pred_data['textual_dist'],  # numpy array or None
                'wavname_with_ext': wav_filename_ext # Keep original for reference
            }
        
        
        # Define a structured output filename for these detailed predictions
        # Example: detailed_preds_MODEL-NAME_DATASET-NAME.pt
        dataset_name_indicator = os.path.splitext(os.path.basename(args.test_list_path))[0] # e.g., 'dev_mos_list' or 'test_mos_list'
        ckpt_basename = os.path.splitext(os.path.basename(args.predict_only_ckpt_path))[0]
        # Store these detailed predictions in the experiment directory (CKPT_DIR) or a dedicated 'predictions' folder
        predictions_output_dir = os.path.join(CKPT_DIR, "model_predictions_for_ensemble")
        os.makedirs(predictions_output_dir, exist_ok=True)
        
        detailed_pred_filename = f"detailed_preds_{ckpt_basename}_{dataset_name_indicator}.pt"
        detailed_pred_filepath = os.path.join(predictions_output_dir, detailed_pred_filename)
        try:
            torch.save(predictions_to_save, detailed_pred_filepath)
            logging.info(f"Detailed predictions for ensembling saved to: {detailed_pred_filepath}")
        except Exception as e:
            logging.error(f"Failed to save detailed predictions: {e}")
        
        logging.info(f"Prediction finished. Answer file saved to {answer_file_path_direct}")
        sys.exit(0) # Successfully exit after prediction



    writer = SummaryWriter(log_dir=os.path.join(CKPT_DIR, 'runs_eval_custom_split'))   
    
    wavdir = os.path.join(DATA_DIR, 'wav')
    # Train and validation lists are now from args
    trainlist_path_from_arg = args.train_list_path
    validlist_path_from_arg = args.validation_list_path
    testlist_path_from_arg = args.test_list_path # For final testing
    

    if MODEL_TYPE == 'muq_roberta_transformer_beta_pmf':
        trainset = PersonMosDataset(wavdir, trainlist_path_from_arg, 
                                target_sr=target_sr, max_duration_seconds=max_audio_seconds,
                                num_ranks=args.num_ranks)
        validset = PersonMosDataset(wavdir, validlist_path_from_arg, 
                                    target_sr=target_sr, max_duration_seconds=max_audio_seconds,
                                    num_ranks=args.num_ranks)
        testset = PersonMosDataset(wavdir, testlist_path_from_arg, 
                                target_sr=target_sr, max_duration_seconds=max_audio_seconds,
                                num_ranks=args.num_ranks)
    else: # General MosDataset
        trainset = MosDataset(wavdir=wavdir, mos_list=trainlist_path_from_arg, target_sr=target_sr, max_duration_seconds=max_audio_seconds)
        validset = MosDataset(wavdir=wavdir, mos_list=validlist_path_from_arg, target_sr=target_sr, max_duration_seconds=max_audio_seconds)
        testset = MosDataset(wavdir=wavdir, mos_list=testlist_path_from_arg, target_sr=target_sr, max_duration_seconds=max_audio_seconds)
        
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=trainset.collate_fn, pin_memory=True) # Reduced num_workers
    validloader = DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, collate_fn=validset.collate_fn, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, collate_fn=testset.collate_fn, pin_memory=True)


    logging.info(f"Training {MODEL_TYPE} model. Is CORAL: {is_coral_model}. Is Distribution (KLDiv): {is_distribution_model}")

    if is_distribution_model: 
        criterion = nn.KLDivLoss(reduction='batchmean')
    else: 
        criterion = nn.L1Loss()
    
    if is_coral_model:
        criterion = coral_loss_function # The function itself

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else: # Default to AdamW
        optimizer = optim.AdamW(net.parameters(), lr=args.lr) 

    # Checkpointing state (paths use CKPT_DIR from args.expname)
    # These variables will track best performance on the VALIDATION set (from --validation_list_path)
    BEST_VAL_LOSS = float('inf')
    BEST_UTT_SRCC_OVERALL = -float('inf') 
    BEST_COMBINED_SCORE = -float('inf') 
    BEST_VAL_LOSS_EPOCH, BEST_UTT_SRCC_OVERALL_EPOCH, BEST_COMBINED_EPOCH = 0,0,0
    PATH_BEST_LOSS = os.path.join(CKPT_DIR, 'ckpt_best_val_loss.pth') # Suffix indicates it's from validation set
    PATH_BEST_SRCC = os.path.join(CKPT_DIR, 'ckpt_best_val_srcc.pth')
    PATH_BEST_COMBINED = os.path.join(CKPT_DIR, 'ckpt_best_val_combined.pth')
    patience_counter = 0   

    # --- Training Loop (Identical to your mos_muq.py) ---
    # This loop uses `trainloader` for training and `validloader` for per-epoch validation.
    # Checkpointing is based on `validloader` performance.
    # The entire training and validation loop from your `mos_muq.py` goes here.
    # For brevity, I'm showing the structure. Ensure you copy the exact, working loop.
    import pdb
    total_trainable_params = 0
    print("\n--- Trainable Parameters Calculation ---")

    for param in net.muq.parameters():
        param.requires_grad = False
    for param in net.roberta.parameters():
        param.requires_grad = False

    # Iterate through all parameters of the model
    for name, param in net.named_parameters():
        if param.requires_grad:
            num_params = param.numel()  # Get the number of elements in the tensor
            total_trainable_params += num_params
            # Optional: Print the name and size of each trainable layer
            # print(f"Layer: {name}, Parameters: {num_params}")

    print("-" * 30)
    print(f"Total number of trainable parameters: {total_trainable_params}")
    # For better readability, format with commas
    print(f"Total number of trainable parameters: {total_trainable_params:,}")
    print("-" * 30)

    # Optional: You can also compute total (trainable + frozen) parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters (trainable + frozen): {total_params:,}")

    # Define a realistic batch for inference
    batch_size = 12  # A typical inference batch size
    max_audio_seq_len = 3000 # Example: MuQ might output ~1000 frames for a 10s clip
    max_text_seq_len = 128   # From your tokenizer max_length

    # Create dummy input tensors
    dummy_wavs = torch.randn(batch_size, 16000 * 10).to(device) # ~10 seconds of audio at 16kHz
    dummy_texts = ["A test sentence for evaluation."] * batch_size

    # --- Memory for Model Weights ---
    torch.cuda.synchronize() # Wait for model to be fully moved
    model_mem_bytes = torch.cuda.memory_allocated()
    print(f"Memory for Model Weights: {model_mem_bytes / 1e9:.3f} GB")

    # --- Memory for a Forward Pass ---
    # Reset peak memory stats before the forward pass
    torch.cuda.reset_peak_memory_stats(device)
    peak_mem_before_forward = torch.cuda.max_memory_allocated()

    print("\nRunning a forward pass to measure peak activation memory...")
    # Wrap in torch.no_grad() for pure inference measurement
    with torch.no_grad():
        # Your model's forward pass
        _ = net(dummy_wavs, dummy_texts)

    torch.cuda.synchronize() # Wait for all GPU operations to complete

    # Get the peak memory after the forward pass
    peak_mem_after_forward = torch.cuda.max_memory_allocated()

    # The dynamic memory is the increase from the baseline (model weights)
    dynamic_activation_mem_bytes = peak_mem_after_forward - model_mem_bytes

    print("-" * 40)
    print("--- RUNTIME MEMORY FOOTPRINT ANALYSIS ---")
    print(f"Static Memory (Model Weights):      {model_mem_bytes / 1e6:,.2f} MB")
    print(f"Dynamic Memory (Peak Activations):  {dynamic_activation_mem_bytes / 1e6:,.2f} MB")
    print(f"Total Peak PyTorch Memory Usage:    {peak_mem_after_forward / 1e6:,.2f} MB")
    print(f"                                    ({peak_mem_after_forward / 1e9:.3f} GB)")
    print("-" * 40)
    print(f"Configuration: batch_size={batch_size}, audio_seq_len=~{max_audio_seq_len}, text_seq_len={max_text_seq_len}")



    logging.info("Starting training loop...")
    for epoch in range(1, args.num_epochs + 1):
        net.train()

        # --- START OF COPIED/ADAPTED TRAINING EPOCH LOGIC ---
        train_epoch_loss, train_epoch_loss1, train_epoch_loss2 = 0.0, 0.0, 0.0
        train_total_samples = 0
        
            
        all_train_labels1, all_train_preds1, all_train_labels2, all_train_preds2 = [], [], [], []
        pbar_train = tqdm(trainloader, desc=f"Epoch {epoch} Training", ncols=100, leave=False)
        for i, data in enumerate(pbar_train):
            if MODEL_TYPE == 'muq_roberta_transformer_beta_pmf':
                wavs, mean_scores_q_orig, mean_scores_a_orig, filenames, target_pmf_q, target_pmf_a = data
                labels1_orig = mean_scores_q_orig.float().to(device) # For MSE/SRCC metrics
                labels2_orig = mean_scores_a_orig.float().to(device) # For MSE/SRCC metrics
                target_pmf_q = target_pmf_q.float().to(device) # Target for KLDiv loss
                target_pmf_a = target_pmf_a.float().to(device) # Target for KLDiv loss
            else: 
                wavs, labels1_orig, labels2_orig, filenames = data 
            current_batch_size = wavs.size(0); train_total_samples += current_batch_size
            
            wavs_processed = wavs.to(device) # Keep original wavs for mixup if needed
            # The original script did wavs.squeeze(1).to(device) at validation.
            # For training, MuQ models might prefer (B, T) or (B, 1, T).
            # Assuming MosDataset gives (B, 1, T) for mono.
            if wavs_processed.ndim == 3 and wavs_processed.size(1) == 1:
                 current_wavs_for_model = wavs_processed.squeeze(1)
            else: # (B,T) or (B,C,T) C>1 -> model must handle
                 current_wavs_for_model = wavs_processed


            
            labels1 = torch.as_tensor(labels1_orig, dtype=torch.float32, device=device)
            labels2 = torch.as_tensor(labels2_orig, dtype=torch.float32, device=device)
            texts = get_texts_from_filename(DATA_DIR, filenames)
            optimizer.zero_grad()
            
            # Handle current_wavs for model input vs. current_wavs_for_mixup (original shape)
            current_wavs_for_mixup = wavs.to(device) # Use original shape for mixup if it expects (B,C,T)
            input_to_model = current_wavs_for_model # Default input to model
            lam = 1.0

            if net.training:
                if args.mixup_type == 'standard' and args.mixup_alpha > 0.0:
                    mixed_wavs, l1_a, l1_b, l2_a, l2_b, lam = mixup_data(current_wavs_for_mixup, labels1, labels2, args.mixup_alpha, device=device)
                    if mixed_wavs.ndim == 3 and mixed_wavs.size(1) == 1: input_to_model = mixed_wavs.squeeze(1)
                    else: input_to_model = mixed_wavs
                elif args.mixup_type == 'c_mixup' and args.mixup_alpha > 0.0:
                    mixed_wavs, l1_a, l1_b, l2_a, l2_b, lam = c_mixup_data_paper(current_wavs_for_mixup, labels1, labels2, args.mixup_alpha, device=device)
                    if mixed_wavs.ndim == 3 and mixed_wavs.size(1) == 1: input_to_model = mixed_wavs.squeeze(1)
                    else: input_to_model = mixed_wavs
            
            # --- FORWARD & LOSS (Copied from your mos_muq.py, using `input_to_model`) ---
            if is_coral_model: # <<<< NEW Specific branch for CORAL
                # Model returns: overall_logits, coherence_logits, overall_mos_pred, coherence_mos_pred
                overall_logits, coherence_logits, overall_score, coherence_score = net(input_to_model, texts)
                
                # For loss calculation, we use overall_logits and coherence_logits with raw MOS scores (labels1, labels2)
                # labels1 and labels2 are already torch.float32 tensors on the correct device
                loss1_train = criterion(overall_logits, labels1, num_ranks=net.num_ranks) # Assumes model has num_ranks
                loss2_train = criterion(coherence_logits, labels2, num_ranks=net.num_ranks)

                # For metric tracking (all_train_preds1/2 should be the predicted MOS scores)
                all_train_labels1.extend(labels1.cpu().numpy().flatten()) # True MOS
                all_train_preds1.extend(overall_score.detach().cpu().numpy().flatten()) # Predicted MOS
                all_train_labels2.extend(labels2.cpu().numpy().flatten()) # True MOS
                all_train_preds2.extend(coherence_score.detach().cpu().numpy().flatten()) # Predicted MOS
            elif is_distribution_model:# This branch will be taken by BetaPMF model
                if MODEL_TYPE == 'muq_roberta_transformer_beta_pmf':
                    pred_pmf_q, pred_pmf_a, mos_pred_q_train, mos_pred_a_train = net(input_to_model, texts)
                    # For metric tracking (preds should be scalar MOS predictions)
                    all_train_labels1.extend(labels1_orig.cpu().numpy().flatten())
                    all_train_preds1.extend(mos_pred_q_train.detach().cpu().numpy().flatten())
                    all_train_labels2.extend(labels2_orig.cpu().numpy().flatten())
                    all_train_preds2.extend(mos_pred_a_train.detach().cpu().numpy().flatten())
                    loss1_train = criterion(torch.log(pred_pmf_q + 1e-10), target_pmf_q)
                    loss2_train = criterion(torch.log(pred_pmf_a + 1e-10), target_pmf_a)
                    
                else:
                    overall_dist_pred, coherence_dist_pred, overall_score, coherence_score = net(input_to_model, texts)
                    all_train_labels1.extend(labels1_orig.cpu().numpy() if not isinstance(labels1_orig, list) else [np.mean(s) for s in labels1_orig])
                    all_train_preds1.extend(overall_score.detach().cpu().numpy().flatten())
                    all_train_labels2.extend(labels2_orig.cpu().numpy() if not isinstance(labels2_orig, list) else [np.mean(s) for s in labels2_orig])
                    all_train_preds2.extend(coherence_score.detach().cpu().numpy().flatten())

                if lam < 1.0: 
                    if MODEL_TYPE == 'muq_roberta_annotator_dist': target1_dist_a, target1_dist_b,target2_dist_a, target2_dist_b = l1_a, l1_b, l2_a, l2_b
                    else: 
                        if args.dist_prediction_score_style == 'one_hot':
                            target1_dist_a,target1_dist_b,target2_dist_a,target2_dist_b = scores_to_one_hot(l1_a, args.num_bins, device), scores_to_one_hot(l1_b, args.num_bins, device), scores_to_one_hot(l2_a, args.num_bins, device), scores_to_one_hot(l2_b, args.num_bins, device)
                        elif args.dist_prediction_score_style == 'gaussian':
                            target1_dist_a,target1_dist_b,target2_dist_a,target2_dist_b = scores_to_gaussian_target(l1_a, args.num_bins, device), scores_to_gaussian_target(l1_b, args.num_bins, device), scores_to_gaussian_target(l2_a, args.num_bins, device), scores_to_gaussian_target(l2_b, args.num_bins, device)
                    loss1_train = lam * criterion(torch.log(overall_dist_pred + 1e-10), target1_dist_a) + (1.0 - lam) * criterion(torch.log(overall_dist_pred + 1e-10), target1_dist_b)
                    loss2_train = lam * criterion(torch.log(coherence_dist_pred + 1e-10), target2_dist_a) + (1.0 - lam) * criterion(torch.log(coherence_dist_pred + 1e-10), target2_dist_b)
                else:
                    if MODEL_TYPE != 'muq_roberta_transformer_beta_pmf': 
                        if MODEL_TYPE == 'muq_roberta_annotator_dist': target1_dist, target2_dist = labels1, labels2
                        else:
                            if args.dist_prediction_score_style == 'one_hot':
                                target1_dist, target2_dist = scores_to_one_hot(labels1, args.num_bins, device), scores_to_one_hot(labels2, args.num_bins, device)
                            elif args.dist_prediction_score_style == 'gaussian':
                                target1_dist, target2_dist = scores_to_gaussian_target(labels1, args.num_bins, device), scores_to_gaussian_target(labels2, args.num_bins, device)
                        
                        loss1_train = criterion(torch.log(overall_dist_pred + 1e-10), target1_dist); loss2_train = criterion(torch.log(coherence_dist_pred + 1e-10), target2_dist)
            else:
                output1, output2 = net(input_to_model, texts)
                all_train_labels1.extend(labels1.cpu().numpy().flatten()) # labels1 already tensor
                all_train_preds1.extend(output1.detach().cpu().numpy().flatten())
                all_train_labels2.extend(labels2.cpu().numpy().flatten()) # labels2 already tensor
                all_train_preds2.extend(output2.detach().cpu().numpy().flatten())
                labels1_target_shaped = labels1.unsqueeze(1) if output1.ndim == 2 and output1.size(1) == 1 and labels1.ndim == 1 else labels1
                labels2_target_shaped = labels2.unsqueeze(1) if output2.ndim == 2 and output2.size(1) == 1 and labels2.ndim == 1 else labels2
                if lam < 1.0: # Mixup was applied, shape l1_a, l1_b if necessary
                    l1_a_shaped = l1_a.unsqueeze(1) if output1.ndim == 2 and output1.size(1) == 1 and l1_a.ndim == 1 else l1_a
                    l1_b_shaped = l1_b.unsqueeze(1) if output1.ndim == 2 and output1.size(1) == 1 and l1_b.ndim == 1 else l1_b
                    l2_a_shaped = l2_a.unsqueeze(1) if output2.ndim == 2 and output2.size(1) == 1 and l2_a.ndim == 1 else l2_a
                    l2_b_shaped = l2_b.unsqueeze(1) if output2.ndim == 2 and output2.size(1) == 1 and l2_b.ndim == 1 else l2_b
                    loss1_val_train = lam * criterion(output1, l1_a_shaped) + (1.0 - lam) * criterion(output1, l1_b_shaped)
                    loss2_val_train = lam * criterion(output2, l2_a_shaped) + (1.0 - lam) * criterion(output2, l2_b_shaped)

                loss1_train, loss2_train = loss1_val_train, loss2_val_train
            #pdb.set_trace() # Debugging breakpoint, remove in production
            train_loss_iter = (loss1_train + loss2_train) / 2
            loss1_train.backward(retain_graph=True); loss2_train.backward()
            optimizer.step()
            train_epoch_loss += train_loss_iter.item() * current_batch_size
            pbar_train.set_postfix(loss=train_loss_iter.item())
        
        avg_train_loss = train_epoch_loss / train_total_samples if train_total_samples > 0 else 0
        train_mse1_ep, _, train_srcc1_ep, _ = compute_metrics(np.array(all_train_labels1), np.array(all_train_preds1))
        logging.info(f"Epoch {epoch} Train: Loss={avg_train_loss:.4f}, MSE_O={train_mse1_ep:.4f}, SRCC_O={train_srcc1_ep:.4f}")
        writer.add_scalar('Train/Loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('Train/MSE_overall_epoch', train_mse1_ep, epoch)
        writer.add_scalar('Train/SRCC_overall_epoch', train_srcc1_ep, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # --- END OF COPIED/ADAPTED TRAINING EPOCH LOGIC ---

        # --- PER-EPOCH VALIDATION (using `validloader` from --validation_list_path) ---
        # This uses the evaluate_model helper function.
        val_results, _ ,_ = evaluate_model(net, validloader, criterion, args, MODEL_TYPE, DATA_DIR, is_distribution_model, is_coral_model, device)
        
        avg_val_loss_epoch = val_results["loss"] # Renamed to avoid conflict with BEST_VAL_LOSS
        val_srcc_o_epoch = val_results["utt_srcc_o"]
        val_srcc_system_epoch = val_results['sys_srcc_o']
        

        logging.info(f"Epoch {epoch} Validation: Loss={avg_val_loss_epoch:.4f}, Loss1={val_results['loss1']:.4f}, Loss2={val_results['loss2']:.4f}")
        # Log UTT OVERALL (as before)
        logging.info(f"Epoch {epoch} Validation [UTT OVERALL]: MSE={val_results['utt_mse_o']:.4f}, LCC={val_results['utt_lcc_o']:.4f}, SRCC={val_srcc_o_epoch:.4f}, KTAU={val_results['utt_ktau_o']:.4f}")
        # Log UTT TEXTUAL (NEW)
        logging.info(f"Epoch {epoch} Validation [UTT TEXTUAL]: MSE={val_results['utt_mse_t']:.4f}, LCC={val_results['utt_lcc_t']:.4f}, SRCC={val_results['utt_srcc_t']:.4f}, KTAU={val_results['utt_ktau_t']:.4f}")
        # Log SYS OVERALL (NEW)
        logging.info(f"Epoch {epoch} Validation [SYS OVERALL]: MSE={val_results['sys_mse_o']:.4f}, LCC={val_results['sys_lcc_o']:.4f}, SRCC={val_results['sys_srcc_o']:.4f}, KTAU={val_results['sys_ktau_o']:.4f}")
        # Log SYS TEXTUAL (NEW)
        logging.info(f"Epoch {epoch} Validation [SYS TEXTUAL]: MSE={val_results['sys_mse_t']:.4f}, LCC={val_results['sys_lcc_t']:.4f}, SRCC={val_results['sys_srcc_t']:.4f}, KTAU={val_results['sys_ktau_t']:.4f}")
        for key, value in val_results.items(): # Log all validation metrics to TensorBoard
            writer.add_scalar(f'Validation/{key.replace("_o", "_overall").replace("_t", "_textual")}', value, epoch)

        # Checkpointing & Early Stopping (based on THIS validation set)
        improved = False
        if avg_val_loss_epoch < BEST_VAL_LOSS:
            BEST_VAL_LOSS, BEST_VAL_LOSS_EPOCH, improved = avg_val_loss_epoch, epoch, True
            torch.save(net.state_dict(), PATH_BEST_LOSS)
            logging.info(f"Epoch {epoch}: New best validation loss: {BEST_VAL_LOSS:.4f}. Saved model to {PATH_BEST_LOSS}")
        
        current_srcc_val = val_srcc_system_epoch if not np.isnan(val_srcc_system_epoch) else -1.0
        if current_srcc_val > BEST_UTT_SRCC_OVERALL:
            BEST_UTT_SRCC_OVERALL, BEST_UTT_SRCC_OVERALL_EPOCH, improved = current_srcc_val, epoch, True
            torch.save(net.state_dict(), PATH_BEST_SRCC)
            logging.info(f"Epoch {epoch}: New best validation SYSTEM! SRCC (Overall): {BEST_UTT_SRCC_OVERALL:.4f}. Saved model to {PATH_BEST_SRCC}")

        combined_score_current_val = ((val_srcc_o_epoch if not np.isnan(val_srcc_o_epoch) else -1.0) + val_results['sys_srcc_t']) / 2.0 
        if combined_score_current_val > BEST_COMBINED_SCORE:
            BEST_COMBINED_SCORE, BEST_COMBINED_EPOCH, improved = combined_score_current_val, epoch, True
            torch.save(net.state_dict(), PATH_BEST_COMBINED)
            logging.info(f"Epoch {epoch}: New best validation combined score of music and text system performance: {BEST_COMBINED_SCORE:.4f}. Saved model to {PATH_BEST_COMBINED}")

        if improved: patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"Epoch {epoch}: No improvement on validation set. Patience: {patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            logging.info(f"Early stopping at epoch {epoch} based on validation set performance.")
            break
    # --- End of Training Loop ---

    writer.close()
    logging.info("Finished Training Phase.")
    logging.info(f"Best Validation Loss: {BEST_VAL_LOSS:.4f} at Epoch {BEST_VAL_LOSS_EPOCH}")
    logging.info(f"Best Validation UTT SRCC (Overall): {BEST_UTT_SRCC_OVERALL:.4f} at Epoch {BEST_UTT_SRCC_OVERALL_EPOCH}")
    logging.info(f"Best Validation Combined Score: {BEST_COMBINED_SCORE:.4f} at Epoch {BEST_COMBINED_EPOCH}")

    # --- FINAL EVALUATION ON THE SEPARATE TEST SET (from --test_list_path) ---
    logging.info("\n" + "="*30 + " FINAL EVALUATION ON TEST SET " + "="*30)
    
    # Here, we evaluate all three saved models.
    
    # Test model with best validation loss
    if os.path.exists(PATH_BEST_LOSS):
        logging.info(f"\nLoading best validation loss model (Epoch {BEST_VAL_LOSS_EPOCH}) from {PATH_BEST_LOSS} for final test evaluation...")
        net.load_state_dict(torch.load(PATH_BEST_LOSS, map_location=device))
        test_results_loss_model, test_predictions_loss_model, detailed_info_pred_loss = evaluate_model(net, testloader, criterion, args, MODEL_TYPE, DATA_DIR, is_distribution_model, is_coral_model, device)
        logging.info(f"TEST SET (Model from Best Validation Loss):")
        for key, value in test_results_loss_model.items(): logging.info(f"  {key.replace('_o','_overall').replace('_t','_textual')}: {value:.4f}")
        answer_file_path_loss = os.path.join(CKPT_DIR, f'answer_test_best_loss_epoch{BEST_VAL_LOSS_EPOCH}.txt')
        generate_answer_file(
            test_predictions_loss_model["wavnames"],
            test_predictions_loss_model["overall_scores"],
            test_predictions_loss_model["textual_scores"],
            answer_file_path_loss
        )

    else:
        logging.warning(f"Checkpoint {PATH_BEST_LOSS} not found. Skipping test evaluation for best loss model.")

    # Test model with best validation SRCC
    if os.path.exists(PATH_BEST_SRCC):
        logging.info(f"\nLoading best validation SRCC model (Epoch {BEST_UTT_SRCC_OVERALL_EPOCH}) from {PATH_BEST_SRCC} for final test evaluation...")
        net.load_state_dict(torch.load(PATH_BEST_SRCC, map_location=device))
        test_results_srcc_model, test_predictions_srcc_model, detailed_test_predictions_srcc_model = evaluate_model(net, testloader, criterion, args, MODEL_TYPE, DATA_DIR, is_distribution_model, is_coral_model, device)
        logging.info(f"TEST SET (Model from Best Validation SRCC):")
        for key, value in test_results_srcc_model.items(): logging.info(f"  {key.replace('_o','_overall').replace('_t','_textual')}: {value:.4f}")
        answer_file_path_srcc = os.path.join(CKPT_DIR, f'answer_test_best_sys_srcc_overall_epoch{BEST_UTT_SRCC_OVERALL_EPOCH}.txt')
        generate_answer_file(
            test_predictions_srcc_model["wavnames"],
            test_predictions_srcc_model["overall_scores"],
            test_predictions_srcc_model["textual_scores"],
            answer_file_path_srcc
        )
    else:
        logging.warning(f"Checkpoint {PATH_BEST_SRCC} not found. Skipping test evaluation for best SRCC model.")

    # Test model with best validation combined score
    if os.path.exists(PATH_BEST_COMBINED):
        logging.info(f"\nLoading best validation combined model (Epoch {BEST_COMBINED_EPOCH}) from {PATH_BEST_COMBINED} for final test evaluation...")
        net.load_state_dict(torch.load(PATH_BEST_COMBINED, map_location=device))
        test_results_combined_model, test_predictions_combined_model, detailed_info_pred = evaluate_model(net, testloader, criterion, args, MODEL_TYPE, DATA_DIR, is_distribution_model, is_coral_model, device)
        logging.info(f"TEST SET (Model from Best Validation Combined Score):")
        for key, value in test_results_combined_model.items(): logging.info(f"  {key.replace('_o','_overall').replace('_t','_textual')}: {value:.4f}")
        answer_file_path_combined = os.path.join(CKPT_DIR, f'answer_test_best_combined_sys_score_epoch{BEST_COMBINED_EPOCH}.txt')
        generate_answer_file(
            test_predictions_combined_model["wavnames"],
            test_predictions_combined_model["overall_scores"],
            test_predictions_combined_model["textual_scores"],
            answer_file_path_combined
        )

    else:
        logging.warning(f"Checkpoint {PATH_BEST_COMBINED} not found. Skipping test evaluation for best combined model.")

    logging.info("="*70 + " SCRIPT FINISHED " + "="*70)


if __name__ == '__main__':
    main()