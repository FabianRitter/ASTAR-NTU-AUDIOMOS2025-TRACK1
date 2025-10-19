#!/bin/bash

# Script to reproduce the experiments for the ASTAR-NTU AudioMOS Challenge 2025 Track 1 submission.

# IMPORTANT: Set the DATA_DIR variable to the root of your MusicEval-phase1 dataset.
DATA_DIR="/dataset/speech_and_audio_datasets/MusicEval-phase1"
TRAIN_LIST="${DATA_DIR}/sets/train_mos_list.txt"
DEV_LIST="${DATA_DIR}/sets/dev_mos_list.txt"


LR="5e-5"
BATCH_SIZE=32
VALID_BATCH_SIZE=16
OPTIMIZER="adamw"
MIXUP_TYPE="none"
NUM_BINS=20

# Create directory for logs if it doesn't exist
mkdir -p logs

echo "Starting experiments. Ensure DATA_DIR is set correctly: ${DATA_DIR}"

# Function to run a training job
run_training() {
    local EXP_NAME=$1
    local MODEL_TYPE=$2
    local STYLE=$3
    local SEED=$4
    
    echo "Launching ${EXP_NAME}..."
    
    # Construct the command using the refactored train.py
    CMD="nohup python train.py \
        --expname ${EXP_NAME} \
        --model_type ${MODEL_TYPE} \
        --train_list_path ${TRAIN_LIST} \
        --validation_list_path ${DEV_LIST} \
        --test_list_path ${DEV_LIST} \
        --datadir ${DATA_DIR} \
        --batch_size ${BATCH_SIZE} \
        --valid_batch_size ${VALID_BATCH_SIZE} \
        --lr ${LR} \
        --optimizer ${OPTIMIZER} \
        --dist_prediction_score_style ${STYLE} \
        --mixup_type ${MIXUP_TYPE} \
        --num_bins ${NUM_BINS}"

    # Add seed if provided
    if [ ! -z "${SEED}" ]; then
        CMD="${CMD} --seed ${SEED}"
    fi
    
    # Redirect output to log file and run in background
    CMD="${CMD} > logs/${EXP_NAME}.log 2>&1 &"
    
    # Execute the command (use eval to handle redirection correctly)
    eval $CMD
}

# --- Ensemble Experiments ---

# 1. Main Model (Gaussian Distribution) - Default Seed (e.g., 1984 if not specified)
run_training "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias" "muq_roberta_transformer_dist" "gaussian" ""

# 2. Main Model (Gaussian) - Seed 1990
run_training "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990" "muq_roberta_transformer_dist" "gaussian" "1990"

# 3. Main Model (Gaussian) - Seed 1910
run_training "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1910" "muq_roberta_transformer_dist" "gaussian" "1910"

# 4. Main Model (Gaussian) - Seed 1510
# Note: Original expname was 'getparams', renamed here for consistency.
run_training "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510" "muq_roberta_transformer_dist" "gaussian" "1510"

# 5. Main Model (One-Hot Distribution)
run_training "transformer_cross_attention_standard_with_whole_data_again" "muq_roberta_transformer_dist" "one_hot" ""

# 6. CORAL Model (Ordinal Regression) - Seed 1510
run_training "transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata_seed_1510" "muq_roberta_transformer_dist_coral" "coral" "1510"

# 7. Decoupled Input with LSTM Head (Gaussian)
run_training "transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_LSTM_SCORE_inner_dev_case_with_gaussian_bias_whole" "muq_roberta_transformer_decoupled_and_lst_dist" "gaussian" ""

# 8. Decoupled Input without LSTM Head (Gaussian)
run_training "transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole" "muq_roberta_transformer_decoupled_dist" "gaussian" ""

echo "All 8 experiments launched in the background. Check the logs/ directory for output."