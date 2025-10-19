#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="mos_muq_inner_dev.py"
BASE_CKPT_DIR="./track1_ckpt" # Base directory for experiment checkpoints

# Common dataset paths and arguments (modify if your paths are different)
DATADIR="/dataset/speech_and_audio_datasets/MusicEval-phase1"
TRAIN_LIST_PATH="/dataset/speech_and_audio_datasets/MusicEval-phase1/sets/train_mos_list.txt"
# This is the validation set used for training the meta-model in the ensemble script
VAL_SET_FOR_ENSEMBLE_META_TRAINING="/dataset/speech_and_audio_datasets/MusicEval-phase1/sets/dev_mos_list.txt"
# This is the final test set for which you want to generate submission predictions
FINAL_TEST_SET_FOR_SUBMISSION="/dataset/speech_and_audio_datasets/MusicEval-phase1/sets/test_mos_list.txt"
# OFFICIAL EVAL set (unlabeled, for final Codabench submission)
OFFICIAL_EVAL_LIST_DIR="$DATADIR/audiomos2025-track1-eval-phase/DATA"
OFFICIAL_EVAL_LIST="$OFFICIAL_EVAL_LIST_DIR/sets/eval_list.txt"

VALID_BATCH_SIZE=16
COMMON_ARGS="--datadir $DATADIR --train_list_path $TRAIN_LIST_PATH --validation_list_path $VAL_SET_FOR_ENSEMBLE_META_TRAINING --valid_batch_size $VALID_BATCH_SIZE"
#COMMON_ARGS="--datadir $DATADIR --train_list_path $TRAIN_LIST_PATH --validation_list_path $DEV_MOS_LIST --valid_batch_size $VALID_BATCH_SIZE"

declare -a models=(
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias/ckpt_best_val_srcc.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990/ckpt_best_val_combined.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990/ckpt_best_val_srcc.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1990;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960/ckpt_best_val_combined.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960/ckpt_best_val_srcc.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1960;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1910/ckpt_best_val_combined.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1910;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510/ckpt_best_val_combined.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510;muq_roberta_transformer_dist;gaussian;20"
    "transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510/ckpt_best_val_srcc.pth;transformer_cross_attention_standard_WHOLE_DATA_case_with_gaussian_bias_seed1510;muq_roberta_transformer_dist;gaussian;20"

    "transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/ckpt_best_val_combined.pth;transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole;muq_roberta_transformer_decoupled_dist;gaussian;20"
    "transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/ckpt_best_val_srcc.pth;transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_NOLSTM_SCORE_inner_dev_case_with_gaussian_bias_whole;muq_roberta_transformer_decoupled_dist;gaussian;20"

    "transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_LSTM_SCORE_inner_dev_case_with_gaussian_bias_whole/ckpt_best_val_combined.pth;transformer_cross_attention_with_DECOUPLED_INPUT_SCORE_AND_LSTM_SCORE_inner_dev_case_with_gaussian_bias_whole;muq_roberta_transformer_decoupled_and_lst_dist;gaussian;20"

    # Scalar models (CORAL)
    "transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata/ckpt_best_val_combined.pth;transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata;muq_roberta_transformer_dist_coral;coral;5"
    "transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata/ckpt_best_val_srcc.pth;transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata;muq_roberta_transformer_dist_coral;coral;5"
    # Note: The user listed the next one twice. Assuming it's a single distinct model checkpoint.
    "transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata_seed_1510/ckpt_best_val_combined.pth;transformer_cross_attention_standard_but_with_ordinal_regression_with_monotonicity_wholedata_seed_1510;muq_roberta_transformer_dist_coral;coral;5"
)

# --- Main Loop ---
for model_config in "${models[@]}"; do
    IFS=';' read -r REL_CKPT_PATH EXP_NAME MODEL_TYPE DIST_STYLE NUM_BINS_OR_RANKS <<< "$model_config"

    FULL_CKPT_PATH="$BASE_CKPT_DIR/$REL_CKPT_PATH"
    CKPT_BASENAME=$(basename "$FULL_CKPT_PATH" .pth) # e.g., ckpt_best_val_srcc

    echo "====================================================================================="
    echo "Processing checkpoint: $FULL_CKPT_PATH"
    echo "Experiment Name: $EXP_NAME"
    echo "Model Type: $MODEL_TYPE"
    echo "Distribution Style: $DIST_STYLE"
    echo "Num Bins/Ranks: $NUM_BINS_OR_RANKS"

.
    MODEL_SPECIFIC_ARGS="--num_bins $NUM_BINS_OR_RANKS --dist_prediction_score_style $DIST_STYLE"
    if [ "$DIST_STYLE" == "coral" ]; then
        MODEL_SPECIFIC_ARGS="$MODEL_SPECIFIC_ARGS --num_ranks $NUM_BINS_OR_RANKS" # Explicitly set num_ranks for coral if needed by script
    fi


    # Generate predictions for the VAL_SET_FOR_ENSEMBLE_META_TRAINING
    # These will be used to train the meta-model
    echo "-------------------------------------------------------------------------------------"
    echo "Generating predictions on: $VAL_SET_FOR_ENSEMBLE_META_TRAINING (for meta-model training)"
    # The --predict_output_filename_base is for the human-readable .txt answer file.
    # The detailed .pt file will be named uniquely inside the python script based on ckpt and dataset list.
    PREDICT_OUT_BASE_VAL="answer_${CKPT_BASENAME}_on_val_set_for_ensemble"

    python $PYTHON_SCRIPT \
        --predict_only_ckpt_path "$FULL_CKPT_PATH" \
        --expname "$EXP_NAME" \
        --model_type "$MODEL_TYPE" \
        $COMMON_ARGS \
        $MODEL_SPECIFIC_ARGS \
        --test_list_path "$VAL_SET_FOR_ENSEMBLE_META_TRAINING" \
        --predict_output_filename_base "$PREDICT_OUT_BASE_VAL"
    
    echo "Finished VAL SET for $FULL_CKPT_PATH"
    echo ""

    # Generate predictions for the FINAL_TEST_SET_FOR_SUBMISSION
    # These will be fed into the trained meta-model for the final submission
    echo "-------------------------------------------------------------------------------------"
    echo "Generating predictions on: $FINAL_TEST_SET_FOR_SUBMISSION (for final submission)"
    # --- Generate predictions for OFFICIAL EVAL SET (for final submission) ---
    if [ -f "$OFFICIAL_EVAL_LIST" ]; then
        echo "-------------------------------------------------------------------------------------"
        echo "Generating predictions on OFFICIAL EVAL SET: $OFFICIAL_EVAL_LIST (for final submission)"
        PREDICT_OUT_BASE_EVAL="preds_on_eval_for_${CKPT_BASENAME}"

        python $PYTHON_SCRIPT \
            --predict_only_ckpt_path "$FULL_CKPT_PATH" \
            --expname "$EXP_NAME" \
            --model_type "$MODEL_TYPE" \
            $COMMON_ARGS \
            $MODEL_SPECIFIC_ARGS \
            --test_list_path "$OFFICIAL_EVAL_LIST" \
            --predict_output_filename_base "$PREDICT_OUT_BASE_EVAL"
        
        echo "Finished OFFICIAL EVAL SET for $FULL_CKPT_PATH"
        echo "Detailed predictions: $BASE_CKPT_DIR/$EXP_NAME/model_predictions_for_ensemble/detailed_preds_${CKPT_BASENAME}_eval_list.pt"
    else
        echo "WARNING: Official eval list not found at $OFFICIAL_EVAL_LIST. Skipping prediction generation for it."
    fi
    PREDICT_OUT_BASE_TEST="answer_${CKPT_BASENAME}_on_final_test_set"

    # python $PYTHON_SCRIPT \
    #     --predict_only_ckpt_path "$FULL_CKPT_PATH" \
    #     --expname "$EXP_NAME" \
    #     --model_type "$MODEL_TYPE" \
    #     $COMMON_ARGS \
    #     $MODEL_SPECIFIC_ARGS \
    #     --test_list_path "$FINAL_TEST_SET_FOR_SUBMISSION" \
    #     --predict_output_filename_base "$PREDICT_OUT_BASE_TEST"

    # echo "Finished FINAL TEST SET for $FULL_CKPT_PATH"
done

echo "====================================================================================="
echo "All prediction generation runs complete."
echo "Detailed prediction files (.pt) should be saved in respective experiment directories:"
echo "e.g., $BASE_CKPT_DIR/YOUR_EXP_NAME/model_predictions_for_ensemble/detailed_preds_YOUR_CKPT_BASENAME_DATASETLISTNAME.pt"
echo "Please verify the paths and update MODEL_INFO in ensemble_predictions.py script."
echo "====================================================================================="

