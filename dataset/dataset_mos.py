# utils.py (or datasets.py)

import os
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from torch.utils.data.dataset import Dataset
from collections import Counter
import numpy as np
import os

import logging

logger = logging.getLogger(__name__)

# --- Dataset Definition ---
class PersonMosDataset(Dataset):
    def __init__(self, wavdir, person_mos_list_file,  target_sr=24000, max_duration_seconds=30, num_ranks=5,
                 audio_transform=None, text_transform=None): # num_ranks for PMF size
        self.wavdir = wavdir
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_duration_seconds)
        self.num_ranks = num_ranks # K (e.g., 5 for MOS 1-5)
        self.audio_transform = audio_transform # For potential future audio augmentations
        self.text_transform = text_transform # For potential future text augmentations

        self.data = []
        self.filenames = [] # Unique filenames

        # Load and process person_mos_list_file
        df = pd.read_csv(person_mos_list_file, header=None, names=['filename', 'rater_id', 'mos_quality', 'mos_alignment'])
        grouped = df.groupby('filename')


        for filename, group in grouped:
            self.filenames.append(filename)
            
            quality_scores = group['mos_quality'].tolist()
            alignment_scores = group['mos_alignment'].tolist()
            
            # Calculate empirical PMF for quality
            quality_counts = Counter(quality_scores)
            quality_pmf = torch.zeros(self.num_ranks)
            for score_val, count in quality_counts.items():
                if 1 <= score_val <= self.num_ranks: # Scores are 1-indexed
                    quality_pmf[int(score_val) - 1] = count
            quality_pmf = quality_pmf / torch.sum(quality_pmf) # Normalize
            
            # Calculate empirical PMF for alignment
            alignment_counts = Counter(alignment_scores)
            alignment_pmf = torch.zeros(self.num_ranks)
            for score_val, count in alignment_counts.items():
                if 1 <= score_val <= self.num_ranks:
                    alignment_pmf[int(score_val) - 1] = count
            alignment_pmf = alignment_pmf / torch.sum(alignment_pmf) # Normalize -> [0.2,0.3,.... ] -> 5 rates..... -> parametrized as beta dist.
            
            mean_quality = np.mean(quality_scores)
            mean_alignment = np.mean(alignment_scores)
            
            self.data.append({
                'filename': filename,
                'quality_scores_list': quality_scores, # Keep original list if needed for other things
                'alignment_scores_list': alignment_scores,
                'quality_pmf': quality_pmf,
                'alignment_pmf': alignment_pmf,
                'mean_quality': float(mean_quality),
                'mean_alignment': float(mean_alignment)
            })

        logging.info(f"PersonMosDataset: Loaded {len(self.filenames)} unique audio files from {person_mos_list_file}")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        item_data = self.data[idx]
        filename = item_data['filename']
        wav_path = os.path.join(self.wavdir, filename)
        
        try:
            wav, sr = torchaudio.load(wav_path)
        except Exception as e:
            logging.error(f"Error loading audio {wav_path}: {e}")
            # Return a dummy item or raise error, for now, let's try to load next if possible or error out
            # This should ideally be caught by a pre-processing check
            # For simplicity in example, we might just return the next valid item or a dummy
            # In a real scenario, filter out such files during __init__
            # For now, let's make it return the first item's data as a placeholder if loading fails
            # This is NOT a good practice for production but helps avoid crashing during dev.
            # A better way is to filter bad files in __init__
            logging.warning(f"Returning data for index 0 due to error with index {idx}")
            item_data = self.data[0] # Fallback to first item
            filename = item_data['filename']
            wav_path = os.path.join(self.wavdir, filename)
            wav, sr = torchaudio.load(wav_path)


        if wav.shape[0] > 1: # If stereo, convert to mono by averaging
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)
            
        if wav.shape[1] > self.max_samples:
            wav = wav[:, :self.max_samples]
        else:
            padding_needed = self.max_samples - wav.shape[1]
            if padding_needed > 0:
                wav = torch.nn.functional.pad(wav, (0, padding_needed))
        
        # Apply transforms if any
        if self.audio_transform:
            wav = self.audio_transform(wav)
        # Text would be fetched in main script using get_texts_from_filename as before

        # For the training script, labels1_orig/labels2_orig are expected to be the scalar target for metrics.
        # We will use the mean of rater scores for this.
        # The target for the loss function (KLDiv) will be the PMFs.
        return (wav, 
                item_data['mean_quality'],   # This will become labels1_orig (for metrics)
                item_data['mean_alignment'], # This will become labels2_orig (for metrics)
                filename,
                item_data['quality_pmf'],    # Additional data for loss
                item_data['alignment_pmf'])  # Additional data for loss

    def collate_fn(self, batch):
        wavs, mean_qualities, mean_alignments, filenames, quality_pmfs, alignment_pmfs = zip(*batch)
        
        wavs = torch.stack(wavs) # (B, C, T) or (B, T) if squeeze happened
        quality_pmfs = torch.stack(quality_pmfs) # (B, num_ranks)
        alignment_pmfs = torch.stack(alignment_pmfs) # (B, num_ranks)
        
        # mean_qualities and mean_alignments are already floats, convert to tensor
        mean_qualities_tensor = torch.tensor(mean_qualities, dtype=torch.float32)
        mean_alignments_tensor = torch.tensor(mean_alignments, dtype=torch.float32)

        # The main training loop expects 4 items from data loader (wavs, labels1_orig, labels2_orig, filenames)
        # We need to pass the PMFs as well. Modify the main loop to handle this.
        # For now, let's return them and adjust the loop later.
        return wavs, mean_qualities_tensor, mean_alignments_tensor, filenames, quality_pmfs, alignment_pmfs
        return output_wavs, overall_scores, coherence_scores, wavnames

class MosDataset(Dataset):
    def __init__(self, wavdir, mos_list, target_sr=24000, max_duration_seconds=30, is_eval_mode=False):
        self.wavdir = wavdir
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_duration_seconds)
        self.is_eval_mode = is_eval_mode # Store the flag

        self.mos_overall_lookup = {}
        self.mos_coherence_lookup = {}
        # This list will store the IDENTIFIER from the mos_list (e.g., 'file.wav' or 'file_id_no_ext')
        # It will be used as the key for lookups and for reconstructing paths in __getitem__
        self.list_item_identifiers = []
        self.resampler_cache = {}

        logging.info(f"Initializing MosDataset: target_sr={target_sr}, max_duration={max_duration_seconds}s, is_eval_mode={self.is_eval_mode}")
        logging.info(f"Reading MOS list: {mos_list}")
        logging.info(f"Expecting WAV files in: {wavdir}")

        try:
            with open(mos_list, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    identifier_from_list = parts[0] # Name as it appears in the list

                    actual_wav_filename_for_disk_check = identifier_from_list
                    mos_overall = 0.0
                    mos_coherence = 0.0

                    if self.is_eval_mode:
                        if len(parts) != 1:
                            logging.warning(f"Skipping malformed line in eval_list {mos_list} (line {line_num+1}): '{line}'. Expected single entry.")
                            continue
                        # identifier_from_list is e.g., "audiomos2025-track1-S030_P050"
                        actual_wav_filename_for_disk_check = identifier_from_list + ".wav" # Assume files on disk HAVE .wav
                    else: # Not eval mode (dev or train list)
                        if len(parts) < 3:
                            logging.warning(f"Skipping malformed line in {mos_list} (line {line_num+1}): '{line}'. Expected wav,mos1,mos2[,...].")
                            continue
                        # identifier_from_list is e.g., "audiomos2025-track1-S002_P044.wav"
                        if not identifier_from_list.lower().endswith('.wav'):
                            logging.warning(f"Filename '{identifier_from_list}' in {mos_list} (line {line_num+1}) does not end with .wav. Errors may occur if file is not found without it.")
                            # Keep actual_wav_filename_for_disk_check as is, or append .wav if you are sure
                            # For dev/train, the list usually dictates the exact filename.
                        
                        try:
                            mos_overall = float(parts[1])
                            mos_coherence = float(parts[2])
                        except ValueError:
                            logging.warning(f"Skipping line with non-float MOS in {mos_list} (line {line_num+1}): '{line}'")
                            continue
                    
                    full_wav_path_to_check = os.path.join(self.wavdir, actual_wav_filename_for_disk_check)
                    
                    if os.path.exists(full_wav_path_to_check):
                        # Use identifier_from_list (which is parts[0]) as the key
                        self.mos_overall_lookup[identifier_from_list] = mos_overall
                        self.mos_coherence_lookup[identifier_from_list] = mos_coherence
                        self.list_item_identifiers.append(identifier_from_list)
                    else:
                        logging.warning(f"Wav file not found: {full_wav_path_to_check} (derived from list entry '{identifier_from_list}', used '{actual_wav_filename_for_disk_check}' for check). Skipping.")

            self.list_item_identifiers.sort() 
            logging.info(f"Found and successfully processed {len(self.list_item_identifiers)} valid audio file entries from {mos_list}.")
            if not self.list_item_identifiers:
                logging.error(f"CRITICAL: No valid audio files were loaded from {mos_list}. Please check paths and file contents.")
        except FileNotFoundError:
            logging.error(f"CRITICAL: MOS list file not found: {mos_list}")
            raise

    def _get_resampler(self, original_sr):
        if original_sr not in self.resampler_cache:
            self.resampler_cache[original_sr] = T.Resample(orig_freq=original_sr, new_freq=self.target_sr)
        return self.resampler_cache[original_sr]

    def __getitem__(self, idx):
        identifier_from_list = self.list_item_identifiers[idx] 
        
        # Determine the actual filename for loading based on mode and identifier format
        filename_to_load = identifier_from_list
        if self.is_eval_mode:
            # For eval mode, identifier_from_list is "audiomos2025-track1-S0XX_P0YY"
            # We need to append ".wav" for the actual file.
            if not filename_to_load.lower().endswith('.wav'): # Should be true for eval_mode keys
                filename_to_load += ".wav"
        # If not is_eval_mode, identifier_from_list is "audiomos2025-track1-S0XX_P0YY.wav" already.

        wavpath = os.path.join(self.wavdir, filename_to_load)
        
        # Default values in case of errors
        wav = torch.zeros(1, self.max_samples) # Dummy wav
        # Fetch scores using the original identifier from the list, which was used as the key
        overall_score = self.mos_overall_lookup.get(identifier_from_list, 0.0) 
        coherence_score = self.mos_coherence_lookup.get(identifier_from_list, 0.0)

        try:
            loaded_wav, sr = torchaudio.load(wavpath) # Load using filename_to_load

            if sr != self.target_sr:
                resampler = self._get_resampler(sr)
                loaded_wav = resampler(loaded_wav)
            
            if loaded_wav.shape[0] > 1: # To mono
                loaded_wav = torch.mean(loaded_wav, dim=0, keepdim=True)

            current_len = loaded_wav.shape[1]
            if current_len > self.max_samples:
                wav = loaded_wav[:, :self.max_samples]
            elif current_len < self.max_samples: # Pad shorter audio
                padding_needed = self.max_samples - current_len
                wav = torch.nn.functional.pad(loaded_wav, (0, padding_needed), 'constant', 0)
            else:
                wav = loaded_wav
            
            # Scores are already fetched/defaulted based on identifier_from_list
            
        except Exception as e:
            logging.error(f"Error processing audio file {wavpath} (identifier: {identifier_from_list}, index {idx}): {e}. Returning dummy waveform.")
            # wav, overall_score, coherence_score are already set to dummy/default values

        # The evaluate_model function needs the filename as it appears on disk for get_texts_from_filename etc.
        # which is filename_to_load (includes .wav)
        return wav, overall_score, coherence_score, filename_to_load

    def __len__(self):
        return len(self.list_item_identifiers)

    @staticmethod
    def collate_fn(batch):
        # (Your existing collate_fn which pads to max_len in batch)
        # This is fine, or can be simplified if __getitem__ always pads to self.max_samples
        wavs, overall_scores, coherence_scores, wavnames_in_batch = zip(*batch)
        
        # If __getitem__ pads to self.max_samples, all wavs here should have that length.
        # Direct stacking would be: wavs_tensor = torch.stack(wavs, dim=0)
        # The per-batch padding below is more robust if lengths from __getitem__ could vary.
        
        max_len_in_batch = max(w.shape[1] for w in wavs) # Determine max length in this specific batch
        output_wavs = []
        for wav_item in wavs:
            if wav_item.shape[1] == max_len_in_batch:
                output_wavs.append(wav_item)
            else:
                amount_to_pad = max_len_in_batch - wav_item.shape[1]
                # Ensure padding amount is not negative (should not happen if max_len_in_batch is correct)
                if amount_to_pad < 0: amount_to_pad = 0 
                padded_wav = torch.nn.functional.pad(wav_item, (0, amount_to_pad), 'constant', 0)
                output_wavs.append(padded_wav)

        wavs_tensor = torch.stack(output_wavs, dim=0)
        overall_scores_tensor = torch.tensor(list(overall_scores), dtype=torch.float32)
        coherence_scores_tensor = torch.tensor(list(coherence_scores), dtype=torch.float32)

        return wavs_tensor, overall_scores_tensor, coherence_scores_tensor, list(wavnames_in_batch)
# class MosDataset(Dataset):
#     """
#     Dataset class for loading audio, resampling, truncating/padding,
#     and retrieving MOS scores.
#     """
#     def __init__(self, wavdir, mos_list, target_sr=24000, max_duration_seconds=30, is_eval_mode=False):
#         """
#         Args:
#             wavdir (str): Directory containing wav files.
#             mos_list (str): Path to the text file with wavname,mos1,mos2 entries.
#             target_sr (int): The target sample rate to resample audio to.
#             max_duration_seconds (int): Maximum duration of audio clips in seconds.
#         """
#         self.wavdir = wavdir
#         self.target_sr = target_sr
#         self.max_samples = int(target_sr * max_duration_seconds)
#         self.mos_overall_lookup = {}
#         self.mos_coherence_lookup = {}
#         self.wavnames = []
#         self.resampler_cache = {} # Cache resamplers to avoid reinitialization
#         self.is_eval_mode = is_eval_mode

#         print(f"Initializing MosDataset: target_sr={target_sr}, max_duration={max_duration_seconds}s")

#         try:
#             with open(mos_list, 'r') as f:
#                 for line in f:
#                     try:
#                         parts = line.strip().split(',')
#                         if self.is_eval_mode:
#                             wavname = parts[0]
#                             mos_overall = 0.0
#                             mos_coherence = 0.0
#                             wavpath = os.path.join(self.wavdir, wavname)
#                             wavpath = wavpath if wavname.endswith('.wav') else wavpath + ".wav" # Ensure .wav extension
#                             if os.path.exists(wavpath):
#                                 self.mos_overall_lookup[wavname] = mos_overall
#                                 self.mos_coherence_lookup[wavname] = mos_coherence
#                                 self.wavnames.append(wavname)
#                             else:
#                                 logger.warning(f"Wav file not found, skipping: {wavpath}")
#                         else:
#                             if len(parts) == 3:
#                                 wavname = parts[0]
#                                 mos_overall = float(parts[1])
#                                 mos_coherence = float(parts[2])
#                                 # Check if wav file exists before adding
#                                 wavpath = os.path.join(self.wavdir, wavname)
#                                 if os.path.exists(wavpath):
#                                     self.mos_overall_lookup[wavname] = mos_overall
#                                     self.mos_coherence_lookup[wavname] = mos_coherence
#                                     self.wavnames.append(wavname)
#                                 else:
#                                     logger.warning(f"Wav file not found, skipping: {wavpath}")
#                             else:
#                                 logger.warning(f"Skipping malformed line in {mos_list}: {line.strip()}")
#                     except ValueError:
#                          logger.warning(f"Skipping line with non-float MOS in {mos_list}: {line.strip()}")
#                     except Exception as e:
#                          logger.error(f"Error processing line: {line.strip()} in {mos_list}. Error: {e}")

#             # Sort for consistency
#             self.wavnames = sorted(self.wavnames)
#             print(f"Found {len(self.wavnames)} valid audio files listed in {mos_list}.")

#         except FileNotFoundError:
#             raise FileNotFoundError(f"MOS list file not found: {mos_list}")

#     def _get_resampler(self, original_sr):
#         """Gets or creates a resampler for the given original SR."""
#         if original_sr not in self.resampler_cache:
#             # Create resampler on the fly; assumes CPU initially, move later if needed
#             # Note: If using GPU dataloading, creating on GPU might be faster but uses more memory
#             self.resampler_cache[original_sr] = T.Resample(orig_freq=original_sr, new_freq=self.target_sr)
#         return self.resampler_cache[original_sr]

#     def __getitem__(self, idx):
#         wavname = self.wavnames[idx]
#         wavpath = os.path.join(self.wavdir, wavname)
#         try:
#             wav, sr = torchaudio.load(wavpath)
#         except Exception as e:
#             logger.error(f"Error loading audio file {wavpath}: {e}")
#             # Return dummy data or skip? Returning dummy data might be safer for dataloader.
#             # Create a silent tensor of expected max length
#             wav = torch.zeros(1, self.max_samples)
#             sr = self.target_sr # Assume target SR for dummy data
#             # Assign placeholder scores?
#             overall_score = 0.0
#             coherence_score = 0.0
#             # Log issue and return dummy
#             logger.warning(f"Returning dummy data for index {idx} due to loading error.")
#             # Fall through to process the dummy tensor

#         # Resample if necessary
#         if sr != self.target_sr:
#             try:
#                 resampler = self._get_resampler(sr)
#                 wav = resampler(wav)
#             except Exception as e:
#                  logger.error(f"Error resampling {wavname} from {sr} to {self.target_sr}: {e}")
#                  # Return dummy data as above
#                  wav = torch.zeros(1, self.max_samples)
#                  overall_score = 0.0
#                  coherence_score = 0.0
#                  logger.warning(f"Returning dummy data for index {idx} due to resampling error.")


#         # Truncate if longer than max_samples
#         if wav.size(1) > self.max_samples:
#             wav = wav[:, :self.max_samples]

#         # Get scores (handle potential loading errors where scores might be dummy)
#         if wavname in self.mos_overall_lookup:
#             overall_score = self.mos_overall_lookup[wavname]
#             coherence_score = self.mos_coherence_lookup[wavname]
#         # else: scores are already dummy from error handling above

#         # Return wav (channel, time), scores, wavname
#         return wav, overall_score, coherence_score, wavname

#     def __len__(self):
#         return len(self.wavnames)

#     @staticmethod
#     def collate_fn(batch):
#         """
#         Collates batch data: Pads waveforms to max length in batch.
#         """
#         # Filter out potential None items if error handling returns None (though current returns dummy)
#         #batch = [b for b in batch if b is not None]
#         #if not batch:
#         #    return None, None, None, None # Handle empty batch case

#         wavs, overall_scores, coherence_scores, wavnames = zip(*batch)

#         # Find max length in this batch
#         max_len = max(w.shape[1] for w in wavs)

#         output_wavs = []
#         for wav in wavs:
#             amount_to_pad = max_len - wav.shape[1]
#             if amount_to_pad < 0: # Should not happen if truncation works, but safety check
#                 logger.warning(f"Unexpected negative padding: {amount_to_pad}. Clamping to 0.")
#                 amount_to_pad = 0
#             padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
#             output_wavs.append(padded_wav)

#         # Stack tensors
#         output_wavs = torch.stack(output_wavs, dim=0) # (batch, channel, time)
#         # Ensure scores are float tensors
#         overall_scores = torch.tensor(list(overall_scores), dtype=torch.float32)
#         coherence_scores = torch.tensor(list(coherence_scores), dtype=torch.float32)

#         return output_wavs, overall_scores, coherence_scores, wavnames


# --- Text Prompt Loading ---

SPECIAL_S_IDS = {"S001", "S006", "S013", "S017", "S018", "S033"} # DEMO system prompts

def get_texts_from_filename(data_dir, filenames):
    prompt_ids = []
    system_ids = []
    for fn in filenames:     # audiomos2025-track1-S002_P044.wav
        try:
            base_fn = fn.replace("audiomos2025-track1-", "")
            s_id = base_fn.split("_")[0]
            p_id = base_fn.split("_")[1].split(".")[0]
            system_ids.append(s_id)
            prompt_ids.append(p_id)
        except IndexError:
            logger.error(f"Could not parse system/prompt ID from filename: {fn}")
            system_ids.append(None)
            prompt_ids.append(None)


    prompt_info_path = os.path.join(data_dir, 'prompt_info.txt')
    demo_prompt_info_path = os.path.join(data_dir, 'demo_prompt_info.txt')

    try:
        df = pd.read_csv(prompt_info_path, sep='	')
        demo_df = pd.read_csv(demo_prompt_info_path, sep='	')
    except FileNotFoundError as e:
        logger.error(f"Prompt info file not found: {e}")
        # Return None for all texts if files are missing
        return [None] * len(filenames)
    except Exception as e:
         logger.error(f"Error reading prompt info files: {e}")
         return [None] * len(filenames)

    texts = []
    for s_id, p_id, orig_fn in zip(system_ids, prompt_ids, filenames):
        text_val = None # Default to None
        if s_id is not None and p_id is not None: # Check if parsing succeeded
            try:
                if s_id in SPECIAL_S_IDS:   # demo_prompt_info
                    # demo_id = 'audiomos2025-track1-' + s_id + '_' + p_id + '.wav' # Original had .wav, but keys might not
                    # Let's try matching without .wav first, assuming df IDs don't have it
                    demo_id_key = 'audiomos2025-track1-' + s_id + '_' + p_id
                    # Find rows where 'id' *starts with* the key, to handle potential variations if needed
                    # text = demo_df.loc[demo_df['id'].str.startswith(demo_id_key), 'text'].values
                    text = demo_df.loc[demo_df['id'] == demo_id_key + '.wav', 'text'].values # Match exact ID with extension

                else:   # prompt_info
                    # text = df.loc[df['id'] == p_id, 'text'].values # p_id is like P044
                     text = df.loc[df['id'] == p_id, 'text'].values # Match exact ID

                if len(text) > 0:
                    text_val = text[0]
                else:
                     logger.warning(f"Text not found for s_id={s_id}, p_id={p_id} in corresponding prompt file.")
            except KeyError:
                 logger.warning(f"'text' column not found in prompt dataframe for s_id={s_id}, p_id={p_id}.")
            except Exception as e:
                 logger.error(f"Error looking up text for s_id={s_id}, p_id={p_id}: {e}")

        texts.append(text_val)

    # Check if any texts are None, log if needed
    none_count = sum(1 for t in texts if t is None)
    if none_count > 0:
        logger.warning(f"Could not find text prompts for {none_count}/{len(filenames)} files.")

    return texts


# --- System ID Helper ---
def systemID(wavID):
    try:
        return wavID.replace("audiomos2025-track1-","").split('_')[0]
    except:
        return None # Handle potential parsing errors