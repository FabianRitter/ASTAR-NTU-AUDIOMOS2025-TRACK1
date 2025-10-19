import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BasePredictor
from transformers import RobertaTokenizer
import math # For positional encoding
import ipdb as pdb
from mamba_ssm import Mamba
from augment import OrdinalPredictorHeadMonotonic
from models.uncertainty_components import BetaParameterHead
from models.base_models import PositionalEncoding, AttentivePooling, BaseTransformerPredictor


class MuQMulanRoBERTaTransformerDistributionPredictor(BasePredictor):
    pass # not implemented in this open code, easy to make by changing other classes.

class MuQRoBERTaTransformerScalarPredictor(nn.Module): # Or BasePredictor
    def __init__(self, muq_model, roberta_model,
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_dim=1024,
                 text_transformer_layers=1, text_transformer_heads=4, text_transformer_dim=768,
                 common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3):
        super().__init__()
        # All the __init__ logic is identical to the distribution model, except for the MLPs.
        self.muq = muq_model
        self.roberta = roberta_model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.audio_transformer_dim = audio_transformer_dim
        self.text_transformer_dim = text_transformer_dim
        self.muq_output_dim = 1024
        self.roberta_output_dim = 768

        self.audio_pos_encoder = PositionalEncoding(d_model=1024, dropout=dropout_rate)

        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=audio_transformer_heads,
            dim_feedforward=audio_transformer_dim * 2, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer, num_layers=audio_transformer_layers
        )
        self.audio_attentive_pool = AttentivePooling(input_dim=1024)

        self.common_embed_dim = common_embed_dim
        self.audio_seq_proj = nn.Linear(self.muq_output_dim, self.common_embed_dim)
        self.text_seq_proj = nn.Linear(self.roberta_output_dim, self.common_embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_embed_dim, num_heads=cross_attention_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(self.common_embed_dim)
        self.fused_attentive_pool = AttentivePooling(input_dim=self.common_embed_dim)

        # --- PREDICTION HEADS (THE ONLY MAJOR CHANGE) ---
        # The MLPs now output a single value (dim=1) and have no Softmax.

        # Overall quality (from pooled audio)
        self.overall_mlp = nn.Sequential(
            nn.Linear(self.muq_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # <--- CHANGED: Output a single scalar value
            # Softmax is removed
        )
 
        # Coherence (from pooled cross-attended features)
        self.coherence_input_dim = self.common_embed_dim
        self.coherence_mlp = nn.Sequential(
            nn.Linear(self.coherence_input_dim, self.coherence_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.coherence_input_dim // 2, self.coherence_input_dim // 4),
            nn.ReLU(),
            nn.Linear(self.coherence_input_dim // 4, 1), # <--- CHANGED: Output a single scalar value
            # Softmax is removed
        )
        # The buffer for bin_centers is no longer needed.

    def create_padding_mask(self, seq_lens, max_len):
        batch_size = seq_lens.size(0)
        mask = torch.arange(max_len, device=seq_lens.device).expand(batch_size, max_len) >= seq_lens.unsqueeze(1)
        return mask

    def forward(self, wavs, texts):
        # The forward pass logic is identical until the final return statement.
        with torch.no_grad():
            muq_output = self.muq(wavs, output_hidden_states=False)
            audio_seq_embed = muq_output.last_hidden_state

            text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            text_attention_mask = text_inputs['attention_mask'].to(wavs.device)
            text_inputs_on_device = {k: v.to(wavs.device) for k, v in text_inputs.items()}
            roberta_output = self.roberta(**text_inputs_on_device)
            text_seq_embed = roberta_output.last_hidden_state
        
        audio_padding_mask = None
        audio_seq_embed_pe = self.audio_pos_encoder(audio_seq_embed)
        audio_transformed = self.audio_transformer_encoder(src=audio_seq_embed_pe, src_key_padding_mask=audio_padding_mask)
        pooled_audio_features = self.audio_attentive_pool(audio_transformed, mask=audio_padding_mask)
        
        # --- Overall Quality Prediction ---
        overall_score = self.overall_mlp(pooled_audio_features) # <--- CHANGED: Directly get the score

        # --- Audio-Text Fusion ---
        audio_seq_proj = self.audio_seq_proj(audio_transformed)
        text_seq_proj = self.text_seq_proj(text_seq_embed)
        text_padding_mask = (text_attention_mask == 0)

        cross_attended_output, _ = self.cross_attention(
            query=text_seq_proj, key=audio_seq_proj, value=audio_seq_proj,
            key_padding_mask=audio_padding_mask
        )
        cross_attended_output_norm = self.cross_attention_norm(cross_attended_output)
        fused_features = self.fused_attentive_pool(cross_attended_output_norm, mask=text_padding_mask)
  
        # --- Coherence Prediction ---
        coherence_score = self.coherence_mlp(fused_features) # <--- CHANGED: Directly get the score

        # --- CHANGED: Return the two scalar scores ---
        return overall_score, coherence_score




class MuQRoBERTaTransformerDistributionPredictor(BaseTransformerPredictor):
    def __init__(self, muq_model, roberta_model, num_bins=20,
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_dim=1024,
                 text_transformer_layers=1, text_transformer_heads=4, text_transformer_dim=768, # For text if not using RoBERTa directly
                common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3):
        super().__init__()
        

        # --- Prediction Heads ---
        # Overall quality (from pooled audio)
        self.overall_mlp = nn.Sequential(
            nn.Linear(self.muq_output_dim, 512), # Input from attentive_pool_audio
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_bins),
            nn.Softmax(dim=1)
        )
 
        # Coherence (from pooled cross-attended features)
        self.coherence_input_dim = self.common_embed_dim # From fused_attentive_pool output
        self.coherence_mlp = nn.Sequential(
            nn.Linear(self.coherence_input_dim, self.coherence_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.coherence_input_dim // 2, self.coherence_input_dim // 4),
            nn.ReLU(),
            nn.Linear(self.coherence_input_dim // 4, num_bins),
            nn.Softmax(dim=1)
        )

        self.register_buffer('bin_centers', torch.linspace(1, 5, num_bins))
    

    def forward(self, wavs, texts):
        pooled_audio_features, fused_features = self.forward_features(
            wavs, texts, use_decoupled_audio_for_cross_attn=False
        )
        overall_dist = self.overall_mlp(pooled_audio_features)
        overall_expected = torch.sum(overall_dist * self.bin_centers, dim=1, keepdim=True)
        # --- Coherence Prediction ---
        coherence_dist = self.coherence_mlp(fused_features)
        coherence_expected = torch.sum(coherence_dist * self.bin_centers, dim=1, keepdim=True)

        return overall_dist, coherence_dist, overall_expected, coherence_expected



class MuQRoBERTaTransformerDistributionPredictorCORALPredictor(BasePredictor):
    def __init__(self, muq_model, roberta_model, num_ranks=5, # K=5 for MOS 1-5
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_dim=1024,
                 common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3):
        super().__init__()
        self.muq = muq_model
        self.roberta = roberta_model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.num_ranks = num_ranks

        # --- Audio Path (identical to your MuQRoBERTaTransformerDistributionPredictor) ---
        self.audio_pos_encoder = PositionalEncoding(d_model=1024, dropout=dropout_rate)
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=audio_transformer_heads,
            dim_feedforward=audio_transformer_dim * 2, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer, num_layers=audio_transformer_layers
        )
        self.audio_attentive_pool = AttentivePooling(input_dim=1024) # Pooled audio features

        # --- Text Path & Projection (identical) ---
        self.common_embed_dim = common_embed_dim
        self.muq_output_dim = 1024
        self.roberta_output_dim = 768
        self.audio_seq_proj = nn.Linear(self.muq_output_dim, self.common_embed_dim)
        self.text_seq_proj = nn.Linear(self.roberta_output_dim, self.common_embed_dim)

        # --- Cross-Attention Module (identical) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_embed_dim, num_heads=cross_attention_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(self.common_embed_dim)
        self.fused_attentive_pool = AttentivePooling(input_dim=self.common_embed_dim) # Pooled fused features

        ordinal_head_hidden_dims = [512, 256] # New argument for hidden dimensions in OrdinalPredictorHead

        # --- Prediction Heads now use OrdinalPredictorHead ---
        # Overall quality head (operates on pooled audio features)
        self.overall_quality_head = OrdinalPredictorHeadMonotonic(
            input_dim=1024, # from self.audio_attentive_pool
            num_ranks=self.num_ranks,
            hidden_dims=ordinal_head_hidden_dims, # Pass the new arg
            dropout_rate=dropout_rate
        )
        # Coherence head (operates on pooled fused features)
        self.coherence_head = OrdinalPredictorHeadMonotonic(
            input_dim=self.common_embed_dim, # from self.fused_attentive_pool
            num_ranks=self.num_ranks,
            hidden_dims=ordinal_head_hidden_dims, # Pass the new arg
            dropout_rate=dropout_rate
        )
        
    def create_padding_mask(self, seq_lens, max_len): # Keep this helper
        batch_size = seq_lens.size(0)
        mask = torch.arange(max_len, device=seq_lens.device).expand(batch_size, max_len) >= seq_lens.unsqueeze(1)
        return mask

    def forward(self, wavs, texts):
        # --- Base Feature Extraction (MuQ & RoBERTa) ---        
        
        muq_output = self.muq(wavs, output_hidden_states=False)
        audio_seq_embed = muq_output.last_hidden_state

        text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        text_attention_mask = text_inputs['attention_mask'].to(wavs.device)
        text_inputs_on_device = {k: v.to(wavs.device) for k, v in text_inputs.items()}
        roberta_output = self.roberta(**text_inputs_on_device)
        text_seq_embed = roberta_output.last_hidden_state
        
        # --- Audio Path (identical) ---
        audio_padding_mask = None # Placeholder - ideally, create from audio lengths
        audio_seq_embed_pe = self.audio_pos_encoder(audio_seq_embed)
        audio_transformed = self.audio_transformer_encoder(src=audio_seq_embed_pe, src_key_padding_mask=audio_padding_mask)
        pooled_audio_features = self.audio_attentive_pool(audio_transformed, mask=audio_padding_mask)

        # --- Audio-Text Fusion (Cross-Attention) (identical) ---
        audio_seq_proj = self.audio_seq_proj(audio_transformed)
        text_seq_proj = self.text_seq_proj(text_seq_embed)
        text_padding_mask_for_cross_attn_output_pooling = (text_attention_mask == 0) # True for padding
        
        # key_padding_mask for MultiheadAttention refers to the PADDING in KEY sequence (audio)
        cross_attended_output, _ = self.cross_attention(
            query=text_seq_proj, key=audio_seq_proj, value=audio_seq_proj,
            key_padding_mask=audio_padding_mask # Mask for audio sequence (Keys)
        )
        cross_attended_output_norm = self.cross_attention_norm(cross_attended_output)
        fused_features = self.fused_attentive_pool(
            cross_attended_output_norm,
            mask=text_padding_mask_for_cross_attn_output_pooling # Mask for text sequence (Query output)
        )

        # --- Predictions using OrdinalPredictorHead ---
        # overall_logits: (batch, num_ranks-1), overall_mos_pred: (batch, 1)
        overall_logits, overall_mos_pred = self.overall_quality_head(pooled_audio_features)
        # coherence_logits: (batch, num_ranks-1), coherence_mos_pred: (batch, 1)
        coherence_logits, coherence_mos_pred = self.coherence_head(fused_features)

        # Return logits for CORAL loss, and MOS predictions for metrics
        return overall_logits, coherence_logits, overall_mos_pred, coherence_mos_pred
    



class MuQRoBERTaTransformerDecoupledDist(MuQRoBERTaTransformerDistributionPredictor):

    def forward(self, wavs, texts):
        # Use the decoupled path for feature extraction
        pooled_audio_features, fused_features = self.forward_features(
            wavs, texts, use_decoupled_audio_for_cross_attn=True
        )

        # Prediction heads are the same as the parent class
        overall_dist = self.overall_mlp(pooled_audio_features)
        overall_expected = torch.sum(overall_dist * self.bin_centers, dim=1, keepdim=True)

        coherence_dist = self.coherence_mlp(fused_features)
        coherence_expected = torch.sum(coherence_dist * self.bin_centers, dim=1, keepdim=True)

        return overall_dist, coherence_dist, overall_expected, coherence_expected



class MuQRoBERTaTransformerLSTMHeadCrossAttnDecoupledPredictor(BasePredictor):
    def __init__(self, muq_model, roberta_model, num_bins=20,
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_ff_dim_factor=2, # d_model * factor for feedforward
                 common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3,
                 # LSTM specific parameters for prediction heads
                 lstm_hidden_size_overall=512, lstm_num_layers_overall=1, bidirectional_lstm_overall=False,
                 lstm_hidden_size_coherence=384, lstm_num_layers_coherence=1, bidirectional_lstm_coherence=False): # coherence input is common_embed_dim (768)
        super().__init__()
        self.muq = muq_model
        self.roberta = roberta_model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.num_bins = num_bins

        self.muq_output_dim = 1024 # Hardcoded based on MuQ model
        self.roberta_output_dim = 768 # Hardcoded based on roberta-base
        self.common_embed_dim = common_embed_dim

        # --- Audio Path (for overall quality) ---
        # This path processes audio features specifically for the overall quality prediction
        self.audio_pos_encoder = PositionalEncoding(d_model=self.muq_output_dim, dropout=dropout_rate)
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.muq_output_dim,
            nhead=audio_transformer_heads,
            dim_feedforward=self.muq_output_dim * audio_transformer_ff_dim_factor,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer,
            num_layers=audio_transformer_layers
        )
        self.audio_attentive_pool = AttentivePooling(input_dim=self.muq_output_dim)

        # --- Text Path Projection (for cross-attention query) ---
        self.text_seq_proj = nn.Linear(self.roberta_output_dim, self.common_embed_dim)

        # --- CHANGE 2: New projection for raw audio_seq_embed for cross-attention key/value ---
        # This uses the MuQ output directly, before the audio-specific transformer, for cross-attention
        self.audio_embed_to_common_for_crossattn = nn.Linear(self.muq_output_dim, self.common_embed_dim)

        # --- Cross-Attention Module ---
        # Text sequence (Query) attends to Audio sequence (Key, Value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_embed_dim,
            num_heads=cross_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(self.common_embed_dim)
        self.fused_attentive_pool = AttentivePooling(input_dim=self.common_embed_dim)


        # --- CHANGE 1: Prediction Heads with LSTM + Linear ---

        # Overall quality head (operates on pooled_audio_features from audio_transformer_encoder path)
        self.lstm_hidden_size_overall = lstm_hidden_size_overall
        self.bidirectional_lstm_overall = bidirectional_lstm_overall
        self.overall_lstm = nn.LSTM(
            input_size=self.muq_output_dim, # Input from self.audio_attentive_pool
            hidden_size=self.lstm_hidden_size_overall,
            num_layers=lstm_num_layers_overall,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers_overall > 1 else 0,
            bidirectional=self.bidirectional_lstm_overall
        )
        overall_lstm_fc_input_dim = self.lstm_hidden_size_overall * (2 if self.bidirectional_lstm_overall else 1)
        self.overall_fc = nn.Linear(overall_lstm_fc_input_dim, num_bins)
        self.overall_softmax = nn.Softmax(dim=1)
 
        # Coherence head (operates on fused_features from cross-attention path)
        self.lstm_hidden_size_coherence = lstm_hidden_size_coherence
        self.bidirectional_lstm_coherence = bidirectional_lstm_coherence
        self.coherence_lstm = nn.LSTM(
            input_size=self.common_embed_dim, # Input from self.fused_attentive_pool
            hidden_size=self.lstm_hidden_size_coherence,
            num_layers=lstm_num_layers_coherence,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers_coherence > 1 else 0,
            bidirectional=self.bidirectional_lstm_coherence
        )
        coherence_lstm_fc_input_dim = self.lstm_hidden_size_coherence * (2 if self.bidirectional_lstm_coherence else 1)
        self.coherence_fc = nn.Linear(coherence_lstm_fc_input_dim, num_bins)
        self.coherence_softmax = nn.Softmax(dim=1)

        self.register_buffer('bin_centers', torch.linspace(1, 5, num_bins))
    
    def create_padding_mask(self, seq_lens, max_len):
        # Creates a boolean mask (True where padded)
        batch_size = seq_lens.size(0)
        mask = torch.arange(max_len, device=seq_lens.device).expand(batch_size, max_len) >= seq_lens.unsqueeze(1)
        return mask # Shape: (batch_size, max_len)

    def forward(self, wavs, texts):
        # --- Base Feature Extraction (MuQ & RoBERTa) ---
        # Note: Assuming fine-tuning, so no torch.no_grad() context here
        muq_output = self.muq(wavs, output_hidden_states=False)
        audio_seq_embed = muq_output.last_hidden_state # Shape: (batch, seq_len_audio, muq_output_dim=1024)

        text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        text_attention_mask = text_inputs['attention_mask'].to(wavs.device) # (batch, seq_len_text), 1 for real, 0 for padding
        text_inputs_on_device = {k: v.to(wavs.device) for k, v in text_inputs.items()}
        roberta_output = self.roberta(**text_inputs_on_device)
        text_seq_embed = roberta_output.last_hidden_state # Shape: (batch, seq_len_text, roberta_output_dim=768)

        # --- Audio Path (for overall quality prediction) ---
        # TODO: Create audio_padding_mask if audio sequences have variable lengths and are padded.
        # For now, assuming MuQ output has fixed length or masking is handled internally/not strictly needed for attentive pooling.
        audio_padding_mask = None # Placeholder

        audio_seq_embed_pe = self.audio_pos_encoder(audio_seq_embed)
        audio_transformed = self.audio_transformer_encoder(
            src=audio_seq_embed_pe,
            src_key_padding_mask=audio_padding_mask
        ) # Shape: (batch, seq_len_audio, muq_output_dim)
        
        pooled_audio_features = self.audio_attentive_pool(
            audio_transformed,
            mask=audio_padding_mask # Apply mask if available
        ) # Shape: (batch_size, muq_output_dim)

        # --- CHANGE 1: Overall Quality Prediction with LSTM Head ---
        # LSTM expects input of (batch, seq_len, feature_dim)
        overall_lstm_input = pooled_audio_features.unsqueeze(1) # Shape: (batch, 1, muq_output_dim)
        
        # self.overall_lstm.flatten_parameters() # Optional: for DDP
        overall_lstm_out, _ = self.overall_lstm(overall_lstm_input) # overall_lstm_out: (batch, 1, hidden_size * directions)
        
        overall_features_for_fc = overall_lstm_out[:, -1, :] # Take the output of the last (only) time step
                                                             # Shape: (batch, lstm_hidden_size_overall * directions)
        
        overall_dist_logits = self.overall_fc(overall_features_for_fc)
        overall_dist = self.overall_softmax(overall_dist_logits)
        overall_expected = torch.sum(overall_dist * self.bin_centers.unsqueeze(0), dim=1, keepdim=True)


        # --- Audio-Text Fusion (Cross-Attention) ---
        # Project text sequence to common dimension for Query
        text_seq_proj_out = self.text_seq_proj(text_seq_embed) # Shape: (batch, seq_len_text, common_embed_dim)

        # --- CHANGE 2: Project raw audio_seq_embed to common dimension for Key & Value ---
        audio_key_value_for_crossattn = self.audio_embed_to_common_for_crossattn(audio_seq_embed)
        # Shape: (batch, seq_len_audio, common_embed_dim)

        # Create text padding mask (True where padded, for pooling the cross-attention output)
        text_output_padding_mask = (text_attention_mask == 0)

        # Perform cross-attention: Text (Query) attends to Audio (Key, Value)
        # key_padding_mask for MultiheadAttention refers to padding in Key sequence (audio here)
        cross_attended_output, _ = self.cross_attention(
            query=text_seq_proj_out,
            key=audio_key_value_for_crossattn,
            value=audio_key_value_for_crossattn,
            key_padding_mask=audio_padding_mask # Mask for audio sequence (Keys)
        ) # Shape: (batch, seq_len_text, common_embed_dim)

        cross_attended_output_norm = self.cross_attention_norm(cross_attended_output)
        
        fused_features = self.fused_attentive_pool(
            cross_attended_output_norm,
            mask=text_output_padding_mask # Mask based on text padding (since output seq_len is text_seq_len)
        ) # Shape: (batch_size, common_embed_dim)
  
        # --- CHANGE 1: Coherence Prediction with LSTM Head ---
        coherence_lstm_input = fused_features.unsqueeze(1) # Shape: (batch, 1, common_embed_dim)

        # self.coherence_lstm.flatten_parameters() # Optional: for DDP
        coherence_lstm_out, _ = self.coherence_lstm(coherence_lstm_input) # coherence_lstm_out: (batch, 1, hidden_size * directions)
        
        coherence_features_for_fc = coherence_lstm_out[:, -1, :] # Take the output of the last (only) time step
                                                                 # Shape: (batch, lstm_hidden_size_coherence * directions)

        coherence_dist_logits = self.coherence_fc(coherence_features_for_fc)
        coherence_dist = self.coherence_softmax(coherence_dist_logits)
        coherence_expected = torch.sum(coherence_dist * self.bin_centers.unsqueeze(0), dim=1, keepdim=True)

        return overall_dist, coherence_dist, overall_expected, coherence_expected