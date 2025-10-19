import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from models.base import BasePredictor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len needs to cover longest sequence
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Becomes (max_len, 1, d_model)
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if x.dim() == 3 and x.size(1) == self.pe.size(1): # if batch_first=False (seq_len, batch, dim)
            x = x + self.pe[:x.size(0), :]
        elif x.dim() == 3 and x.size(0) == self.pe.size(1): # if batch_first=True (batch, seq_len, dim)
            x = x + self.pe[:x.size(1), :].transpose(0,1)
        else:
            # Fallback for 2D tensor (e.g. CLS token) - no positional encoding needed or apply differently
             pass # Or raise error if unexpected
        return self.dropout(x)

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional tensor of shape (batch_size, seq_len) for padding
        Returns:
            Tensor of shape (batch_size, input_dim)
        """
        # Calculate attention scores
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, 1)
        attention_scores = self.attention_weights(x)

        if mask is not None:
            # Apply mask (set scores for padding tokens to a very small number)
            attention_scores.masked_fill_(mask.unsqueeze(-1) == 0, -1e9) # Mask out padding

        # Softmax to get probabilities
        attention_probs = F.softmax(attention_scores, dim=1)

        # Weighted sum
        # (batch_size, seq_len, 1) * (batch_size, seq_len, input_dim) -> sum over seq_len
        context = torch.sum(attention_probs * x, dim=1)
        return context

class BaseTransformerPredictor(BasePredictor):
    """
    Base class for the dual-branch transformer architecture.
    Handles feature extraction, temporal modeling, and fusion.
    """
    def __init__(self, muq_model, roberta_model,
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_dim=1024,
                 common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3):
        super().__init__()
        self.muq = muq_model
        self.roberta = roberta_model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        for param in self.muq.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.muq_output_dim = 1024  # MuQ-large
        self.roberta_output_dim = 768 # RoBERTa-base
        self.common_embed_dim = common_embed_dim

        # --- Audio Path (Temporal Modeling) ---
        self.audio_pos_encoder = PositionalEncoding(d_model=self.muq_output_dim, dropout=dropout_rate)
        
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.muq_output_dim, nhead=audio_transformer_heads,
            dim_feedforward=audio_transformer_dim * 2, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer, num_layers=audio_transformer_layers
        )
        # Pooling for MI branch
        self.audio_attentive_pool = AttentivePooling(input_dim=self.muq_output_dim)

        # --- Feature Fusion (Cross-Attention) ---
        # Projections to common space (Linear layers in diagram)
        self.audio_seq_proj = nn.Linear(self.muq_output_dim, self.common_embed_dim)
        self.text_seq_proj = nn.Linear(self.roberta_output_dim, self.common_embed_dim)

        # Cross-Attention (Feature Fusion in diagram)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_embed_dim, num_heads=cross_attention_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(self.common_embed_dim)
        
        # Pooling for TA branch
        self.fused_attentive_pool = AttentivePooling(input_dim=self.common_embed_dim)

    def forward_features(self, wavs, texts, use_decoupled_audio_for_cross_attn=False):
        """
        Forward pass for feature extraction and fusion.
        """
        # --- Base Feature Extraction (Frozen Encoders) ---
        # Encoders are frozen in __init__. We set them to eval() mode.
        self.muq.eval()
        self.roberta.eval()
            
        # We do not need torch.no_grad() here if requires_grad=False is set correctly. Please check if this works, this refactored code has not been verified.
        muq_output = self.muq(wavs, output_hidden_states=False)
        audio_seq_embed_raw = muq_output.last_hidden_state # (B, T_a, D_a)

        # Text Encoder
        text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        text_attention_mask = text_inputs['attention_mask'].to(wavs.device) # (B, T_t)
        text_inputs_on_device = {k: v.to(wavs.device) for k, v in text_inputs.items()}
        roberta_output = self.roberta(**text_inputs_on_device)
        text_seq_embed = roberta_output.last_hidden_state # (B, T_t, D_t)

        # Note: Audio padding mask is currently not utilized in the original implementation.
        audio_padding_mask = None # Transformer expects True where padded

        # --- Audio Path (Temporal Modeling) ---
        audio_seq_embed_pe = self.audio_pos_encoder(audio_seq_embed_raw)
        audio_transformed = self.audio_transformer_encoder(
            src=audio_seq_embed_pe,
            src_key_padding_mask=audio_padding_mask
        )

        # Pooling for MI branch
        pooled_audio_features = self.audio_attentive_pool(
            audio_transformed,
            mask=audio_padding_mask
        )

        
        # Determine audio input for cross-attention (Key, Value)
        if use_decoupled_audio_for_cross_attn:
            audio_input_for_fusion = audio_seq_embed_raw # Decoupled Architecture
        else:
            audio_input_for_fusion = audio_transformed # Standard Architecture

        # Project to common dimension
        audio_seq_proj = self.audio_seq_proj(audio_input_for_fusion)
        text_seq_proj = self.text_seq_proj(text_seq_embed)

        # Text padding mask (True where PADDED)
        text_padding_mask = (text_attention_mask == 0)

        # Cross-attention: Text (Q) attends to Audio (K, V)
        cross_attended_output, _ = self.cross_attention(
            query=text_seq_proj, key=audio_seq_proj, value=audio_seq_proj,
            key_padding_mask=audio_padding_mask
        )

        cross_attended_output_norm = self.cross_attention_norm(cross_attended_output)

        # Pooling for TA branch
        fused_features = self.fused_attentive_pool(
            cross_attended_output_norm,
            mask=text_padding_mask
        )

        return pooled_audio_features, fused_features