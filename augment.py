import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalPredictorHeadMonotonic(nn.Module):
    def __init__(self, input_dim, num_ranks, hidden_dims=[512, 256], dropout_rate=0.3):
        super().__init__()
        self.num_ranks = num_ranks
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
            
        self.shared_fc = nn.Sequential(*layers)
        
        # This layer outputs K-1 values that will be transformed into monotonic logits
        self.raw_outputs_layer = nn.Linear(current_dim, self.num_ranks - 1) # Takes input from last hidden_dim

    def forward(self, x):
        h = self.shared_fc(x)
        raw_outputs = self.raw_outputs_layer(h)

        # --- The rest of the forward pass for monotonic logits and MOS score is IDENTICAL ---
        logits = torch.zeros_like(raw_outputs)
        if self.num_ranks > 1:
            logits[:, 0] = raw_outputs[:, 0] 
            for j in range(1, self.num_ranks - 1):
                logits[:, j] = logits[:, j-1] - F.softplus(raw_outputs[:, j])
        
        if self.num_ranks > 1:
            probs_greater_than_rank = torch.sigmoid(logits)
            prob_class1 = 1.0 - probs_greater_than_rank[:, 0:1]
            if self.num_ranks > 2:
                probs_intermediate = probs_greater_than_rank[:, :-1] - probs_greater_than_rank[:, 1:]
            else:
                probs_intermediate = torch.empty(logits.size(0), 0, device=logits.device)
            prob_classK = probs_greater_than_rank[:, -1:]
            class_probabilities = torch.cat([prob_class1, probs_intermediate, prob_classK], dim=1)
            rank_values = torch.arange(1, self.num_ranks + 1, device=logits.device, dtype=torch.float32).unsqueeze(0)
            predicted_mos = torch.sum(class_probabilities * rank_values, dim=1, keepdim=True)
        else:
            predicted_mos = torch.ones(logits.size(0), 1, device=logits.device)

        return logits, predicted_mos




def mixup_data(x_wavs, y1, y2, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0 # No mixup if alpha is 0 or less

    batch_size = x_wavs.size()[0]
    index = torch.randperm(batch_size, device=device)

    mixed_x_wavs = lam * x_wavs + (1 - lam) * x_wavs[index, :]
    
    # y1 and y2 are expected to be 1D tensors of scores or 2D tensors of distributions
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    
    return mixed_x_wavs, y1_a, y1_b, y2_a, y2_b, lam

# No separate mixup_criterion function needed, logic will be in main loop
# def mixup_criterion(criterion, pred, y_a, y_b, lam):
# return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Helper for converting scores to one-hot for distribution models during mixup
def scores_to_gaussian_target(scores, num_bins, device, sigma=0.5): # sigma controls spread
    """
    Converts continuous scores [1, 5] to soft Gaussian distributions over bins.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.float().to(device)
    clamped_scores = torch.clamp(scores, 1.0, 5.0)

    bin_centers = torch.linspace(1.0, 5.0, num_bins, device=device) # (num_bins,)
    
    # Expand scores and bin_centers for broadcasting
    # scores_expanded: (batch_size, 1)
    # bin_centers_expanded: (1, num_bins)
    scores_expanded = clamped_scores.unsqueeze(1)
    bin_centers_expanded = bin_centers.unsqueeze(0)

    # Calculate Gaussian-like weights
    # Using a simplified Gaussian form: exp(- (x - mu)^2 / (2 * sigma^2) )
    # Normalization will be handled by softmax-like behavior later if needed,
    # or we can use these as direct targets for KLDiv if model output is log_softmax.
    # For KLDiv with softmax output from model, target should sum to 1.
    distances = scores_expanded - bin_centers_expanded # (batch_size, num_bins)
    # Gaussian kernel, higher sigma = wider spread
    # Adjust sigma based on bin width, e.g., sigma related to (5-1)/num_bins
    # Let's use a sigma relative to the scale of scores for now
    soft_targets = torch.exp(- (distances ** 2) / (2 * (sigma ** 2)))
    
    # Normalize to sum to 1 for each score (to be a valid probability distribution)
    soft_targets = soft_targets / torch.sum(soft_targets, dim=1, keepdim=True)
    
    return soft_targets


def scores_to_one_hot(scores, num_bins, device):
    """
    Converts continuous scores (assumed to be in [1, 5] range) to one-hot encoded distributions.
    Args:
        scores (torch.Tensor): Tensor of scores, expected range [1, 5].
        num_bins (int): The number of bins for the output distribution.
        device (torch.device): The device to place the output tensor on.
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (batch_size, num_bins).
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    
    scores = scores.float().to(device) # Ensure float and on correct device

    # Normalize scores from [1, 5] to [0, 1], then scale to [0, num_bins-1]
    # (score - min_score) / (max_score - min_score) * (num_bins - 1)
    # (scores - 1.0) / (5.0 - 1.0) = (scores - 1.0) / 4.0
    
    # Ensure scores are clamped to [1, 5] before normalization to prevent out-of-bounds indices
    # from unexpected input score values.
    clamped_scores = torch.clamp(scores, 1.0, 5.0)
    
    label_bins = torch.floor(((clamped_scores - 1.0) / 4.0) * (num_bins - 1))
    
    # Final clamp to ensure indices are strictly within [0, num_bins-1]
    # This handles potential floating point inaccuracies, especially at the boundaries.
    label_bins = torch.clamp(label_bins.long(), 0, num_bins - 1)
    
    one_hot = torch.zeros(scores.size(0), num_bins, device=device)
    one_hot.scatter_(1, label_bins.unsqueeze(1), 1)
    return one_hot


def scores_to_gaussian_target(scores, num_bins, device, sigma=0.25):
    """
    Converts continuous scores [1, 5] to soft Gaussian distributions over bins.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.float().to(device)
    clamped_scores = torch.clamp(scores, 1.0, 5.0)

    bin_centers = torch.linspace(1.0, 5.0, num_bins, device=device)
    
    scores_expanded = clamped_scores.unsqueeze(1)
    bin_centers_expanded = bin_centers.unsqueeze(0)

    distances = scores_expanded - bin_centers_expanded
    soft_targets = torch.exp(- (distances ** 2) / (2 * (sigma ** 2)))
    
    # Normalize to sum to 1
    soft_targets = soft_targets / torch.sum(soft_targets, dim=1, keepdim=True)
    
    return soft_targets