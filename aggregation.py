"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch

_SELECTED_LAYERS = [11, 15, 19, 23]     # layers to pool

def _masked_mean(layer: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean hidden state across all real tokens where mask == 1, ignoring padding .

    Args:
        layer: ``(seq_len, hidden_dim)`` float tensor.
        mask:  ``(seq_len,)`` tensor (any dtype); moved to layer's device.

    Returns:
        ``(hidden_dim,)`` float tensor on the same device as ``layer``.
    """
    layer  = layer.float()
    mask   = mask.view(-1).to(device=layer.device, dtype=torch.float32)
    mask_f = mask.unsqueeze(-1)                    # (seq_len, 1)
    summed = (layer * mask_f).sum(dim=0)
    count  = mask_f.sum().clamp(min=1.0)
    return summed / count

def _last_token(layer: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the hidden state of the last *real* (non-padding) token.

    Args:
        layer: ``(seq_len, hidden_dim)`` float tensor.
        mask:  ``(seq_len,)`` tensor; 1 for real tokens.

    Returns:
        ``(hidden_dim,)`` float tensor.
    """
    layer = layer.float()
    mask  = mask.view(-1).to(device=layer.device, dtype=torch.float32)
    real_positions = mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    return layer[last_pos]

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(n_selected * 2 * hidden_dim,)``.
    """
    n_layers = hidden_states.size(0)
    parts: list[torch.Tensor] = []

    for idx in _SELECTED_LAYERS:
        safe_idx = min(idx, n_layers - 1)
        layer    = hidden_states[safe_idx].float()

        mean_vec  = _masked_mean(layer, attention_mask)   # (hidden_dim,)
        last_vec  = _last_token(layer, attention_mask)    # (hidden_dim,)
        parts.append(torch.cat([mean_vec, last_vec], dim=0))

    return torch.cat(parts, dim=0)   # (n_selected * 2 * hidden_dim,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
