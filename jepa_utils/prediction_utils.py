from typing import Dict, List
import torch


def decode_predicted_codes(predicted: torch.Tensor, id_to_token: Dict[int, str]) -> List[List[str]]:
    """Convert predicted token tensor to a list of code strings.

    Supports multi-hot vectors (0/1) or integer token IDs of shape
    ``[batch, num_tokens]``. Padding token ``0`` is ignored.
    """
    predicted = predicted.cpu()
    results: List[List[str]] = []
    if predicted.dim() == 2:
        if predicted.max() > 1:
            # token IDs from diffusion
            for row in predicted:
                codes = [id_to_token.get(int(idx), '<UNK>') for idx in row.tolist() if idx != 0]
                results.append(codes)
        else:
            for row in predicted:
                indices = (row == 1).nonzero(as_tuple=True)[0].tolist()
                codes = [id_to_token.get(int(idx), '<UNK>') for idx in indices if idx != 0]
                results.append(codes)
    else:
        for idx in predicted.tolist():
            codes = [id_to_token.get(int(idx), '<UNK>')]
            results.append(codes)
    return results
