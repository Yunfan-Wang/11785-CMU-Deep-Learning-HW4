import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask for padding positions. 
    Args:
        padded_input: The input tensor, shape (N, T, ...).
        input_lengths: Actual lengths before padding, shape (N,).
    Returns:
        Boolean mask tensor with shape (N, T).
    """
    # TODO: Implement PadMask
    N, T = padded_input.shape[:2]
    device = padded_input.device
    positions = torch.arange(T, device=device).unsqueeze(0)
    lengths = input_lengths.unsqueeze(1)

    mask = positions >= lengths
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a causal mask for self-attention. 
    Args:
        padded_input: Input tensor, shape (N, T, ...).
    Returns:
        Boolean mask tensor with shape (T, T).
    """
    T = padded_input.shape[1]
    device = padded_input.device

    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    return mask
