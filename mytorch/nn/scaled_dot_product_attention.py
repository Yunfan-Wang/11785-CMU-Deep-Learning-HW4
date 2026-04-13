import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        self.d_k = Q.shape[-1]
        # Calculate attention scores
        scaled_dot_product = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(self.d_k)
        
        # Apply mask before softmax if provided
        if mask is not None:
            scaled_dot_product = np.where(mask, -1e9, scaled_dot_product)

        self.scaled_dot_product = scaled_dot_product
        # Compute attention scores: 
        # # Think about which dimension you should apply Softmax
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate final output
        output = self.attention_scores @ V

        # Return final output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V
        d_V = np.swapaxes(self.attention_scores, -1, -2) @ d_output
        
        # Calculate gradients for attention scores
        d_attention_scores = d_output @ np.swapaxes(self.V, -1, -2)
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.d_k)
        
        # Calculate gradients for Q and K
        d_Q = d_scaled_dot_product @ self.K
        d_K = np.swapaxes(d_scaled_dot_product, -1, -2) @ self.Q
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

