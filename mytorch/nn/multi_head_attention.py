from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()
        
        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project inputs
        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        # Reshape for multiple heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Combine padding and causal masks
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply attention
        attn_outputs = self.attention.forward(q, k, v, mask=mask)

        # Merge heads
        attn_output = self._concat_heads(attn_outputs)

        # Final projection
        output = self.out_proj.forward(attn_output)

        return output

    def backward(self, d_output):
        """
        Backward pass for multi-head attention.
        """

        # Backpropagate through output projection
        d_attn_output = self.out_proj.backward(d_output)

        # Undo head splitting
        d_attn_outputs = self._split_heads(d_attn_output)

        # Backpropagate through attention
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # Merge head gradients
        d_q = self._concat_heads(d_q)
        d_k = self._concat_heads(d_k)
        d_v = self._concat_heads(d_v)

        # Backpropagate through input projections
        d_q = self.q_proj.backward(d_q)
        d_k = self.k_proj.backward(d_k)
        d_v = self.v_proj.backward(d_v)
        
        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge two mask types into a single mask.
        """
        # Expand masks for broadcasting
        key_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
        attention_mask = attn_mask[None, None, :, :] if attn_mask is not None else None
        
        # Combine masks
        if key_mask is None:
            combined_mask = attention_mask
        elif attention_mask is None:
            combined_mask = key_mask
        else:
            combined_mask = np.logical_or(key_mask, attention_mask)
        return combined_mask

    def _split_heads(self, x):
        """
        Reshape tensor for multi-head attention.
        """
        # Reshape and transpose for heads
        x = x.reshape(self.N, x.shape[1], self.num_heads, self.embed_dim // self.num_heads)
        x = np.transpose(x, (0, 2, 1, 3))
        
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # Transpose and reshape
        x = np.transpose(x, (0, 2, 1, 3))
        x = x.reshape(self.N, x.shape[1], self.embed_dim)
        
        return x