import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        self.Z = Z

        Z_shifted = Z - Z.max(axis=self.dim, keepdims=True)

        exp_Z = np.exp(Z_shifted)
        self.out = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)

        return self.out
    
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        s = self.out

        # Jacobian-vector product for softmax
        dot = np.sum(dLdA * s, axis=self.dim, keepdims=True)
        dLdX = s * (dLdA - dot)

        return dLdX
 

    