import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        # print(z.shape) # (1,)
        sigmoid_func = 1 / (1 + np.exp(-z))
        return np.round(sigmoid_func, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        # print(z.shape) # (3, )
        relu_func = np.maximum(0, z)
        return relu_func
