import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        stability_z = z - np.max(z)
        softmax = np.exp(stability_z) / np.sum(np.exp(stability_z))
        # divided by a scalar value
        return np.round(softmax, 4)
