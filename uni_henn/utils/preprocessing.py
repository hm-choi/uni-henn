import numpy as np
import torch
from .structure import Cuboid

def preprocessing(data: np.array, Img: Cuboid, num_of_data: int, data_size: int):
    """
    The function that preprocesses plaintext data into a format compatible with the current model structure.

    Args:
        - data: Plaintext data to preprocess
        - Img: Width (and height) of used image data
        - num_of_data: Number of data
        - data_size: Maximum data size from the total layers

    Returns:
        - C_out: The output of the flattened result of the input ciphertext list
    """
    ppData = []
    for h in range(Img.z):
        ppData.append([])
        for i in range(num_of_data):
            ppData[h].extend(data[i][h].flatten())
            ppData[h].extend([0] * (data_size - Img.size2d()))
        ppData[h] = torch.Tensor(ppData[h])

    return ppData