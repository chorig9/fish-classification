import random
import numpy as np


def random_split_indexes(size, split = 0.1):
    """
    Args:
        size:  range of indexes to split
        split: defines how to split data

    Returns:
        two numpy arrays of random indexes, first of size: size * split and second of size: size * (1 - split)
    """

    first = np.random.choice(size, int(size * split), replace=False)
    second = [x for x in range(0, size) if x not in first]

    return first, np.array(second)
