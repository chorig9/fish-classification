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

def split_data(X, Y, p, seed=1337, fpaths=None, ret_filepaths=False):
    """
    X - data
    Y - labels
    p - percent of testing data e.g. 0.1 = 10%

    Returns X_train, Y_train, X_test, Y_test, (opt)filepaths
    filepaths - contains paths for VALIDATION SET
    """

    dataset_size = len(X)
    testset_size = int(dataset_size*p)

    X_test = []
    Y_test = []
    filepaths = []

    random.seed(seed)
    for n in range(testset_size):
        # random index to take data from
        r = random.randint(0,len(X)-1)

        X_test.append(X[r])
        del X[r]

        Y_test.append(Y[r])
        del Y[r]

        if fpaths is not None:
            filepaths.append(fpaths[r])
            del fpaths[r]
        
    if ret_filepaths:
        return X, Y, X_test, Y_test, filepaths
    else:
        return X, Y, X_test, Y_test
