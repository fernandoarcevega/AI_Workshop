import numpy as np
#**************************************************
def HoldOut(Y, p):
    c = np.unique(Y)  # Class labels
    nc = len(c)       # Number of classes
    n = len(Y)        # Number of samples
    tt = np.zeros(n, dtype=bool)  # Indices of test data
    for i in range(nc):
        id = (Y == c[i])   # Detect samples of the ith class
        ni = np.sum(id)    # Number of samples in the ith class
        fd = np.where(id)[0]  # Indices of samples in the ith class
        rd = np.random.choice(ni, size=round(p * ni), replace=False)  # Get random indices for test data
        tt[fd[rd]] = True
    tr = ~tt  # Indices of training data
    return tr, tt
#**************************************************
def zscorenorm(X, stats=None):
    if stats is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std, (mean, std)
    else:
        mean, std = stats
        return (X - mean) / std