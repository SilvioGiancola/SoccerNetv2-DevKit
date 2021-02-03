
import numpy as np


def getTimestampTargets(labels):

    targets = np.zeros((1,2), dtype='float')


    time_indexes = np.where(labels==0)[0]

    if len(time_indexes) > 0:
        targets[0,0] = 1.0
        targets[0,1] = time_indexes[0]/(labels.shape[0])
    else:
        return targets

    return targets