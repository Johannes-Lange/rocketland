from config import STATE_MINMAX
from copy import deepcopy


def norm_minmax(val, minmax):
    low, high = minmax[0], minmax[1]
    if low < 0:
        val += abs(low)
        high += abs(low)
        low += abs(low)
    return (val - low) / (high - low)


def normalize_state(state):
    ret = deepcopy(state)
    for i in range(ret.shape[0]):
        ret[i] = norm_minmax(ret[i], STATE_MINMAX[i])
    return ret
