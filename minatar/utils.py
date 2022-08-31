def try2jit(f):
    try:
        from numba import njit
        return njit(f, cache=True)
    except:
        return f

# much much faster than np.random.choice
# grab an array element assuming uniform dist
def choice(arr, rng):
    idx = rng.randint(len(arr))
    return arr[idx]

# much much faster than np.random.choice
# sample an index according to probs
def sample(arr, rng):
    r = rng.random()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we should never get here
    return len(arr) - 1
