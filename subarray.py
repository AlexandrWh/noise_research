import numpy as np
from tqdm import tqdm

def find_max_sub_array(arr):
    arr = list(arr)
    n = len(arr)
    arr = [0] + arr
    prefixes = [sum(arr[0:i+1]) for i in range(n+1)]  # O(n)
    mins = [0]
    _min = 0

    for i in range(2, n+1):
        if prefixes[i - 1] < prefixes[_min]:
            _min = i - 1
            mins.append(i - 1)
        else:
            mins.append(_min)

    _max = (0, prefixes[1] - prefixes[mins[0]])

    for i in range(1, n):
        if _max[1] < prefixes[i+1] - prefixes[mins[i]]:
            _max = (i, prefixes[i+1] - prefixes[mins[i]])

    return mins[_max[0]], _max[0], arr[mins[_max[0]]+1:_max[0]+2]


def test_subarray():
    print('testing subarray...')
    for i in tqdm(range(1000)):
        random_arr = np.random.randint(50, size=100) - 25
        subsums = np.array([[np.sum(random_arr[i:j]) for j in np.arange(100) + 1] for i in np.arange(100)])
        true = np.unravel_index(np.argmax(subsums, axis=None), subsums.shape)

        predict = find_max_sub_array(random_arr)[:2]

        assert true[0] == predict[0] and true[1] == predict[1]