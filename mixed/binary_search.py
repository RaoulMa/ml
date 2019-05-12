
def binary_search(target, source, left):
    """Recursive binary search
    Args:
        target (int): target value to search for
        source (list of int): ordered list of integer values
        left: start position in source, usually 0 at beginning
    Returns:
        index (int): index of target in source, otherwise -1
    """
    assert isinstance(target, int)
    assert isinstance(source, list)
    assert isinstance(left, int)

    if len(source) == 0:
        return -1

    center = len(source) // 2
    if target == source[center]:
        return left + center
    if target < source[center]:
        return binary_search(target, source[:center], left)
    else:
        return binary_search(target, source[center+1:], left + center + 1)

source = [0,1,2,3,4,5,6,7,8,9,10]

target = 0
print(binary_search(target, source, 0))

target = 10
print(binary_search(target, source, 0))

target = 5
print(binary_search(target, source, 0))

target = 6
print(binary_search(target, source, 0))

