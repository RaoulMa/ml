def solution(A, K):

    shift = K % len(A)

    if shift == 0:
        return A

    shifted_array = []

    for i in range(-shift, len(A) - shift):
        shifted_array.append(A[i])

    return shifted_array


