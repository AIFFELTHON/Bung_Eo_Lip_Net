import numpy
import doctest

# 편집 거리(Edit Distance) 문제 - 동적 계획
def wer(r, h):
    """
    Source: https://martin-thoma.com/word-error-rate-calculation/

    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """

    # r, h = [who, is, there], [is, there]

    # initialisation

    # [0 0 0 0 0 0 0 0 0 0 0 0] (12,)
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)

    # [[0 0 0]
    # [0 0 0]
    # [0 0 0]
    # [0 0 0]] (4, 3)
    d = d.reshape((len(r)+1, len(h)+1))

    # [[0 1 2]
    # [1 0 0]
    # [2 0 0]
    # [3 0 0]]
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:  # 0번 행 초기화
                d[0][j] = j
            elif j == 0:  # 0번 열 초기화
                d[i][0] = i

    # computation
    # [[0 1 2]
    # [1 1 2]
    # [2 1 2]
    # [3 2 1]]
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1  # 대체
                insertion    = d[i][j-1] + 1  # 삽입
                deletion     = d[i-1][j] + 1  # 삭제
                d[i][j] = min(substitution, insertion, deletion)  # 편집 거리 저장

    return d[len(r)][len(h)]  # d[3][2] = 1

def wer_sentence(r, h):
    # r, h = "who is there", "is there"
    return wer(r.split(), h.split())

if __name__ == "__main__":
    doctest.testmod()
    result = wer_sentence("who is there", "is there")
    print(result)