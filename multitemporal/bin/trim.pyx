import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return params[1] - params[2] + 1


def window(np.ndarray[np.float32_t, ndim=3, negative_indices=False] data not None,
           float missingval,
           np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef unsigned int nfr = data.shape[0]
    cdef unsigned int nyr = data.shape[1]
    cdef unsigned int npx = data.shape[2]

    cdef int nout = get_nout(nfr, params)

    cdef int day1 = <int>params[0]
    cdef int day2 = <int>params[1]

    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((nout,nyr,npx), dtype='float32')

    cdef unsigned int i,j,k,idx

    for k in range(npx):
        for j in range(nyr):
            idx = 0
            for i in range(nfr):
                if i >= day1 and i <= day2:
                    result[idx,j,k] = data[i,j,k]
                    idx = idx + 1

    return result

