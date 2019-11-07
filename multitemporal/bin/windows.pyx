import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return len(params)/2


def windows(np.ndarray[np.float32_t, ndim=3, negative_indices=False] data not None,
                float missingval,
                np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef unsigned int nfr = data.shape[0]
    cdef unsigned int nyr = data.shape[1]
    cdef unsigned int npx = data.shape[2]

    cdef int nout = get_nout(nfr, params)

    cdef int day1
    cdef int day2

    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((nout,nyr,npx), dtype='float32')

    cdef np.ndarray[np.float32_t, ndim=1] count = np.zeros(nout, dtype='float32')

    cdef unsigned int i,j,k,p

    for k in range(npx):
        for j in range(nyr):
            for i in range(nfr):

                for p in range(nout):
                    day1 = <int>params[p*2]
                    day2 = <int>params[p*2+1]

                    if i >= day1 and i <= day2:
                        if data[i, j, k] != missingval:
                            result[p,j,k] = result[p,j,k] + data[i,j,k]
                            count[p] = count[p] + 1.0

            for p in range(nout):
                if count[p] > 0.0:
                    result[p,j,k] = result[p,j,k]/count[p]
                else:
                    result[p,j,k] = missingval
                count[p] = 0.0

    return result
