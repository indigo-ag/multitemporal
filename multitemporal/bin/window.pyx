import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return 1


def window(np.ndarray[np.float32_t, ndim=3, negative_indices=False] data not None,
                float missingval,
                np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef unsigned int nfr = data.shape[0]
    cdef unsigned int nyr = data.shape[1]
    cdef unsigned int npx = data.shape[2]

    cdef int nout = get_nout(nfr, params)

    cdef int day1 = <int>params[0]
    cdef float day2 = <int>params[1]

    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((nout,nyr,npx), dtype='float32')
    #cdef np.ndarray[np.float32_t, ndim=1] count = np.zeros(nout, dtype='float32')

    cdef float count
    cdef unsigned int i,j,k

    for k in range(npx):
        for j in range(nyr):
            count = 0.0
            for i in range(nfr):
                if i >= day1 and i <= day2:
                    if data[i, j, k] != missingval:
                        result[0,j,k] = result[0,j,k] + data[i,j,k]
                        count = count + 1.0

            for i1 in range(nfr):
                if count > 0.0:
                    result[0,j,k] = result[0,j,k]/count
                else:
                    result[0,j,k] = missingval

    return result
