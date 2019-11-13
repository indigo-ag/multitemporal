import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return nin


def get_nyrout(int nyr, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return params[0] - params[1] + 1


def trimyr(np.ndarray[np.float32_t, ndim=3, negative_indices=False] data not None,
           float missingval,
           np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef unsigned int nfr = data.shape[0]
    cdef unsigned int nyr = data.shape[1]
    cdef unsigned int npx = data.shape[2]

    cdef int nout = get_nout(nfr, params)
    cdef int nyrout = get_nyrout(nyr, params)

    cdef int yr1 = <int>params[0] - 1
    cdef int yr2 = <int>params[1] - 1

    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((nout,nyrout,npx), dtype='float32')

    cdef unsigned int i,j,k,idx

    for k in range(npx):
        jdx = 0
        for j in range(nyr):
            if j>= yr1 and j <= yr2:
                for i in range(nfr):
                    result[i,jdx,k] = data[i,j,k]
                jdx = jdx + 1

    return result
