import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return nin

def get_nyrout(int nyr, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return nyr

def shifttime(np.ndarray[np.float32_t, ndim=3, negative_indices=False] data not None,
                float missingval,
                np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef unsigned int nfr = data.shape[0]
    cdef unsigned int nyr = data.shape[1]
    cdef unsigned int npx = data.shape[2]

    cdef int offset = <int>params[0] - 1

    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((nfr, nyr, npx), dtype='float32')

    cdef int i, j, k
    cdef int t, nt
    cdef int t1, i1, j1
    cdef float count, ave

    nt = nfr*nyr

    for k in range(npx):
        for t in range(nt):
            t1 = t - offset
            if t1 < 0:
                continue
            i = t % nfr
            j = t / nfr
            i1 = t1 % nfr
            j1 = t1 / nfr
            result[i1,j1,k] = data[i,j,k]

    return result
