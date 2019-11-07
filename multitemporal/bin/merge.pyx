import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return nin


def merge(np.ndarray[np.float32_t, ndim=4, negative_indices=False] data not None,
            float missingval,
            np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef int nbd = data.shape[0]
    cdef int nfr = data.shape[1]
    cdef int nyr = data.shape[2]
    cdef int npx = data.shape[3]

    cdef int nout = get_nout(nfr, params)

    cdef np.ndarray[np.float32_t, ndim=1] x = np.zeros(nfr*nyr, dtype='float32')

    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros(
        (nout,nyr,npx), dtype='float32')

    # cdef float relweight = params[0]

    cdef int i,j,k
    cdef float count
    cdef float value

    for k in range(npx):
        for j in range(nyr):
            for i in range(nfr):

                count = 0.0
                value = 0.0

                if data[0,i,j,k] != missingval and data[2,i,j,k] == 1:
                    value = value + data[0,i,j,k]
                    count = count + 1.0

                if data[1,i,j,k] != missingval and data[3,i,j,k] == 1:
                    value = value + data[1,i,j,k]
                    count = count + 1.0

                if count == 0.0:
                    value = missingval
                else:
                    value = value/count

                result[i,j,k] = value

    return result
