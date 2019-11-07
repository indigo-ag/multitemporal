import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return nin

def linearmodel(np.ndarray[np.float32_t, ndim=1] x not None,
                np.ndarray[np.float32_t, ndim=1] y not None,
                float missingval):

    cdef int nt = x.shape[0]
    cdef float sumx = 0.0
    cdef float sumy = 0.0
    cdef float sumxy = 0.0
    cdef float sumxx = 0.0
    cdef float sumyy = 0.0
    cdef float n = 0.0
    cdef float slope = missingval
    cdef float intercept = missingval
    cdef float corrcoef = missingval
    cdef float tstat = missingval
    cdef float denom1, denom2, denom3
    cdef int i
    for i in range(nt):
        if x[i] != missingval and y[i] != missingval:
            sumx = sumx + x[i]
            sumy = sumy + y[i]
            sumxy = sumxy + (x[i]*y[i])
            sumxx = sumxx + x[i]*x[i]
            sumyy = sumyy + y[i]*y[i]
            n = n + 1.0
    if n > 0.0:
        meanx = sumx/n
        meany = sumy/n
        denom1 = (sumxx - sumx*meanx)
        if denom1 != 0.0:
            slope = (sumxy - sumx*meany)/denom1
            intercept = meany - slope*meanx
            denom2 = sqrt((n*sumxx - sumx*sumx)*(n*sumyy - sumy*sumy))
            if denom2 != 0.0:
                corrcoef = (n*sumxy - sumx*sumy)/denom2
                denom3 = (1.0 - corrcoef*corrcoef)
                if denom3 != 0.0:
                    tstat = corrcoef*sqrt((n - 2.0)/denom3)

    return slope, intercept, corrcoef, tstat


def combine(np.ndarray[np.float32_t, ndim=4, negative_indices=False] data not None,
            float missingval,
            np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    cdef int nbd = data.shape[0]
    cdef int nfr = data.shape[1]
    cdef int nyr = data.shape[2]
    cdef int npx = data.shape[3]

    cdef int nout = get_nout(nfr, params)

    cdef np.ndarray[np.float32_t, ndim=1] x = np.zeros(nfr*nyr, dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=1] y = np.zeros(nfr*nyr, dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros(
        (nout,nyr,npx), dtype='float32')

    cdef float relweight = params[0]

    cdef int i,j,k
    cdef float count
    cdef float w0, w1, sumw

    for k in range(npx):
        count = 0.0
        for j in range(nyr):
            for i in range(nfr):
                if data[0,i,j,k] != missingval and data[1,i,j,k] != missingval:
                    x[i] = data[1,i,j,k]
                    y[i] = data[0,i,j,k]
                    count = count + 1.0
                else:
                    x[i] = missingval
                    y[i] = missingval

        if count <= 2.0:
            tstat = missingval
        else:
            slope, intercept, corrcoef, tstat = linearmodel(x, y, missingval)

        for j in range(nyr):
            for i in range(nfr):

                # both are missing
                if data[0,i,j,k] == missingval and data[1,i,j,k] == missingval:
                    result[i,j,k] = missingval

                # target is not missing and covariate is missing
                elif data[0,i,j,k] != missingval and data[1,i,j,k] == missingval:
                    result[i,j,k] = data[0,i,j,k]

                # target is missing and covariate is not missing
                elif data[0,i,j,k] == missingval and data[1,i,j,k] != missingval:
                    if tstat == missingval:
                        result[i,j,k] = missingval
                    else:
                        result[i,j,k] = slope*data[1,i,j,k] + intercept

                # target is not missing and covariate is not missing
                elif data[0,i,j,k] != missingval and data[1,i,j,k] != missingval:
                    if tstat == missingval:
                        result[i,j,k] = data[0,i,j,k]
                    else:
                        w0 = relweight
                        w1 = 1.0 - relweight
                        result[i,j,k] = w0*data[0,i,j,k] + w1*(slope*data[1,i,j,k] + intercept)

    return result
