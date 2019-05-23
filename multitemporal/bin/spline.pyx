import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport M_PI, sin, pow, ceil, log

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def get_nout(int nin, np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):
    return nin


def spline2(np.ndarray[np.float32_t, ndim=1, negative_indices=False] x not None,
            np.ndarray[np.float32_t, ndim=1, negative_indices=False] y not None,
            float yp1, float ypn):

    # used by _spline

    cdef int n = x.shape[0]
    cdef int i, k
    cdef float p, qn, sig, un

    cdef np.ndarray[np.float32_t, ndim=1] y2 = np.zeros(n, dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=1] u = np.zeros(n-1, dtype='float32')

    if (yp1 > 0.99e30):
        y2[0] = 0.0
        u[0] = 0.0
    else:
        y2[0] = -0.5
        u[0] = (3.0/(x[1] - x[0]))*((y[1] - y[0])/(x[1] - x[0]) - yp1)

    for i in range(1, n-1):
        sig = (x[i] - x[i-1])/(x[i+1] - x[i-1])
        p = sig*y2[i-1] + 2.0
        y2[i] = (sig - 1.0)/p
        u[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
        u[i] = (6.0*u[i]/(x[i+1]-x[i-1]) - sig*u[i-1])/p

    if (ypn > 0.99e30):
        qn = 0.0
        un = 0.0
    else:
        qn = 0.0
        un = (3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))

    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.0)

    for k in range(n-2,-1,-1): # (k=n-1;k>=1;k--)
        y2[k] = y2[k]*y2[k+1]+u[k]

    return y2


def splint(np.ndarray[np.float32_t, ndim=1, negative_indices=False] xa not None,
           np.ndarray[np.float32_t, ndim=1, negative_indices=False] ya not None,
           np.ndarray[np.float32_t, ndim=1, negative_indices=False] y2a not None,
           np.ndarray[np.float32_t, ndim=1, negative_indices=False] x not None):

    # used by _spline

    cdef int n = xa.shape[0]
    cdef int nx = x.shape[0]

    cdef np.ndarray[np.float32_t, ndim=1] y = np.zeros(nx, dtype='float32')

    cdef int k, klo, khhi
    cdef int ix
    cdef float h, b, a

    for ix in range(nx):

        klo = 0
        khi = n-1

        while (khi - klo > 1):
            k = (khi + klo) >> 1
            if (xa[k] > x[ix]):
                khi = k
            else:
                klo = k

        h = xa[khi] - xa[klo]
        if (h == 0.0):
            print "Bad xa input to routine splint"

        a = (xa[khi] - x[ix])/h
        b = (x[ix] - xa[klo])/h

        y[ix] = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi])*(h*h)/6.0

    return y


def _spline(np.ndarray[np.float32_t, ndim=1, negative_indices=False] data not None,
           float missingval):

    # calculate the spline

    cdef int n = data.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] result = np.arange(n, dtype='float32')

    cdef int i, kg, kb
    cdef int ng = 0, nb = 0

    for i in range(n):
        if data[i] != missingval:
            ng = ng + 1
        else:
            nb = nb + 1

    cdef np.ndarray[np.float32_t, ndim=1] x = np.arange(ng, dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=1] y = np.arange(ng, dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=1] xfit = np.arange(nb, dtype='float32')

    kg = 0
    kb = 0
    for i in range(n):
        if data[i] != missingval:
            x[kg] = <float>i
            y[kg] = data[i]
            kg = kg + 1
        else:
            xfit[kb] = <float>i
            kb = kb + 1

    # call the previous two helper functions
    y2 = spline2(x, y, 0., 0.)
    yfit = splint(x, y, y2, xfit)

    kg = 0
    for i in range(n):
        if data[i] != missingval:
            result[i] = data[i]
        else:
            result[i] = yfit[kg]
            kg = kg + 1

    return result


def spline(np.ndarray[np.float32_t, ndim=3, negative_indices=False] data not None,
            float missingval,
            np.ndarray[np.float32_t, ndim=1, negative_indices=False] params not None):

    # the actual multitemporal module function

    cdef unsigned int nfr = data.shape[0]
    cdef unsigned int nyr = data.shape[1]
    cdef unsigned long npx = data.shape[2]

    nout = get_nout(nfr, params)
    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((nout,nyr,npx), dtype='float32')

    cdef np.ndarray[np.float32_t, ndim=1] ts = np.zeros(nyr*npx, dtype='float32')
    cdef unsigned int m

    for k in range(npx):

        for j in range(nyr):
            for i in range(nfr):
                m = j*nfr + i
                ts[m] = data[i,j,k]

        tsnew = _spline(ts)

        for j in range(nyr):
            for i in range(nfr):
                m = j*nfr + i
                result[i,j,k] = tsnew[m]

    return result


