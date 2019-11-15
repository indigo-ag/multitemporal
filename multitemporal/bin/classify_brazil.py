import pickle

import numpy as np
import sklearn

from pdb import set_trace


def get_nout(nin, params):
    return 1


def get_nyrout(nyr, params):
    return nyr


def classify_brazil(data, missingval, params):

    nfr = data.shape[0]
    nyr = data.shape[1]
    npx = data.shape[2]

    nout = get_nout(nfr, params)
    nyrout = get_nyrout(nyr, params)

    output = np.zeros((nout, nyrout, npx), dtype=np.float32)

    model_file = params[0]
    model = pickle.load(open(model_file, 'rb'))

    for k in range(npx):
        for j in range(nyr):
            feature = data[:,j,k]
            if feature[0] > 0:
                output[0,j,k] = model.predict(feature.reshape(1,15))

    return output
