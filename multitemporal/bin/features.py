import pickle

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths, peak_prominences
import sklearn


from pdb import set_trace


VARNAMES = [
    'evi2_mean_gapfill_smooth_median',
    'peak1_height',
    'peak1_dos',
    'peak1_dos_start',
    'peak1_height_p75',
    'peak1_ndays_start2peak_p25',
    'peak1_ndays_start2peak_p100',
    'peak1_ndays_peak2end_p25',
    'peak1_ndays_peak2end_p100',
    'peak1_ndpi_p25',
    'peak1_ndpi_p75',
    'peak2_height_p100',
    'peak2_ndays_start2peak_p100',
    'peak2_ndpi_p50',
    'peak2_ndpi_p100'
]


def get_nout(nin, params):
    return len(VARNAMES)


def get_nyrout(nyr, params):
    return nyr


def calc_timeseries_features(
    ts,
    height_thresh=0.3,
    distance=45,
    prominence=0.1,
    npeaks_max=2):

    # :Provides simple metrics for the top 3 peaks in a time series.
    # :param df: data frame with daily 'value_col' per geo_id. Function designed to work on gap-filled, smoothed data.
    # :param value_col: name(s) of column on which to calculate features.
    # :param height_thresh: Required height of peaks. See scipy.signal.find_peaks docs for more info.
    # :param distance: Required minimal horizontal distance (>= 1) in samples (i.e., days) between neighbouring peaks. See scipy.signal.find_peaks docs for more info.
    # :param prominence: Required prominence of peaks. See scipy.signal.find_peaks docs for more info.
    # :param npeaks_max: Maximum number of peaks for which to calculate features. Not yet really implemented. Currently, it takes the first npeaks_max peaks. Future = take the npeaks_max largest peaks. Use None to characterize all peaks.  
    # In addition to the 'value_col,' the function expects df to have geo_id, metric_date, crop (can be None), and season columns.
    # :return: pd.DataFrame

    result = np.zeros(len(VARNAMES), dtype=np.float32)

    ts = savgol_filter(ts, 31, 1)

    # Peak detection
    pos_peaks_detect = find_peaks(ts, height=height_thresh, distance=distance, prominence=prominence)
    pos_peaks = pos_peaks_detect[0]
    npeaks = len(pos_peaks_detect[0])

    if npeaks == 0:
        return result

    # Peak characterization
    widths_p25 = peak_widths(ts, pos_peaks, rel_height=0.25)
    widths_p50 = peak_widths(ts, pos_peaks, rel_height=0.50)
    widths_p75 = peak_widths(ts, pos_peaks, rel_height=0.75)
    widths_p100 = peak_widths(ts, pos_peaks, rel_height=1)

    result[0] = np.median(ts)
    result[1] = pos_peaks_detect[1]['peak_heights'][0]
    result[2] = pos_peaks_detect[0][0]
    result[3] = pos_peaks_detect[1]['left_bases'][0]
    result[4] = widths_p75[1][0]
    result[5] = pos_peaks_detect[0][0] - widths_p25[2][0]
    result[6] = pos_peaks_detect[0][0] - widths_p100[2][0]
    result[7] = widths_p25[3][0] - pos_peaks_detect[0][0]
    result[8] = widths_p100[3][0] - pos_peaks_detect[0][0]
    result[9] = (widths_p25[3][0] - 2*pos_peaks_detect[0][0] + widths_p25[2][0])\
                    / (widths_p25[3][0] - widths_p25[2][0])
    result[10] = (widths_p75[3][0] - 2*pos_peaks_detect[0][0] + widths_p75[2][0])\
                    / (widths_p75[3][0] - widths_p75[2][0])

    if npeaks > 1:
        result[11] = widths_p75[1][1]
        result[12] = pos_peaks_detect[0][1] - widths_p100[2][1]
        result[13] = (widths_p50[3][1] - 2*pos_peaks_detect[0][1] + widths_p50[2][1])\
                        / (widths_p50[3][1] - widths_p50[2][1])
        result[14] = (widths_p100[3][1] - 2*pos_peaks_detect[0][1] + widths_p100[2][1])\
                        / (widths_p100[3][1] - widths_p100[2][1])
    else:
        result[11] = 0.0
        result[12] = 0.0
        result[13] = 0.0
        result[14] = 0.0

    return result


def features(data, missingval, params):

    nfr = data.shape[0]
    nyr = data.shape[1]
    npx = data.shape[2]

    nout = get_nout(nfr, params)
    nyrout = get_nyrout(nyr, params)

    output = np.zeros((nout, nyrout, npx), dtype=np.float32)

    for k in range(npx):
        for j in range(nyr):
            output[:,j,k] = calc_timeseries_features(data[:,j,k])

    return output
