import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths, peak_prominences


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

    contour_heights = ts[pos_peaks] - pos_peaks_detect[1]['prominences']

    # Season-wide metrics
    value_col = "evi2_mean_gapfill_smooth"
    ts_out = {}

    ts_out['npeaks'] = npeaks
    ts_out['ndays_total_season'] = pos_peaks_detect[1]['right_bases'][-1] - pos_peaks_detect[1]['left_bases'][0]

    ts_out[f'{value_col}_max'] = ts.max()
    ts_out[f'{value_col}_median'] = np.median(ts)
    ts_out[f'{value_col}_stddev'] = ts.std()

    for npeak in list(range(npeaks)):

        # Height of peak
        ts_out[f'peak{npeak+1}_height'] = pos_peaks_detect[1]['peak_heights'][npeak]
        # day of season that peak occurs (will be lat dependent)
        ts_out[f'peak{npeak+1}_dos'] = pos_peaks_detect[0][npeak]
        # Prominence of peak (curve may not start at 0, so this is a slight play on height)
        ts_out[f'peak{npeak+1}_prominence'] = pos_peaks_detect[1]['prominences'][npeak]
        # Day of season start and end
        ts_out[f'peak{npeak+1}_dos_start'] = pos_peaks_detect[1]['left_bases'][npeak]
        ts_out[f'peak{npeak+1}_dos_end'] = pos_peaks_detect[1]['right_bases'][npeak]
        # number of days within this peak
        ts_out[f'peak{npeak+1}_ndays'] = ts_out[f'peak{npeak+1}_dos_end'] - ts_out[f'peak{npeak+1}_dos_start']
        # number of days within this peak at various percentiles (getting at shape)
        ts_out[f'peak{npeak+1}_ndays_p25'] = widths_p25[0][npeak]
        ts_out[f'peak{npeak+1}_ndays_p50'] = widths_p50[0][npeak]
        ts_out[f'peak{npeak+1}_ndays_p75'] = widths_p75[0][npeak]
        ts_out[f'peak{npeak+1}_ndays_p100'] = widths_p100[0][npeak]
        # height (e.g., EVI2) at each percentile
        ts_out[f'peak{npeak+1}_height_p25'] = widths_p25[1][npeak]
        ts_out[f'peak{npeak+1}_height_p50'] = widths_p50[1][npeak]
        ts_out[f'peak{npeak+1}_height_p75'] = widths_p75[1][npeak]
        ts_out[f'peak{npeak+1}_height_p100'] = widths_p100[1][npeak]
        # ndays from the start of the curve (at various p values) to the peak.
        ts_out[f'peak{npeak+1}_ndays_start2peak_p25'] = ts_out[f'peak{npeak+1}_dos'] - widths_p25[2][npeak]
        ts_out[f'peak{npeak+1}_ndays_start2peak_p50'] = ts_out[f'peak{npeak+1}_dos'] - widths_p50[2][npeak]
        ts_out[f'peak{npeak+1}_ndays_start2peak_p75'] = ts_out[f'peak{npeak+1}_dos'] - widths_p75[2][npeak]
        ts_out[f'peak{npeak+1}_ndays_start2peak_p100'] = ts_out[f'peak{npeak+1}_dos'] - widths_p100[2][npeak]
        # ndays from peak to end of curve
        ts_out[f'peak{npeak+1}_ndays_peak2end_p25'] = widths_p25[3][npeak] - ts_out[f'peak{npeak+1}_dos']
        ts_out[f'peak{npeak+1}_ndays_peak2end_p50'] = widths_p50[3][npeak] - ts_out[f'peak{npeak+1}_dos']
        ts_out[f'peak{npeak+1}_ndays_peak2end_p75'] = widths_p75[3][npeak] - ts_out[f'peak{npeak+1}_dos']
        ts_out[f'peak{npeak+1}_ndays_peak2end_p100'] = widths_p100[3][npeak] - ts_out[f'peak{npeak+1}_dos']
        # Normalized diff of ndays front of peak line vs. back of peak line - calling it the normalized difference peak index??
        # Shoot - ndpi apparently already taken for the phytoplankton index. Need to come up w/ new name if this thing actually helps!
        ts_out[f'peak{npeak+1}_ndpi_p25'] = (ts_out[f'peak{npeak+1}_ndays_peak2end_p25'] - ts_out[f'peak{npeak+1}_ndays_start2peak_p25']) / (ts_out[f'peak{npeak+1}_ndays_peak2end_p25'] + ts_out[f'peak{npeak+1}_ndays_start2peak_p25'])
        ts_out[f'peak{npeak+1}_ndpi_p50'] = (ts_out[f'peak{npeak+1}_ndays_peak2end_p50'] - ts_out[f'peak{npeak+1}_ndays_start2peak_p50']) / (ts_out[f'peak{npeak+1}_ndays_peak2end_p50'] + ts_out[f'peak{npeak+1}_ndays_start2peak_p50'])
        ts_out[f'peak{npeak+1}_ndpi_p75'] = (ts_out[f'peak{npeak+1}_ndays_peak2end_p75'] - ts_out[f'peak{npeak+1}_ndays_start2peak_p75']) / (ts_out[f'peak{npeak+1}_ndays_peak2end_p75'] + ts_out[f'peak{npeak+1}_ndays_start2peak_p75'])
        ts_out[f'peak{npeak+1}_ndpi_p100'] = (ts_out[f'peak{npeak+1}_ndays_peak2end_p100'] - ts_out[f'peak{npeak+1}_ndays_start2peak_p100']) / (ts_out[f'peak{npeak+1}_ndays_peak2end_p100'] + ts_out[f'peak{npeak+1}_ndays_start2peak_p100'])

    for i,varname in enumerate(VARNAMES):
        try:
            result[i] = ts_out[varname]
        except KeyError:
            result[i] = 0

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
