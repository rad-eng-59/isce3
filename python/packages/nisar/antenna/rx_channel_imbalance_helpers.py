from __future__ import annotations
from warnings import warn
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np
import h5py

from nisar.products.readers.Raw import Raw
from nisar.antenna import get_calib_range_line_idx, CalPath
from isce3.core import speed_of_light


@dataclass(frozen=True)
class RX_CHANNEL_IMBALANCE_PRODUCT:
    """
    RX channel imbalance product extracted from LNA/CALTONE ratio
    for a certain frequency band and polarization.

    Attributes
    ----------
    lna_caltone_ratio: np.ndarray(complex)
        Peak-normalized complex LNA/CALTONE ratio over all RXs
    ntap_dominant: np.ndarray(int)
        Dominant tap number, a value within [1,3] over all RXs.
    time_delays_sec: np.ndarray(float)
        Time delays from the phase of outlier qFSP in seconds for all RXs.
    max_amp_ratio: float
        Max amplitude ratio used in peak normalizing `lna_caltone_ratio`.

    """
    lna_caltone_ratio: np.ndarray
    ntap_dominant: np.ndarray
    time_delays_sec: np.ndarray
    max_amp_ratio: float

    def __post_init__(self):
        # XXX Size of all arrays must be 12 for L-band NISAR but
        # not enforced due to failure of special cases such as unit test
        if (self.lna_caltone_ratio.size != self.ntap_dominant.size
                != self.time_delays_sec.size):
            raise ValueError('The size of all arrays must be equal!')


def compute_all_rx_channel_imbalances_from_l0b(
        l0b_file: str | Raw,
        *,
        caltone_freq: float = 1214.88e6,
        freq_band: str | None = None,
        txrx_pol: str | None = None
) -> Dict[Tuple[str, str], RX_CHANNEL_IMBALANCE_PRODUCT]:
    """
    Compute 12 complex RX channel imbalance based on LNA/CALTONE ratio
    for over all bands and polarizations. The bands and polarizations are
    used as dictionary keys in the form of [freq_band, txrx_pol].

    Also report the dominant tap number our of 3 for LNA three-tap
    correlator as well as detected relative time delays for all RX channels
    for debugging purposes.

    Parameters
    ----------
    l0b_file : str or nisar.products.readers.Raw
        L0B filename or Raw object
    caltone_freq : float, default=1214.88e6
        Caltone frequency in Hz.
    freq_band : str, optional
        "A" or "B". Default is all.
    txrx_pol: str, optional
        TR pol in `freq_band` such as "HH", "HV", etc.
        Default is all.

    Returns
    -------
    dict:
        A dict with keys (freq_band, txrx_pol) and values of type
        `RX_CHANNEL_IMBALANCE_PRODUCT`

    """
    if isinstance(l0b_file, str):
        raw = Raw(hdf5file=l0b_file)
    else:
        raw = l0b_file
    frq_pols = raw.polarizations
    # get freq_bands and txrx_pols
    if freq_band is not None:
        frq_pols = {freq_band: frq_pols[freq_band]}
    if txrx_pol is not None:
        frq_pols = {f: [txrx_pol] for f in frq_pols if txrx_pol in frq_pols[f]}

    out = dict()
    for freq_band in frq_pols:
        for txrx_pol in frq_pols[freq_band]:
            (lna_caltone_ratio, n_tap_dominant, time_delays, max_ratio
             ) = compute_rx_channel_imbalance(
                raw,
                freq_band,
                txrx_pol,
                caltone_freq=caltone_freq
            )
            out[freq_band, txrx_pol] = RX_CHANNEL_IMBALANCE_PRODUCT(
                lna_caltone_ratio=lna_caltone_ratio,
                ntap_dominant=n_tap_dominant,
                time_delays_sec=time_delays,
                max_amp_ratio=max_ratio
            )
    return out


def compute_rx_channel_imbalance(
        raw: Raw,
        freq_band: str,
        txrx_pol: str,
        caltone_freq: float = 1214.88e6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute 12 complex RX channel imbalance based on LNA/CALTONE ratio
    for a desired frequency band and TR polarization.

    Also report the dominant tap number our of 3 for LNA three-tap
    correlator as well as detected relative time delays for all RX channels
    for debugging purposes.

    Returns
    -------
    lna_caltone_ratio: np.ndarray(complex)
        Peak-normalized complex LNA/CALTONE ratio over all 12 RXs
    n_tap_dominant: np.ndarray(int)
        Dominant tap number, a value within [1,3] over all 12 RXs.
    time_delays: np.ndarray(float)
        Time delays from the phase of qFSP outlier
    max_ratio : float
        Report peak power among all channels used for amplitude
        normalization of RX channel imbalances.

    """
    lna_mean, n_tap_dominant = get_lna_cal_mean(
        raw, freq_band, txrx_pol)
    # get caltone mean over all RX channels
    caltone_mean = get_caltone_mean(raw, freq_band, txrx_pol)
    # Get complex ratio LNA/Caltone over all channels
    lna_caltone_ratio = lna_mean / caltone_mean
    # correct the ratio for the second band if necessary
    lna_caltone_ratio, time_delays = correct_lna_caltone_ratio_for_second_band(
        lna_caltone_ratio,
        raw,
        freq_band,
        txrx_pol,
        caltone_freq=caltone_freq
    )
    # peak normalized
    max_ratio = np.nanmax(abs(lna_caltone_ratio))
    if not np.isclose(max_ratio, 0):
        lna_caltone_ratio /= max_ratio
    return lna_caltone_ratio, n_tap_dominant, time_delays, max_ratio


def parse_chirp_corr_from_hrt_qfsp(
        raw: Raw,
        txrx_pol: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse three-tap chirp correlator array with shape (lines, 12, 3)
    as well ass cal type with shape (lines,) from HRT QFSP.
    """
    # get HRT path
    hrt_path = raw.TelemetryPath.replace('low', 'high')
    qfsp_path = f'{hrt_path}/tx{txrx_pol[0]}/rx{txrx_pol[1]}/QFSP'
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        # loop over three qfsp
        for i_qfsp in range(3):
            p_qfsp = f'{qfsp_path}{i_qfsp}'
            # loop over 4 channels per qfsp:
            for nn in range(4):
                i_chn = nn + i_qfsp * 4
                n_rx = i_chn + 1
                # loop over 3 taps
                for i_tap in range(3):
                    n_tap = i_tap + 1
                    # form the path to the dataset per I and Q
                    # use RX pol!
                    p_ds_i = (f'{p_qfsp}/CHIRP_CORRELATOR_I{n_tap}_'
                              f'{txrx_pol[1]}{n_rx:02d}')
                    p_ds_q = (f'{p_qfsp}/CHIRP_CORRELATOR_Q{n_tap}_'
                              f'{txrx_pol[1]}{n_rx:02d}')
                    # initialize the 3-D array, lines by 12 by 3
                    if i_qfsp == nn == i_tap == 0:
                        # XXX get caltype from the very first qFSP assuming
                        # it is qFSP independent!
                        p_type = f'{p_qfsp}/CP_CAL_TYPE_{txrx_pol[1]}{i_qfsp}'
                        # XXX Following Try/exception block is added to
                        # support old sim L0B products lacking HRT!
                        try:
                            ds_cal_type = f5[p_type]
                        except KeyError:
                            warn(f'Missing dataset "{p_type}" in '
                                 f'"{raw.filename}". LNA CAL values '
                                 'from co-pol will be used instead. '
                                 'Results may be invalid!')
                            freq_band = [f for f in raw.frequencies if
                                         txrx_pol in raw.polarizations[f]][0]
                            chp_cor = raw.getChirpCorrelator(
                                freq_band, txrx_pol[0])
                            cal_type = raw.getCalType(freq_band, txrx_pol[0])
                            return chp_cor, cal_type
                        else:
                            cal_type = ds_cal_type[()].astype(CalPath)
                            # initialize the 3-D array for chirp correlator
                            num_lines = f5[p_ds_i].size
                            chp_cor = np.ones((num_lines, 12, 3), dtype='c8')
                    chp_cor[:, i_chn, i_tap].real = f5[p_ds_i][()]
                    chp_cor[:, i_chn, i_tap].imag = f5[p_ds_q][()]
    return chp_cor, cal_type


def get_lna_cal_mean(
    raw: Raw,
    freq_band: str,
    txrx_pol: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mean complex LNA values and dominant tap
    numbers within [1, 2, 3] for all channels
    """
    if txrx_pol[0] == txrx_pol[1]:
        chp_cor = raw.getChirpCorrelator(freq_band, txrx_pol[0])
        cal_type = raw.getCalType(freq_band, txrx_pol[0])
    else:  # get x-pol chirp correlator from HRT
        chp_cor, cal_type = parse_chirp_corr_from_hrt_qfsp(
            raw,
            txrx_pol
        )
    n_rxs = chp_cor.shape[1]
    _, idx_byp, idx_lna, _ = get_calib_range_line_idx(cal_type)
    if len(idx_lna) == 0:
        warn('No LNA CAL to represent RX! Use BYPASS Cal instead!')
        if len(idx_byp) == 0:
            # XXX to avoid failure in unit test or very short L0B
            # lacking LNA/BYP CAL datasets, a warning will be issued
            # and the values will all be set to unity!
            warn('No LNA or BYPASS CAL! LNA mean will be all unity. '
                 'The results will be invalid!')
            lna_mean = np.ones(n_rxs, dtype='c8')
            n_tap_dominant = np.full(n_rxs, fill_value=2)
            return lna_mean, n_tap_dominant
        idx_lna = idx_byp
    # get  LNA for all three taps (or BYPASS)
    lna_mean_tap3 = np.zeros((3, n_rxs), dtype='c16')
    for nn in range(3):
        lna_cal = chp_cor[idx_lna, :, nn]
        # get complex mean for all RX channels
        lna_mean_tap3[nn] = _mean_2d(lna_cal)
    # get dominat taps
    abs_lna_mean_tap3 = abs(lna_mean_tap3)
    idx_lna_taps = np.nanargmax(abs_lna_mean_tap3, axis=0)
    amp_lna_mean = np.zeros(n_rxs)
    for nn in range(n_rxs):
        amp_lna_mean[nn] = abs_lna_mean_tap3[idx_lna_taps[nn], nn]
    _check_if_zero(amp_lna_mean, msg=f'{txrx_pol[0]}-pol LNA Cal')
    # get the phase part at a fixed common tap rather than dominant one
    phs_lna_mean = np.angle(lna_mean_tap3[1])
    # form complex lna
    lna_mean = amp_lna_mean * np.exp(1j * phs_lna_mean)
    n_tap_dominant = idx_lna_taps + 1
    return lna_mean, n_tap_dominant


def correct_lna_caltone_ratio_for_second_band(
        lna_caltone_ratio: np.ndarray,
        raw: Raw,
        freq_band: str,
        txrx_pol: str,
        caltone_freq: float = 1214.88e6
) -> Tuple[np.ndarray, np.ndarray]:
    # XXX check if product from the second band so we can modify
    # the results from the first band only if there is a
    # relative delay offset in one of qFSP vs others, that is
    # one of the qFSP is an outlier due to  ADC clock/delay issue
    # check if there is delay anomaly among three qFSP
    fc_a, _, _, _ = raw.getChirpParameters('A', txrx_pol[0])
    # get diff of chirp (band=A) and caltone freq for delay detection
    dif_chirp_caltone_freq = fc_a - caltone_freq
    time_delay = _get_qfsp_delay_anomaly(
        lna_caltone_ratio, dif_chirp_caltone_freq)
    if _is_product_from_second_band(raw, freq_band, txrx_pol):
        warn(f'correcting LNA/CALTONE for band={freq_band} and pol={txrx_pol}')
        # if there is then get diff of frequency bands A dn B
        # to be used to correct phase from A for B
        fc_b, _, _, _ = raw.getChirpParameters('B', txrx_pol[0])
        phs_adj = 2 * np.pi * (fc_b - fc_a) * time_delay
        # correct the LNA/CALTONE by delay amount via phase if any.
        lna_caltone_ratio *= np.exp(1j * phs_adj)
    return lna_caltone_ratio, time_delay


def get_caltone_mean(
        raw: Raw,
        freq_band: str,
        txrx_pol: str
) -> np.ndarray:
    # now get caltone always from swath
    caltone = raw.getCaltone(freq_band, txrx_pol)
    caltone_mean = _mean_2d(caltone)
    _check_if_zero(caltone_mean, msg=f'{txrx_pol}-pol Caltone')
    return caltone_mean


def _is_product_from_second_band(
        raw: Raw,
        freq_band: str,
        txrx_pol: str):
    """
    Determine whether the produt is avolable on both bands
    and it is from the second band.
    """
    if freq_band == "B" and len(raw.frequencies) == 2:
        if txrx_pol in raw.polarizations['A']:
            return True
        return False
    return False


def _get_qfsp_delay_anomaly(
        lna_caltone_ratio: np.ndarray,
        dif_chirp_caltone_freq: float,
        adc_clock: float = 240e6) -> np.ndarray:
    """
    get time delays for a qfSP with phase anomaly only for
    12 channel NISAR L-band product.
    For other case, it will be set to zero!
    """
    if lna_caltone_ratio.size == 12:
        # group them into three 4-channels, one per qFSP
        lna2cal_ratio = lna_caltone_ratio.reshape(3, 4)
        # get unwrap phase across 4 channels per qFSP (radians)
        lna2cal_phs = np.unwrap(np.angle(lna2cal_ratio), axis=1)
        # get median phase per qfsp, total 3 phase values (radians)
        # and then unwrap three values
        qfps_phs = np.unwrap(np.nanmedian(lna2cal_phs, axis=1))
        # use median among all three to be used as a reference to
        # catch a single outlier
        phs_ref = np.median(qfps_phs)
        # phase due to ADC delay
        phs_adc_delay = 2 * np.pi * dif_chirp_caltone_freq / adc_clock
        n_delay_qfsp = np.round((qfps_phs - phs_ref) / phs_adc_delay)
        # now repeat sample delay 4x per qFSP
        n_delays = np.repeat(
            n_delay_qfsp[:, np.newaxis], repeats=4, axis=1).ravel()
        time_delays = n_delays / adc_clock
    else:
        time_delays = np.zeros(lna_caltone_ratio.size)
    return time_delays


def _mean_2d(data: np.ndarray, perc: float = 0.0) -> np.asarray:
    """
    Compute mean within percentile [perc, 100-perc],
    of a 2-D complex array with shape (rangelines, channels)
    due to bad telemetry.
    """
    # or simply np.nanmean(data, axis=0)
    d = np.sort(np.abs(data), axis=0)
    q1_all, q3_all = np.percentile(d, q=[perc, 100 - perc], axis=0)
    mean_all = []
    for cc, (q1, q3) in enumerate(zip(q1_all, q3_all)):
        data_q1_q3 = data[(d[:, cc] >= q1) & (d[:, cc] <= q3), cc]
        mean_all.append(np.nanmean(data_q1_q3))
    return np.asarray(mean_all)


def _check_if_zero(arr: np.ndarray, msg: str):
    is_zero = np.isclose(arr, 0)
    if is_zero.all():
        # XXX to avoid unit test failure and old sim L0B
        # a warning will be issued and all values will be set
        # to unity!
        warn(f'All values are zero for {msg}! They are set to untiy. '
             'Result may be invalid!')
        arr[...] = 1.0
    if is_zero.any():
        warn(f'Some values are zero for {msg}!')


def get_range_delay_from_raw(
        raw: Raw,
        freq_band: str,
        txrx_pol: str
) -> float:
    """
    Get delay (seconds) of the second pulse wrt the pulsewidth
    of the first TX pulse in sequential split-spectrum transmit
    for a desired dataset in L0B.
    """
    # check if band is B and it is split spectrum
    if freq_band == 'B' and len(raw.frequencies) == 2:
        pols = raw.polarizations
        # check if this is sequential transmit
        if txrx_pol in pols['A']:
            sr_b = raw.getRanges('B', txrx_pol[0])
            sr_a = raw.getRanges('A', txrx_pol[0])
            delay = 2 * (sr_b.first - sr_a.first) / speed_of_light
            return delay
    return 0.0
