from __future__ import annotations
from warnings import warn
from typing import Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
import h5py

from nisar.products.readers.Raw import Raw
from nisar.antenna import get_calib_range_line_idx, CalPath
from isce3.core import speed_of_light


@dataclass(frozen=True)
class RxChannelImbalanceProduct:
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
        if self.lna_caltone_ratio.size != 12:
            warn('The size of LNA-CALTONE ratio is '
                 f'{self.lna_caltone_ratio.size} instead of 12!')


@unique
class PolarizationTypeId(IntEnum):
    """Enumeration for polarization types of L-band NISAR"""
    single_h = 0
    single_v = 1
    dual_h = 2
    dual_v = 3
    quad = 4
    compact = 5
    none = 6
    quasi_quad = 7
    quasi_dual = 8


def compute_all_rx_channel_imbalances_from_l0b(
        l0b_file: str | Raw,
        *,
        caltone_freq: float | None = None,
        freq_band: str | None = None,
        txrx_pol: str | None = None
) -> Dict[Tuple[str, str], RxChannelImbalanceProduct]:
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
    caltone_freq : float or None. Optional
        Caltone frequency in Hz.
        If None (default), it will be extracted from DRT in L0B.
    freq_band : str. Optional
        "A" or "B". Default is all.
    txrx_pol : str. Optional
        TR pol in `freq_band` such as "HH", "HV", etc.
        Default is all.

    Returns
    -------
    dict:
        A dict with keys (freq_band, txrx_pol) and values of type
        `RxChannelImbalanceProduct`

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
                raw=raw,
                freq_band=freq_band,
                txrx_pol=txrx_pol,
                caltone_freq=caltone_freq
            )
            out[freq_band, txrx_pol] = RxChannelImbalanceProduct(
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
        caltone_freq: float | None = None,
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
        Time delays from the phase of qFSP outlier in seconds
    max_ratio : float
        Report peak power among all channels used for amplitude
        normalization of RX channel imbalances.

    """
    lna_mean, n_tap_dominant = get_lna_cal_mean(raw, txrx_pol)
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


def polarization_type_from_drt(raw: Raw) -> PolarizationTypeId:
    """Get polarization ID and type from L0B DRT"""
    pol_path = f'{raw.TelemetryPath}/DRT/MISC/CP_IFSW_POLARIZATION'
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        try:
            ds_pol = f5[pol_path]
        except KeyError:
            warn(f'Missing dataset "{pol_path}" in "{raw.filename}"')
            id_pol = 6
        else:
            i_pol = ds_pol[()]
            id_pol = np.nanmedian(i_pol)
    return PolarizationTypeId(id_pol)


def is_raw_quad_pol(raw: Raw) -> bool:
    """Determine whether raw L0B is Quad or not"""
    return polarization_type_from_drt(raw) == PolarizationTypeId.quad


def parse_rangeline_index_from_hrt(
        raw: Raw,
        txrx_pol: str = None) -> np.ndarray | None:
    """
    Get range line index over all range lines from
    HRT if exists otherwise None!

    Returns
    -------
    np.ndarray(uint) or None
        If not available in L0b, None will be returned.

    """
    hrt_path = raw.TelemetryPath.replace('low', 'high')
    freq_band = sorted(raw.frequencies)[0]
    pols = raw.polarizations[freq_band]
    if txrx_pol is None:
        txrx_pol = pols[0]
    elif txrx_pol not in pols:
        raise ValueError(f'Available pols {pols} but got {txrx_pol}!')
    rgl_idx_path = (f'{hrt_path}/tx{txrx_pol[0]}/rx{txrx_pol[1]}/'
                    'RangeLine/RH_RANGELINE_INDEX')
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        try:
            ds_rgl_idx = f5[rgl_idx_path]
        except KeyError as err:
            warn(f'Can not parse range line index from HRT. Error -> {err}')
            return None
        else:
            return ds_rgl_idx[()]


def first_tx_pol_for_quad(raw: Raw) -> str:
    """Get first TX polarization, H or V, from only Quad pol product"""
    if not is_raw_quad_pol(raw):
        raise ValueError('Not a quad pol!')
    idx_rgl = parse_rangeline_index_from_hrt(raw)[0]
    # if not in HRT parse single-pol version from swath path
    if idx_rgl is None:
        idx_rgl_h = raw.getRangeLineIndex('A', 'H')[0]
        idx_rgl_v = raw.getRangeLineIndex('A', 'V')[0]
        if idx_rgl_v < idx_rgl_h:
            return 'V'
        return 'H'
    else:  # odd range line is V pol first and even is H pol first!
        return {0: 'H', 1: 'V'}.get(idx_rgl % 2)


def parse_chirpcorrelator_from_hrt_qfsp(
        raw: Raw,
        txrx_pol: str) -> np.ndarray | None:
    """
    Parse three-tap chirp correlator array with shape (lines, 12, 3)
    as well ass cal type with shape (lines,) from HRT QFSP.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    txrx_pol : str
        TxRx polarization such as HH, VH, etc

    Returns
    -------
    np.ndarray(complex) or None
        3-D complex array of chirp correlator with shape (Lines, channels, 3)
        If the field does not exist None will be returned.

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
                # loop over 3 taps per channel
                for i_tap in range(3):
                    n_tap = i_tap + 1
                    # form the path to the dataset per I and Q
                    # use RX pol!
                    p_ds_i = (f'{p_qfsp}/CHIRP_CORRELATOR_I{n_tap}_'
                              f'{txrx_pol[1]}{n_rx:02d}')
                    p_ds_q = (f'{p_qfsp}/CHIRP_CORRELATOR_Q{n_tap}_'
                              f'{txrx_pol[1]}{n_rx:02d}')
                    try:
                        ds_i = f5[p_ds_i]
                    except KeyError as err:
                        warn(
                            f'Missing dataset {p_ds_i} in {raw.filename}.'
                            f' Detailed error -> {err}'
                        )
                        return None
                    else:
                        # initialize the 3-D array, lines by 12 by 3
                        if i_qfsp == nn == i_tap == 0:
                            # initialize the 3-D array for chirp correlator
                            num_lines = ds_i.size
                            chp_cor = np.ones((num_lines, 12, 3), dtype='c8')
                        chp_cor[:, i_chn, i_tap].real = ds_i[()]
                        chp_cor[:, i_chn, i_tap].imag = f5[p_ds_q][()]
        return chp_cor


def parse_caltype_from_hrt_qfsp(
        raw: Raw,
        txrx_pol: str) -> np.ndarray | None:
    """
    Parse cal type with shape (lines,) from HRT QFSP.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    txrx_pol : str
        TxRx polarization such as HH, VH, etc

    Returns
    -------
    np.ndarray(uint8) or None
        1-D array of cal type w/ values HPA=0, LNA=1, BYPASS=2, and
        INVALID=255. If the field does not exist None will be returned.

    """
    # get HRT path
    hrt_path = raw.TelemetryPath.replace('low', 'high')
    qfsp_path = f'{hrt_path}/tx{txrx_pol[0]}/rx{txrx_pol[1]}/QFSP'
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        # XXX get caltype from the very first qFSP assuming
        # it is qFSP independent!
        i_qfsp = 0
        p_qfsp = f'{qfsp_path}{i_qfsp}'
        p_type = f'{p_qfsp}/CP_CAL_TYPE_{txrx_pol[1]}{i_qfsp}'
        # XXX Following Try/exception block is added to
        # support old sim L0B products lacking HRT!
        try:
            ds_cal_type = f5[p_type]
        except KeyError as err:
            warn(f'Missing dataset "{p_type}" in '
                 f'"{raw.filename}". Detailed error -> {err}')
            return None
        else:
            return ds_cal_type[()].astype(CalPath)


def _opposite_pol(pol: str) -> str:
    """Get the oppsoite pol"""
    if pol == 'H':
        return 'V'
    elif pol == 'V':
        return 'H'
    else:
        return pol


def chirpcorrelator_caltype_from_raw(
        raw: Raw,
        txrx_pol: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse three-tap chirp correlator array with shape (lines, 12, 3)
    as well ass cal type with shape (lines,) from Raw L0B for a certain
    TxRX pol

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    txrx_pol : str
        TxRx polarization such as HH, VH, etc

    Returns
    -------
    np.ndarray(complex)
        3-D complex array of chirp correlator with shape (Lines, channels, 3)
    np.ndarray(uint8)
        1-D array of cal type w/ values HPA=0, LNA=1, BYPASS=2, and INVALID=255

    """
    chp_cor = parse_chirpcorrelator_from_hrt_qfsp(raw, txrx_pol=txrx_pol)
    cal_type = parse_caltype_from_hrt_qfsp(raw, txrx_pol=txrx_pol)
    # XXX if the respective field does not exist then use co-pol under
    # swath in L0B for the sake of backward compatibility
    if chp_cor is None or cal_type is None:
        freq_band = [f for f in raw.frequencies if
                     txrx_pol in raw.polarizations[f]][0]
        chp_cor = raw.getChirpCorrelator(freq_band, txrx_pol[0])
        cal_type = raw.getCalType(freq_band, txrx_pol[0])
        return chp_cor, cal_type
    # Quad pol case
    if is_raw_quad_pol(raw):
        tx_pol_first = first_tx_pol_for_quad(raw)
        if txrx_pol[0] == tx_pol_first:
            chp_cor = chp_cor[::2]
            cal_type = cal_type[::2]
        else:  # the second TX pol
            # get data from the opssoite TX pol
            x_pol = _opposite_pol(txrx_pol[0]) + txrx_pol[1]
            chp_cor_x, cal_type_x = chirpcorrelator_caltype_from_raw(
                raw, txrx_pol=x_pol)
            # if co-pol get HPA value from same TX but
            # fill in LNA/BYP from oppsoite TX
            if txrx_pol[0] == txrx_pol[1]:
                chp_cor = chp_cor[1::2]
                cal_type = cal_type[1::2]
                _, idx_byp, idx_lna, _ = get_calib_range_line_idx(cal_type_x)
                chp_cor[idx_byp] = chp_cor_x[idx_byp]
                chp_cor[idx_lna] = chp_cor_x[idx_lna]
                cal_type[idx_byp] = CalPath.BYPASS
                cal_type[idx_lna] = CalPath.LNA
            else:  # x-pol product
                chp_cor = chp_cor_x
                cal_type = cal_type_x
    # set x-pol HPA to INVALID given they are the mix of
    # LNA from co-pol and HPA from x-pol!
    if txrx_pol in ('HV', 'VH'):
        idx_hpa, _, _, _ = get_calib_range_line_idx(cal_type)
        if idx_hpa.size > 0:
            warn(f'Set HPA cal type for x-pol {txrx_pol} to INVALID!')
            cal_type[idx_hpa] = CalPath.INVALID
    return chp_cor, cal_type


def get_lna_cal_mean(
    raw: Raw,
    txrx_pol: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mean complex LNA values and dominant tap
    numbers within [1, 2, 3] for all channels
    """
    chp_cor, cal_type = chirpcorrelator_caltype_from_raw(
        raw=raw,
        txrx_pol=txrx_pol
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
        caltone_freq: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    # Get caltone frequency from DRT if not provided
    if caltone_freq is None:
        caltone_freq = parse_caltone_freq_from_drt(raw, txrx_pol)
        warn(f'Caltone frequency is extracted from {txrx_pol[1]}-pol DRT '
             f'-> {caltone_freq * 1e-6:.3f} (MHz)')
    # Check if product from the second band so we can modify
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
        # if there is then get diff of frequency bands A and B
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


def _get_qfsp_delay_anomaly(
        lna_caltone_ratio: np.ndarray,
        dif_chirp_caltone_freq: float,
        adc_clock: float = 240e6) -> np.ndarray:
    """
    If the product is a 12-channel NISAR L-band product,
    return the time delays for a qFSP with phase anomaly.
    Else, return zeros.
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


def parse_caltone_freq_from_drt(
        raw: Raw,
        txrx_pol: str
) -> float:
    """get caltone frequency in Hz from low rate telemetry in L0B."""
    # default caltone if dataset is not available (Hz)
    default = 1214.88e6
    # frequency of local oscillator (Hz)
    lo = 1200e6
    # ADC clock (Hz)
    clock = 240e6
    c_p = (f'{raw.TelemetryPath}/DRT/MISC/CP_IFSW_CALTONE_PHASE_STEP_'
           f'{txrx_pol[1]}')
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        try:
            ds_caltone_phase = f5[c_p]
        except KeyError:
            warn(f'Missing path "{c_p}" in L0B! Caltone frequency will '
                 f'be set to {default} (Hz)')
            return default
        else:
            i_cal = np.median(ds_caltone_phase[()]).astype(int)
            caltone_freq = (i_cal / 2**32) * clock + lo
            return caltone_freq
