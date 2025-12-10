import numpy as np

from scipy.signal import detrend
from scipy.ndimage import convolve1d


def _Symm_Asym(data: np.ndarray):
    """
    Compute symmetric and asymmetric components against equator of data.

    Parameters
    ----------
    data : np.ndarray, with shape (time, lat, lon)

    Returns
    -------
    symm : np.ndarray
        Symmetric component of data.
    asym : np.ndarray
        Asymmetric component of data.
    """

    if len(data.shape) != 3:
        raise Exception("data must have shape (time, lat, lon)")

    symm = (data + np.flip(data, axis=1)) / 2
    asym = (data - np.flip(data, axis=1)) / 2

    return symm, asym


def _Windowing(data: np.ndarray, n_window: int = 96, n_overlap: int = 48):
    """
    Windowing data for power spectrum calculation.

    Parameters
    ----------
    data : np.ndarray
        Data to be windowed, the shape of array needs to be (time, lat, lon).
    n_window : int, optional
        Window size. Defaults to 96.
    n_overlap : int, optional
        Number of overlapping samples. Defaults to 48.

    Returns
    -------
    chunked_data : np.ndarray
        Windowed data.

    Notes
    -----
    Data is divided into chunks of size n_window. Each chunk is then
    detrended and multiplied by a Hanning window. The resulting
    chunks are then stacked along the first axis.

    """
    if len(data.shape) != 3:
        raise Exception("data must have shape (time, lat, lon)")

    chunks = []  # pre-allocate array for chunked data
    hanning = np.hanning(n_window)[:, None, None]

    for i in range(data.shape[0] // n_window):

        start = i * n_overlap
        end = start + n_window

        chunked = detrend(data[start:end], axis=0) * hanning

        chunks.append(chunked)

    return np.stack(chunks, axis=0)


def compute_spectrum(data: np.ndarray):
    """
    Compute power spectrum of data.

    Parameters
    ----------
    data : np.ndarray
        Data to be computed power spectrum, shape (time, lat, lon).

    Returns
    -------
    ps : np.ndarray
        Power spectrum of data.

    Notes
    -----
    Data is first transformed to frequency domain using FFT. Then,
    the power spectrum is computed by multiplying the FFT of data by
    its conjugate and dividing by the product of the number of time
    samples and the number of lat samples.

    """
    if len(data.shape) != 3:
        raise Exception("data must have shape (time, lat, lon)")

    data_fft = np.fft.fft(data, axis=0)
    data_fft = np.fft.ifft(data_fft, axis=2) * data.shape[2]

    ps = (data_fft * data_fft.conj()) / (data.shape[0] * data.shape[2]) ** 2.0

    return ps


def background(
    data: np.ndarray,
    kernel: np.ndarray = np.array([1, 2, 1]) / 4.0,
    f_running: int = 10,
    low_k_running: int = 10,
    high_k_running: int = 40,
):
    """
    Compute background power spectrum of data.

    Parameters
    ----------
    data : np.ndarray
        Data to be computed background power spectrum, shape (time, lat, lon).
    kernel : np.ndarray, optional
        Kernel to use for running mean, default is np.array([1, 2, 1]) / 4.0.
    f_running : int, optional
        Number of times to run the filter in the time direction, default is 10.
    low_k_running : int, optional
        Number of times to run the filter in the low lat direction, default is 10.
    high_k_running : int, optional
        Number of times to run the filter in the high lat direction, default is 40.

    Returns
    -------
    background : np.ndarray
        Background power spectrum of data.

    Notes
    -----
    The background power spectrum is computed by running a mean filter
    in the time and lat directions.

    """
    data = data.copy()

    kernel = kernel

    half_freq = data.shape[0] // 2

    for i in range(f_running):
        data = convolve1d(data, kernel, axis=0, mode="reflect")

    for i in range(low_k_running):
        data[:half_freq] = convolve1d(data[:half_freq], kernel, axis=1, mode="reflect")

    for i in range(high_k_running):
        data[half_freq:] = convolve1d(data[half_freq:], kernel, axis=1, mode="reflect")

    return data


def output(data: np.ndarray, n_window: int = 96, n_overlap: int = 48):
    """
    Compute symmetric and asymmetric power spectrum of data and remove background.

    Parameters
    ----------
    data : np.ndarray
        Data to be computed power spectrum, shape (time, lat, lon).
    n_window : int, optional
        Window size. Defaults to 96.
    n_overlap : int, optional
        Number of overlapping samples. Defaults to 48.

    Returns
    -------
    symm_ps : np.ndarray
        Symmetric power spectrum of data.
    asym_ps : np.ndarray
        Asymmetric power spectrum of data.
    symm_ps_rm_bg : np.ndarray
        Symmetric power spectrum of data after removing background.
    asym_ps_rm_bg : np.ndarray
        Asymmetric power spectrum of data after removing background.
    wn : np.ndarray
        Wavenumber of data.
    fr : np.ndarray
        Frequency of data.

    Notes
    -----
    The power spectrum is computed by windowing the data, computing the
    symmetric and asymmetric components, and then computing the power
    spectrum of each component.

    The background power spectrum is computed by running a mean filter
    in the time and lat directions.

    The power spectrum after removing the background is computed by
    dividing the power spectrum by the background power spectrum.
    """
    if len(data.shape) != 3:
        raise Exception("data must have shape (time, lat, lon)")

    symm, asym = _Symm_Asym(data)

    symm_chunk = _Windowing(symm, n_window, n_overlap)
    asym_chunk = _Windowing(asym, n_window, n_overlap)

    symm_ps, asym_ps = [], []

    # calculate wavenumber and frequency
    wn = np.fft.fftshift(np.fft.fftfreq(data.shape[-1], d=1 / data.shape[-1]))
    fr = np.fft.fftshift(np.fft.fftfreq(n_window, d=1))

    for i in range(symm_chunk.shape[0]):
        symm_ps.append(compute_spectrum(symm_chunk[i]))
        asym_ps.append(compute_spectrum(asym_chunk[i]))

    symm_ps = np.fft.fftshift(np.stack(symm_ps, axis=0).mean(axis=0).real.sum(axis=1))[fr > 0]*2.0
    asym_ps = np.fft.fftshift(np.stack(asym_ps, axis=0).mean(axis=0).real.sum(axis=1))[fr > 0]*2.0

    bg_ps = background((symm_ps + asym_ps) / 2)

    symm_ps_rm_bg = symm_ps / bg_ps
    asym_ps_rm_bg = asym_ps / bg_ps

    

    return symm_ps, asym_ps, symm_ps_rm_bg, asym_ps_rm_bg, wn, fr[fr > 0]
